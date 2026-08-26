import gc
import json
from pathlib import Path
from typing import Optional

import holidays
import numpy as np
import pandas as pd
import tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import (
    impute,
    roll_time_series,
)


class FeatureEngineer:
    """Create forecasting features and a single future target.

    Each output row follows this time convention:

    - ``ds`` is forecast-origin time ``t``.
    - ``target_time`` is ``t + horizon``.
    - ``target_t{horizon}`` is the observed value at ``target_time``.

    TSFresh features are calculated from rolling windows ending at ``ds``.
    Calendar features are calculated for ``target_time``.
    """

    def __init__(
        self,
        data_file: str | Path,
        target_column: str,
        time_column: str,
        time_format: str,
        start: str,
        end: str,
        freq: str = "15min",
        roll_sizes: Optional[list[int]] = None,
        roll_feat_params: Optional[dict[int, dict]] = None,
        horizon: int = 96,
        series_id: str = "household",
        target_start: Optional[str] = None,
        target_end: Optional[str] = None,
        split_name: str = "dataset",
    ) -> None:
        if horizon <= 0:
            raise ValueError(
                "horizon must be a positive integer."
            )

        if horizon > 96:
            raise ValueError(
                "horizon must not exceed 96 when lag_96 is used."
            )

        if not roll_sizes:
            raise ValueError(
                "roll_sizes must contain at least one window size."
            )

        if any(roll_size <= 0 for roll_size in roll_sizes):
            raise ValueError(
                "All rolling-window sizes must be positive integers."
            )

        if len(roll_sizes) != len(set(roll_sizes)):
            raise ValueError(
                "roll_sizes must not contain duplicate values."
            )

        if roll_feat_params is None:
            raise ValueError(
                "roll_feat_params must contain parameters "
                "for every window size."
            )

        missing_parameter_sets = sorted(
            set(roll_sizes).difference(roll_feat_params)
        )

        if missing_parameter_sets:
            raise ValueError(
                "Missing TSFresh parameters for rolling-window sizes: "
                f"{missing_parameter_sets}."
            )

        self.data_file = Path(data_file)
        self.target_column = target_column
        self.time_column = time_column
        self.time_format = time_format
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.freq = freq
        self.roll_sizes = list(roll_sizes)
        self.roll_feat_params = roll_feat_params
        self.horizon = horizon
        self.series_id = series_id
        self.split_name = split_name

        self.target_start = (
            pd.Timestamp(target_start)
            if target_start is not None
            else None
        )

        self.target_end = (
            pd.Timestamp(target_end)
            if target_end is not None
            else None
        )

        if self.start > self.end:
            raise ValueError(
                "start must not be later than end."
            )

        if (
            self.target_start is not None
            and self.target_end is not None
            and self.target_start >= self.target_end
        ):
            raise ValueError(
                "target_start must be earlier than target_end."
            )

    @property
    def target_name(self) -> str:
        """Return the target-column name."""
        return f"target_t{self.horizon}"

    def _read_and_preprocess_main(self) -> pd.DataFrame:
        """Read and preprocess the input time series."""
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.data_file}"
            )

        dataframe = pd.read_csv(self.data_file)

        required_columns = {
            self.time_column,
            self.target_column,
        }

        missing_columns = required_columns.difference(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}. "
                f"Available columns: {dataframe.columns.tolist()}"
            )

        dataframe = dataframe[
            [self.time_column, self.target_column]
        ].copy()

        dataframe.rename(
            columns={
                self.time_column: "ds",
                self.target_column: "y",
            },
            inplace=True,
        )

        dataframe["ds"] = pd.to_datetime(
            dataframe["ds"],
            format=self.time_format,
            errors="raise",
        )

        dataframe.set_index("ds", inplace=True)
        dataframe.sort_index(inplace=True)

        dataframe = dataframe.loc[
            self.start:self.end
        ]

        if dataframe.empty:
            raise ValueError(
                "No observations were found in the selected period."
            )

        duplicate_count = int(
            dataframe.index.duplicated().sum()
        )

        if duplicate_count > 0:
            print(
                f"Warning: found {duplicate_count} duplicate "
                "timestamps. Duplicate rows will be averaged."
            )

            dataframe = dataframe.groupby(
                level=0
            ).mean(numeric_only=True)

        full_index = pd.date_range(
            start=self.start,
            end=self.end,
            freq=self.freq,
        )

        dataframe = dataframe.reindex(full_index)
        dataframe.index.name = "ds"

        missing_before = int(
            dataframe["y"].isna().sum()
        )

        if missing_before > 0:
            print(
                f"Interpolating {missing_before} missing "
                "measurements."
            )

        dataframe["y"] = dataframe["y"].interpolate(
            method="linear",
            limit_direction="both",
        )

        missing_after = int(
            dataframe["y"].isna().sum()
        )

        if missing_after > 0:
            raise ValueError(
                f"{missing_after} missing values remain "
                "after interpolation."
            )

        dataframe = dataframe.reset_index()
        dataframe["unique_id"] = self.series_id

        return dataframe

    def create_targets(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create target times and future target values."""
        targets = dataframe[["ds"]].copy()

        targets["target_time"] = dataframe["ds"].shift(
            -self.horizon
        )

        targets[self.target_name] = dataframe["y"].shift(
            -self.horizon
        )

        targets = (
            targets
            .dropna(
                subset=[
                    "target_time",
                    self.target_name,
                ]
            )
            .reset_index(drop=True)
        )

        return targets

    def _extract_tsfresh_features(
        self,
        dataframe: pd.DataFrame,
        roll_size: int,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Extract TSFresh features for one rolling window."""
        rolled_dataframe = roll_time_series(
            dataframe,
            column_id="unique_id",
            column_sort="ds",
            max_timeshift=roll_size - 1,
            min_timeshift=roll_size - 1,
            n_jobs=n_jobs,
        )

        feature_parameters = self.roll_feat_params[
            roll_size
        ]

        features = extract_features(
            rolled_dataframe,
            column_id="id",
            column_sort="ds",
            column_value="y",
            default_fc_parameters=feature_parameters,
            impute_function=impute,
            n_jobs=n_jobs,
        )

        del rolled_dataframe
        gc.collect()

        if not isinstance(
            features.index,
            pd.MultiIndex,
        ):
            raise ValueError(
                "TSFresh did not return the expected "
                "two-level index."
            )

        features.index = features.index.set_names(
            ["unique_id", "ds"]
        )

        features = features.reset_index()

        if features.duplicated(
            subset=["unique_id", "ds"]
        ).any():
            raise ValueError(
                f"Duplicate rows were generated for window "
                f"{roll_size}."
            )

        feature_columns = [
            column
            for column in features.columns
            if column not in {"ds", "unique_id"}
        ]

        features.rename(
            columns={
                column: f"r{roll_size}_{column}"
                for column in feature_columns
            },
            inplace=True,
        )

        return features

    def _add_manual_features(
        self,
        features: pd.DataFrame,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add calendar features and target-relative lags."""
        features = features.copy()

        features["ds"] = pd.to_datetime(
            features["ds"]
        )

        time_step = pd.Timedelta(self.freq)

        target_time = (
            features["ds"]
            + self.horizon * time_step
        )

        features["hour_cos"] = np.cos(
            2 * np.pi * target_time.dt.hour / 24
        )

        features["hour_sin"] = np.sin(
            2 * np.pi * target_time.dt.hour / 24
        )

        features["day_of_week_sin"] = np.sin(
            2 * np.pi * target_time.dt.dayofweek / 7
        )

        features["day_of_week_cos"] = np.cos(
            2 * np.pi * target_time.dt.dayofweek / 7
        )

        features["month_sin"] = np.sin(
            2
            * np.pi
            * (target_time.dt.month - 1)
            / 12
        )

        features["month_cos"] = np.cos(
            2
            * np.pi
            * (target_time.dt.month - 1)
            / 12
        )

        days_in_year = np.where(
            target_time.dt.is_leap_year,
            366,
            365,
        )

        features["day_of_year_sin"] = np.sin(
            2
            * np.pi
            * target_time.dt.dayofyear
            / days_in_year
        )

        features["day_of_year_cos"] = np.cos(
            2
            * np.pi
            * target_time.dt.dayofyear
            / days_in_year
        )

        features["is_weekend"] = (
            target_time.dt.dayofweek >= 5
        ).astype(int)

        features["is_sunday"] = (
            target_time.dt.dayofweek == 6
        ).astype(int)

        holiday_years = sorted(
            int(year)
            for year in target_time.dt.year.unique()
        )

        slovenian_holidays = holidays.SI(
            years=holiday_years
        )

        features["is_holiday"] = (
            target_time.dt.date.map(
                lambda date: int(
                    date in slovenian_holidays
                )
            )
        )

        target_series = (
            dataframe
            .set_index("ds")["y"]
            .sort_index()
        )

        # lag_96 is defined relative to target time.
        daily_shift = 96 - self.horizon

        if daily_shift < 0:
            raise ValueError(
                "The forecast horizon must not exceed 96 "
                "when lag_96 is used."
            )

        features["lag_96"] = (
            target_series
            .shift(daily_shift)
            .reindex(features["ds"])
            .to_numpy()
        )

        # lag_672 is also defined relative to target time.
        weekly_shift = 672 - self.horizon

        if weekly_shift < 0:
            raise ValueError(
                "The forecast horizon must not exceed 672 "
                "when lag_672 is used."
            )

        features["lag_672"] = (
            target_series
            .shift(weekly_shift)
            .reindex(features["ds"])
            .to_numpy()
        )

        return features

    def create_features(
        self,
        dataframe: pd.DataFrame,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Create and merge features for all rolling windows."""
        combined_features = None

        for roll_size in self.roll_sizes:
            print(
                f"Extracting TSFresh features for window "
                f"{roll_size}."
            )

            current_features = (
                self._extract_tsfresh_features(
                    dataframe=dataframe,
                    roll_size=roll_size,
                    n_jobs=n_jobs,
                )
            )

            if combined_features is None:
                combined_features = current_features
            else:
                combined_features = pd.merge(
                    combined_features,
                    current_features,
                    on=["unique_id", "ds"],
                    how="inner",
                    validate="one_to_one",
                    sort=False,
                )

                del current_features
                gc.collect()

            print(
                f"Current combined shape: "
                f"{combined_features.shape}"
            )

        if combined_features is None:
            raise RuntimeError(
                "No feature tables were created."
            )

        combined_features = self._add_manual_features(
            features=combined_features,
            dataframe=dataframe,
        )

        return combined_features

    def save_run_info(
        self,
        dataset_path: str | Path,
        dataset_shape: tuple[int, int],
    ) -> None:
        """Append information about the run to a JSON file."""
        run_info = {
            "dataset_path": str(dataset_path),
            "dataset_rows": int(dataset_shape[0]),
            "dataset_columns": int(dataset_shape[1]),
            "data_file": str(self.data_file),
            "series_id": self.series_id,
            "split_name": self.split_name,
            "target_column": self.target_column,
            "target_name": self.target_name,
            "target_start": (
                self.target_start.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if self.target_start is not None
                else None
            ),
            "target_end_exclusive": (
                self.target_end.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if self.target_end is not None
                else None
            ),
            "time_column": self.time_column,
            "time_format": self.time_format,
            "start": self.start.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": self.end.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "freq": self.freq,
            "horizon": self.horizon,
            "roll_sizes": self.roll_sizes,
            "roll_feat_params": {
                str(roll_size): (
                    self.roll_feat_params[
                        roll_size
                    ].__class__.__name__
                )
                for roll_size in self.roll_sizes
            },
            "package_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "tsfresh": tsfresh.__version__,
            },
            "timestamp": pd.Timestamp.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        run_info_path = Path(
            "data/feature_engineer_runs.json"
        )

        run_info_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            with run_info_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                existing_runs = json.load(file)

            if not isinstance(existing_runs, list):
                existing_runs = [existing_runs]

        except (
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            existing_runs = []

        existing_runs.append(run_info)

        with run_info_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                existing_runs,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def create_dataset(
        self,
        save_path: Optional[str | Path] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Create and save the complete feature dataset."""
        dataframe = self._read_and_preprocess_main()

        targets = self.create_targets(dataframe)

        features = self.create_features(
            dataframe=dataframe,
            n_jobs=n_jobs,
        )

        dataset = pd.merge(
            features,
            targets,
            on="ds",
            how="inner",
            validate="one_to_one",
            sort=False,
        )

        del features
        del targets
        del dataframe
        gc.collect()

        # Assign rows according to target time.
        if self.target_start is not None:
            dataset = dataset.loc[
                dataset["target_time"]
                >= self.target_start
            ]

        if self.target_end is not None:
            dataset = dataset.loc[
                dataset["target_time"]
                < self.target_end
            ]

        dataset = (
            dataset
            .dropna(
                subset=[
                    "lag_96",
                    "lag_672",
                    self.target_name,
                ]
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if dataset.empty:
            raise ValueError(
                "The final dataset is empty after filtering."
            )

        if save_path is None:
            roll_sizes_text = "_".join(
                str(roll_size)
                for roll_size in self.roll_sizes
            )

            save_path = (
                Path("data/features")
                / (
                    f"{self.series_id}_"
                    f"{self.split_name}_features_"
                    f"{roll_sizes_text}_"
                    f"{self.target_name}.pkl"
                )
            )

        output_path = Path(save_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_pickle(output_path)

        self.save_run_info(
            dataset_path=output_path,
            dataset_shape=dataset.shape,
        )

        return dataset


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    # Select "train", "validation", or "test".
    DATASET_SPLIT = "test"

    # Add additional household identifiers when needed,
    # for example: [1, 2, 3, 4].
    HOUSEHOLD_IDS = [1]

    ROLL_SIZES = [
        4,
        8,
        12,
        16,
        24,
        32,
        48,
        96,
        672,
    ]

    FREQUENCY = "15min"
    HORIZON = 96
    N_JOBS = 1

    split_aliases = {
        "val": "validation",
    }

    dataset_split = split_aliases.get(
        DATASET_SPLIT,
        DATASET_SPLIT,
    )

    if dataset_split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "DATASET_SPLIT must be "
            "'train', 'validation', or 'test'."
        )

    roll_feat_params = {
        roll_size: EfficientFCParameters()
        for roll_size in ROLL_SIZES
    }

    time_step = pd.Timedelta(FREQUENCY)

    # The end timestamp of each period is included when reading.
    split_periods = {
        "train": {
            "start": pd.Timestamp(
                "2024-01-01 00:15:00"
            ),
            "end": pd.Timestamp(
                "2025-06-29 23:45:00"
            ),
        },
        "validation": {
            "start": pd.Timestamp(
                "2025-07-01 00:00:00"
            ),
            "end": pd.Timestamp(
                "2025-09-29 23:45:00"
            ),
        },
        "test": {
            "start": pd.Timestamp(
                "2025-10-01 00:00:00"
            ),
            "end": pd.Timestamp(
                "2026-01-01 00:15:00"
            ),
        },
    }

    selected_period = split_periods[
        dataset_split
    ]

    processing_start = selected_period["start"]
    processing_end = selected_period["end"]

    target_start = processing_start
    target_end = processing_end + time_step

    roll_sizes_text = "_".join(
        str(roll_size)
        for roll_size in ROLL_SIZES
    )

    for household_id in HOUSEHOLD_IDS:
        dataset_name = f"df_{household_id}"

        input_path = (
            Path("data/processed")
            / f"{dataset_name}_{dataset_split}.csv"
        )

        output_path = (
            Path("data/features")
            / (
                f"{dataset_name}_"
                f"{dataset_split}_features_"
                f"{roll_sizes_text}_"
                f"target_t{HORIZON}.pkl"
            )
        )

        print()
        print(
            f"=== Processing {input_path} ==="
        )
        print(
            f"Output file: {output_path}"
        )

        feature_engineer = FeatureEngineer(
            data_file=input_path,
            target_column="Energija A+",
            time_column="datetime",
            time_format="%Y-%m-%d %H:%M:%S",
            start=str(processing_start),
            end=str(processing_end),
            freq=FREQUENCY,
            roll_sizes=ROLL_SIZES,
            roll_feat_params=roll_feat_params,
            horizon=HORIZON,
            series_id=dataset_name,
            target_start=str(target_start),
            target_end=str(target_end),
            split_name=dataset_split,
        )

        selected_dataset = (
            feature_engineer.create_dataset(
                save_path=output_path,
                n_jobs=N_JOBS,
            )
        )

        print(f"Saved: {output_path}")
        print(f"Shape: {selected_dataset.shape}")

        del selected_dataset
        del feature_engineer
        gc.collect()