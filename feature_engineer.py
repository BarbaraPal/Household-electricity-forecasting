import json
from functools import reduce
from pathlib import Path
from typing import Optional

import holidays
import numpy as np
import pandas as pd
import tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute, roll_time_series


class FeatureEngineer:
    """Create forecasting features and a single future target.

    Each output row follows this time convention:

    - ``ds`` is forecast-origin time ``t`` and therefore the time of the last
      observation available when the forecast is made.
    - ``target_time`` is ``t + horizon``.
    - ``target_t{horizon}`` is the observed value at ``target_time``.

    TSFresh features are calculated from rolling windows ending at ``ds``.
    Calendar features are calculated for ``target_time``. Lag names are
    defined relative to ``target_time`` and use only observations available
    no later than ``ds``.
    """

    def __init__(
        self,
        data_file: str,
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
        """Initialize the feature-engineering pipeline.

        Parameters
        ----------
        data_file : str
            Path to the input CSV file.
        target_column : str
            Name of the electricity-consumption column.
        time_column : str
            Name of the timestamp column.
        time_format : str
            Datetime format used in the input CSV file.
        start, end : str
            First and last timestamps included in the input period.
        freq : str
            Sampling frequency of the time series.
        roll_sizes : list[int]
            Rolling-window lengths expressed in numbers of time steps.
        roll_feat_params : dict[int, dict]
            Mapping from each rolling-window length to TSFresh feature
            extraction parameters.
        horizon : int
            Forecast horizon expressed in numbers of time steps.
        series_id : str
            Identifier of the household time series.
        target_start : str, optional
            Inclusive lower boundary for ``target_time``.
        target_end : str, optional
            Exclusive upper boundary for ``target_time``.
        split_name : str
            Name of the generated data subset.
        """
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        if not roll_sizes:
            raise ValueError("roll_sizes must contain at least one window size.")

        if any(roll_size <= 0 for roll_size in roll_sizes):
            raise ValueError("All rolling-window sizes must be positive integers.")

        if roll_feat_params is None:
            raise ValueError(
                "roll_feat_params must contain parameters for every window size."
            )

        missing_parameter_sets = sorted(
            set(roll_sizes).difference(roll_feat_params)
        )

        if missing_parameter_sets:
            raise ValueError(
                "Missing TSFresh parameters for rolling-window sizes: "
                f"{missing_parameter_sets}."
            )

        self.data_file = data_file
        self.target_column = target_column
        self.time_column = time_column
        self.time_format = time_format
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.freq = freq
        self.roll_sizes = roll_sizes
        self.roll_feat_params = roll_feat_params
        self.horizon = horizon
        self.series_id = series_id
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
        self.split_name = split_name

        if (
            self.target_start is not None
            and self.target_end is not None
            and self.target_start >= self.target_end
        ):
            raise ValueError("target_start must be earlier than target_end.")

    @property
    def target_name(self) -> str:
        """Return the name of the target column."""
        return f"target_t{self.horizon}"

    def _read_and_preprocess_main(self) -> pd.DataFrame:
        """Read, regularize, and preprocess the input time series."""
        dataframe = pd.read_csv(self.data_file)

        required_columns = {self.time_column, self.target_column}
        missing_columns = required_columns.difference(dataframe.columns)

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

        # Timestamps have already been converted to fixed CET (UTC+1).
        dataframe["ds"] = pd.to_datetime(
            dataframe["ds"],
            format=self.time_format,
            errors="raise",
        )

        dataframe.set_index("ds", inplace=True)
        dataframe.sort_index(inplace=True)

        # Restrict the dataframe to the requested period.
        dataframe = dataframe.loc[self.start:self.end]

        duplicate_count = int(dataframe.index.duplicated().sum())

        if duplicate_count > 0:
            print(
                f"Warning: found {duplicate_count} duplicate timestamps. "
                "Duplicate rows will be averaged."
            )
            dataframe = dataframe.groupby(level=0).mean(numeric_only=True)

        full_index = pd.date_range(
            start=self.start,
            end=self.end,
            freq=self.freq,
        )

        dataframe = dataframe.reindex(full_index)
        dataframe.index.name = "ds"

        missing_before = int(dataframe["y"].isna().sum())

        if missing_before > 0:
            print(
                f"Interpolating {missing_before} missing measurements."
            )

        dataframe["y"] = dataframe["y"].interpolate(
            method="linear",
            limit_direction="both",
        )

        missing_after = int(dataframe["y"].isna().sum())

        if missing_after > 0:
            raise ValueError(
                f"{missing_after} missing values remain after interpolation."
            )

        dataframe = dataframe.reset_index()
        dataframe["unique_id"] = self.series_id

        return dataframe

    def create_targets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create ``target_time`` and the value observed at that time.

        ``ds`` remains forecast-origin time ``t``. The target value in the same
        row is the observation ``horizon`` steps after ``ds``.
        """
        targets = dataframe[["ds"]].copy()

        targets["target_time"] = dataframe["ds"].shift(
            -self.horizon
        )
        targets[self.target_name] = dataframe["y"].shift(
            -self.horizon
        )

        return (
            targets
            .dropna(subset=["target_time", self.target_name])
            .reset_index(drop=True)
        )

    def _extract_tsfresh_features(
        self,
        dataframe: pd.DataFrame,
        roll_size: int,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Extract TSFresh features for one rolling-window length."""
        rolled_dataframe = roll_time_series(
            dataframe,
            column_id="unique_id",
            column_sort="ds",
            max_timeshift=roll_size - 1,
            min_timeshift=roll_size - 1,
            n_jobs=n_jobs,
        )

        feature_parameters = self.roll_feat_params[roll_size]

        features = extract_features(
            rolled_dataframe,
            column_id="id",
            column_sort="ds",
            column_value="y",
            default_fc_parameters=feature_parameters,
            impute_function=impute,
            n_jobs=n_jobs,
        )

        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError(
                "TSFresh did not return the expected two-level index."
            )

        features.index = features.index.set_names(
            ["unique_id", "ds"]
        )
        features = features.reset_index()

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
        """Add target-time calendar features and target-relative lags."""
        features = features.copy()
        features["ds"] = pd.to_datetime(features["ds"])

        # ds is time t; calendar features describe target time t + horizon.
        target_time = (
            features["ds"]
            + self.horizon * pd.Timedelta(self.freq)
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
            2 * np.pi * (target_time.dt.month - 1) / 12
        )
        features["month_cos"] = np.cos(
            2 * np.pi * (target_time.dt.month - 1) / 12
        )

        days_in_year = np.where(
            target_time.dt.is_leap_year,
            366,
            365,
        )

        features["day_of_year_sin"] = np.sin(
            2 * np.pi * target_time.dt.dayofyear / days_in_year
        )
        features["day_of_year_cos"] = np.cos(
            2 * np.pi * target_time.dt.dayofyear / days_in_year
        )

        features["is_weekend"] = (
            target_time.dt.dayofweek >= 5
        ).astype(int)
        features["is_sunday"] = (
            target_time.dt.dayofweek == 6
        ).astype(int)

        holiday_years = sorted(
            int(year) for year in target_time.dt.year.unique()
        )
        slovenian_holidays = holidays.SI(years=holiday_years)

        features["is_holiday"] = target_time.dt.date.map(
            lambda date: int(date in slovenian_holidays)
        )

        # Lag names are defined relative to target time t + horizon.
        target_series = (
            dataframe
            .set_index("ds")["y"]
            .sort_index()
        )

        # y_t = y_((t + horizon) - horizon)
        features["lag_96"] = (
            target_series
            .reindex(features["ds"])
            .to_numpy()
        )

        # y_(t - 576) = y_((t + 96) - 672)
        weekly_shift = 672 - self.horizon

        if weekly_shift < 0:
            raise ValueError(
                "horizon must not exceed 672 when lag_672 is used."
            )

        features["lag_672"] = (
            target_series.shift(weekly_shift)
            .reindex(features["ds"])
            .to_numpy()
        )

        return features

    def create_features(
        self,
        dataframe: pd.DataFrame,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Create TSFresh and manually defined features."""
        feature_dataframes = []

        for roll_size in self.roll_sizes:
            print(
                f"Extracting TSFresh features for window {roll_size}."
            )
            extracted_features = self._extract_tsfresh_features(
                dataframe,
                roll_size,
                n_jobs=n_jobs,
            )
            feature_dataframes.append(extracted_features)

        features = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["ds", "unique_id"],
                how="inner",
            ),
            feature_dataframes,
        )

        return self._add_manual_features(features, dataframe)

    def save_run_info(self, dataset_path: str) -> None:
        """Append information about the current run to a JSON file."""
        run_info = {
            "dataset_path": dataset_path,
            "data_file": self.data_file,
            "series_id": self.series_id,
            "split_name": self.split_name,
            "target_column": self.target_column,
            "target_name": self.target_name,
            "target_start": (
                self.target_start.strftime("%Y-%m-%d %H:%M:%S")
                if self.target_start is not None
                else None
            ),
            "target_end_exclusive": (
                self.target_end.strftime("%Y-%m-%d %H:%M:%S")
                if self.target_end is not None
                else None
            ),
            "time_column": self.time_column,
            "time_format": self.time_format,
            "start": self.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": self.end.strftime("%Y-%m-%d %H:%M:%S"),
            "freq": self.freq,
            "horizon": self.horizon,
            "roll_sizes": self.roll_sizes,
            "roll_feat_params": {
                str(roll_size): parameters.__class__.__name__
                for roll_size, parameters
                in self.roll_feat_params.items()
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

        run_info_path = Path("data/feature_engineer_runs.json")
        run_info_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with run_info_path.open("r", encoding="utf-8") as file:
                existing_runs = json.load(file)

            if not isinstance(existing_runs, list):
                existing_runs = [existing_runs]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_runs = []

        existing_runs.append(run_info)

        with run_info_path.open("w", encoding="utf-8") as file:
            json.dump(
                existing_runs,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def create_dataset(
        self,
        save_path: Optional[str] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Create, merge, and save features and the forecasting target."""
        dataframe = self._read_and_preprocess_main()

        targets = self.create_targets(dataframe)
        features = self.create_features(dataframe, n_jobs=n_jobs)

        dataset = pd.merge(
            features,
            targets,
            on="ds",
            how="inner",
            validate="one_to_one",
        )

        # Assign rows to a subset according to target time, not forecast time.
        if self.target_start is not None:
            dataset = dataset.loc[
                dataset["target_time"] >= self.target_start
            ]

        if self.target_end is not None:
            dataset = dataset.loc[
                dataset["target_time"] < self.target_end
            ]

        # Remove initial rows for which one or more specified lags do not exist.
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

        if save_path is None:
            roll_sizes_text = "_".join(
                str(roll_size) for roll_size in self.roll_sizes
            )
            save_path = (
                f"data/features/"
                f"{self.series_id}_{self.split_name}_features_"
                f"{roll_sizes_text}_{self.target_name}.pkl"
            )

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.to_pickle(output_path)
        self.save_run_info(str(output_path))

        return dataset


if __name__ == "__main__":
    # Select exactly one option: "train", "validation", or "test".
    dataset_split = "train"

    split_aliases = {
        "val": "validation",
    }
    dataset_split = split_aliases.get(
        dataset_split,
        dataset_split,
    )

    if dataset_split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "dataset_split must be "
            "'train', 'validation', or 'test'."
        )

    # Rolling-window lengths used in the experiments.
    roll_sizes = [
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

    roll_feat_params = {
        roll_size: EfficientFCParameters()
        for roll_size in roll_sizes
    }

    frequency = "15min"
    horizon = 96
    time_step = pd.Timedelta(frequency)

    # Each subset is read from its own isolated CSV file.
    split_periods = {
        "train": {
            "start": pd.Timestamp("2024-01-01 00:15:00"),
            "end": pd.Timestamp("2025-06-29 23:45:00"),
        },
        "validation": {
            "start": pd.Timestamp("2025-07-01 00:00:00"),
            "end": pd.Timestamp("2025-09-29 23:45:00"),
        },
        "test": {
            "start": pd.Timestamp("2025-10-01 00:00:00"),
            "end": pd.Timestamp("2026-01-01 00:15:00"),
        },
    }

    selected_period = split_periods[dataset_split]

    processing_start = selected_period["start"]
    processing_end = selected_period["end"]

    target_start = processing_start
    target_end = processing_end + time_step

    # Process all 11 households sequentially.
    for household_id in range(1, 12):
        dataset_name = f"df_{household_id}"

        input_path = (
            f"data/processed/"
            f"{dataset_name}_{dataset_split}.csv"
        )

        roll_sizes_text = "_".join(
            str(roll_size)
            for roll_size in roll_sizes
        )

        output_path = (
            f"data/features/"
            f"{dataset_name}_{dataset_split}_features_"
            f"{roll_sizes_text}_target_t96.pkl"
        )

        print(f"\n=== Processing {input_path} ===")

        feature_engineer = FeatureEngineer(
            data_file=input_path,
            target_column="Energija A+",
            time_column="datetime",
            time_format="%Y-%m-%d %H:%M:%S",
            start=str(processing_start),
            end=str(processing_end),
            freq=frequency,
            roll_sizes=roll_sizes,
            roll_feat_params=roll_feat_params,
            horizon=horizon,
            series_id=dataset_name,
            target_start=str(target_start),
            target_end=str(target_end),
            split_name=dataset_split,
        )

        selected_dataset = feature_engineer.create_dataset(
            save_path=output_path,
            n_jobs=1,
        )

        print(f"Saved: {output_path}")
        print(f"Shape: {selected_dataset.shape}")
