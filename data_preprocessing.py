from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd


class ElectricityDataPreprocessor:
    """
    Preprocess household electricity consumption data.

    The class:
    1. loads CSV files,
    2. converts timestamps to fixed Central European Time (UTC+1),
    3. restricts the data to the selected time period,
    4. checks the target column for missing values,
    5. splits the data chronologically into training, validation,
       and test sets,
    6. saves the processed datasets.
    """

    def __init__(
        self,
        input_files: dict[str, str],
        output_directory: str = "data/processed",
        time_column: str = "Časovna značka",
        target_column: str = "Energija A+",
        start: str = "2024-01-01 00:15:00",
        end: str = "2026-01-01 00:15:00",
    ):
        """
        Initialize the data preprocessor.

        Parameters
        ----------
        input_files : dict[str, str]
            Mapping between dataset names and CSV file paths.

        output_directory : str
            Directory in which the processed datasets will be saved.

        time_column : str
            Name of the column containing timestamps.

        target_column : str
            Name of the target column.

        start : str
            First timestamp included in the processed dataset.

        end : str
            Last timestamp included in the processed dataset.
        """
        self.input_files = input_files
        self.output_directory = Path(output_directory)
        self.time_column = time_column
        self.target_column = target_column

        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)

        if self.start > self.end:
            raise ValueError(
                "The start timestamp must not be later "
                "than the end timestamp."
            )

        # Chronological split boundaries.
        self.training_end = pd.Timestamp(
            "2025-06-30 00:00:00"
        )
        self.validation_start = pd.Timestamp(
            "2025-07-01 00:00:00"
        )
        self.validation_end = pd.Timestamp(
            "2025-09-30 00:00:00"
        )
        self.test_start = pd.Timestamp(
            "2025-10-01 00:00:00"
        )

        self.dataframes: dict[str, pd.DataFrame] = {}

    def prepare_time_index(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert timestamps to fixed Central European Time (UTC+1),
        restrict the data to the selected period, and use the
        timestamps as the dataframe index.
        """
        dataframe = dataframe.copy()

        if self.time_column not in dataframe.columns:
            raise ValueError(
                f"Column '{self.time_column}' is missing "
                "from the dataset."
            )

        # Convert timestamp values to datetime objects.
        dataframe["datetime"] = pd.to_datetime(
            dataframe[self.time_column],
            errors="raise",
        )

        # Preserve the original row order when resolving the repeated
        # hour during the autumn daylight-saving-time transition.
        local_time = pd.DatetimeIndex(
            dataframe["datetime"]
        )

        # Interpret timestamps as local Slovenian time.
        local_time = local_time.tz_localize(
            "Europe/Ljubljana",
            ambiguous="infer",
            nonexistent="raise",
        )

        # Convert timestamps to fixed Central European Time (UTC+1).
        fixed_cet = timezone(
            timedelta(hours=1)
        )

        winter_time = (
            local_time
            .tz_convert(fixed_cet)
            .tz_localize(None)
        )

        dataframe["datetime"] = winter_time

        dataframe = (
            dataframe
            .set_index("datetime")
            .sort_index()
        )

        # Keep the selected period, including both boundaries.
        dataframe = dataframe.loc[
            self.start:self.end
        ].copy()

        if dataframe.empty:
            raise ValueError(
                "No observations were found in the selected "
                "time period."
            )

        return dataframe

    def load_data(self) -> None:
        """
        Load all CSV files and prepare their time indices.
        """
        for name, file_path in self.input_files.items():
            dataframe = pd.read_csv(file_path)

            if self.target_column not in dataframe.columns:
                raise ValueError(
                    f"Column '{self.target_column}' is missing "
                    f"from dataset '{name}'."
                )

            self.dataframes[name] = (
                self.prepare_time_index(dataframe)
            )

            print(
                f"Loaded dataset: {name} "
                f"({len(self.dataframes[name])} rows)"
            )

    def check_missing_values(self) -> None:
        """
        Check the target column for missing values.
        """
        for name, dataframe in self.dataframes.items():
            missing_count = (
                dataframe[self.target_column]
                .isna()
                .sum()
            )

            if missing_count > 0:
                print(
                    f"{name}: {missing_count} missing values "
                    f"found in column '{self.target_column}'."
                )
            else:
                print(
                    f"{name}: no missing values found "
                    f"in column '{self.target_column}'."
                )

    def split_data(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Split a dataset chronologically into training,
        validation, and test sets, leaving a one-day gap
        between consecutive subsets.
        """
        target_data = dataframe[
            [self.target_column]
        ].copy()

        training_data = target_data.loc[
            target_data.index < self.training_end
        ].copy()

        validation_data = target_data.loc[
            (
                target_data.index
                >= self.validation_start
            )
            & (
                target_data.index
                < self.validation_end
            )
        ].copy()

        test_data = target_data.loc[
            target_data.index >= self.test_start
        ].copy()

        return (
            training_data,
            validation_data,
            test_data,
        )

    def save_data(self) -> None:
        """
        Save the complete target series and the training,
        validation, and test sets.
        """
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, dataframe in self.dataframes.items():
            (
                training_data,
                validation_data,
                test_data,
            ) = self.split_data(dataframe)

            # Save the complete processed target series.
            dataframe[[self.target_column]].to_csv(
                self.output_directory / f"{name}.csv"
            )

            training_data.to_csv(
                self.output_directory
                / f"{name}_train.csv"
            )

            validation_data.to_csv(
                self.output_directory
                / f"{name}_validation.csv"
            )

            test_data.to_csv(
                self.output_directory
                / f"{name}_test.csv"
            )

            print(
                f"Saved processed datasets for: {name}"
            )

    def run(self) -> None:
        """
        Run the complete preprocessing pipeline.
        """
        self.load_data()
        self.check_missing_values()
        self.save_data()


if __name__ == "__main__":
    input_files = {
        "df_1": "data/dfs/df_1.csv"
    }

    preprocessor = ElectricityDataPreprocessor(
        input_files=input_files,
        output_directory="data/processed",
        start="2024-01-01 00:15:00",
        end="2026-01-01 00:15:00",
    )

    preprocessor.run()