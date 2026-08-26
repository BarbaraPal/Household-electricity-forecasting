from __future__ import annotations

import gc
import json
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor

from config import (
    FINAL_TRAINING_SPLITS,
    HORIZON,
    HOUSEHOLD_IDS,
    TOP_FEATURE_COUNTS,
    WINDOWS,
    DatasetSplit,
    get_best_window_combination_file,
    get_feature_file,
    get_feature_ranking_directory,
    get_target_name,
)


RF_PARAMS = {
    "n_estimators": 50,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": 2,
}

KEY_COLUMNS = ["unique_id", "ds"]

NON_FEATURE_COLUMNS = {
    "unique_id",
    "ds",
    "target_time",
    "y",
    "index",
    "level_0",
}


def is_tsfresh_feature(column: object) -> bool:
    """Return whether a column is a prefixed TSFresh feature."""
    return re.match(r"^r\d+_", str(column)) is not None


def find_manual_features(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric non-target columns that are not TSFresh features."""
    target_columns = {
        column
        for column in dataframe.columns
        if str(column).startswith("target_t")
    }

    return [
        column
        for column in dataframe.columns
        if column not in NON_FEATURE_COLUMNS
        and column not in target_columns
        and not is_tsfresh_feature(column)
        and (
            is_numeric_dtype(dataframe[column])
            or is_bool_dtype(dataframe[column])
        )
    ]


def load_best_windows(
    path: Path,
    allowed_windows: Sequence[int],
) -> list[int]:
    """Load the validation-selected window combination."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "best_windows" not in data:
        raise ValueError(f"'best_windows' is missing from {path}.")

    selected_windows = [
        int(window) for window in data["best_windows"]
    ]

    if not selected_windows:
        raise ValueError(f"No selected windows in {path}.")

    if len(selected_windows) != len(set(selected_windows)):
        raise ValueError(
            f"Selected windows are not unique: {selected_windows}"
        )

    unknown_windows = sorted(
        set(selected_windows).difference(allowed_windows)
    )

    if unknown_windows:
        raise ValueError(
            "Selected windows are not present in the configured WINDOWS: "
            f"{unknown_windows}."
        )

    return selected_windows


def validate_source_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
    target: str,
) -> None:
    """Validate keys, target, and column uniqueness."""
    if not dataframe.columns.is_unique:
        duplicates = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
        raise ValueError(f"Duplicate columns in {path}: {duplicates}")

    missing_columns = [
        column
        for column in [*KEY_COLUMNS, target]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {missing_columns}"
        )

    if dataframe.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            f"Duplicate combinations of {KEY_COLUMNS} in {path}."
        )


def to_float32(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert every dataframe column to float32."""
    converted = dataframe.apply(pd.to_numeric, errors="raise")
    return converted.astype(np.float32)


def find_invalid_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns containing NaN or infinite values."""
    return [
        column
        for column in dataframe.columns
        if not np.isfinite(
            dataframe[column].to_numpy(copy=False)
        ).all()
    ]


def load_selected_split(
    path: Path,
    target: str,
    selected_windows: Sequence[int],
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load features from the validation-selected windows for one split."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    dataframe = pd.read_pickle(path)
    validate_source_dataframe(dataframe, path, target)

    dataframe["ds"] = pd.to_datetime(dataframe["ds"])
    dataframe = dataframe.set_index(KEY_COLUMNS, verify_integrity=True)

    manual_features = find_manual_features(dataframe)

    if not manual_features:
        raise ValueError(f"No manual features found in {path}.")

    selected_tsfresh_features: list[str] = []

    for window in selected_windows:
        prefix = f"r{window}_"
        window_features = [
            column
            for column in dataframe.columns
            if str(column).startswith(prefix)
        ]

        if not window_features:
            raise ValueError(
                f"No columns with prefix '{prefix}' in {path}."
            )

        selected_tsfresh_features.extend(window_features)

    feature_columns = [
        *manual_features,
        *selected_tsfresh_features,
    ]

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            f"Selected feature names are not unique in {path}."
        )

    X = to_float32(dataframe.loc[:, feature_columns])
    y = pd.to_numeric(
        dataframe[target],
        errors="raise",
    ).astype(np.float32)

    del dataframe
    gc.collect()

    invalid_columns = find_invalid_columns(X)

    if invalid_columns:
        raise ValueError(
            f"{split_name} contains NaN or infinite values in columns: "
            f"{invalid_columns[:20]}"
        )

    if not np.isfinite(y.to_numpy(copy=False)).all():
        raise ValueError(
            f"{split_name} target contains NaN or infinite values."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            f"Feature and target indices differ in {split_name}."
        )

    print(
        f"  {split_name}: {len(X):,} rows, "
        f"{len(X.columns):,} features"
    )

    return X, y, manual_features


def validate_training_splits(
    split_data: dict[
        DatasetSplit,
        tuple[pd.DataFrame, pd.Series, list[str]],
    ],
) -> None:
    """Verify equal schemas and non-overlapping rows across splits."""
    split_names = list(split_data)
    reference_split = split_names[0]
    reference_X, _, reference_manual = split_data[reference_split]

    for split_name in split_names[1:]:
        X, _, manual_features = split_data[split_name]

        if manual_features != reference_manual:
            raise ValueError(
                f"Manual features in {reference_split} and {split_name} "
                "are not identical."
            )

        if list(X.columns) != list(reference_X.columns):
            raise ValueError(
                f"Feature columns in {reference_split} and {split_name} "
                "are not identical."
            )

    for first_index, first_split in enumerate(split_names):
        first_X, _, _ = split_data[first_split]

        for second_split in split_names[first_index + 1:]:
            second_X, _, _ = split_data[second_split]
            overlap = first_X.index.intersection(second_X.index)

            if len(overlap) > 0:
                raise ValueError(
                    f"{first_split} and {second_split} contain "
                    "overlapping rows."
                )


def create_feature_ranking(
    model: RandomForestRegressor,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create a descending ranking from fitted-model importances."""
    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    ranking["feature_type"] = np.where(
        ranking["feature"].map(is_tsfresh_feature),
        "tsfresh",
        "manual",
    )

    ranking["window"] = (
        ranking["feature"]
        .astype(str)
        .str.extract(r"^r(\d+)_", expand=False)
        .astype("Int64")
    )

    ranking = (
        ranking
        .sort_values(
            ["importance", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    return ranking


def create_feature_sets(
    ranking: pd.DataFrame,
    top_feature_counts: Sequence[int],
) -> dict[str, list[str]]:
    """Create the feature sets defined in the methodology."""
    ranked_features = ranking["feature"].tolist()

    invalid_counts = [
        count
        for count in top_feature_counts
        if count <= 0 or count > len(ranked_features)
    ]

    if invalid_counts:
        raise ValueError(
            "Invalid TOP_FEATURE_COUNTS for the available ranking: "
            f"{invalid_counts}."
        )

    feature_sets = {
        f"top_{count}": ranked_features[:count]
        for count in top_feature_counts
    }
    feature_sets["all"] = ranked_features

    return feature_sets


def save_results(
    output_directory: Path,
    household_id: int,
    feature_files: dict[DatasetSplit, Path],
    best_windows_file: Path,
    target: str,
    selected_windows: list[int],
    ranking: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    manual_features: list[str],
    rows_by_split: dict[DatasetSplit, int],
    fit_seconds: float,
    total_seconds: float,
) -> None:
    """Save ranking, feature sets, and run metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)

    ranking.to_pickle(output_directory / "feature_ranking.pkl")
    ranking.to_csv(
        output_directory / "feature_ranking.csv",
        index=False,
    )

    feature_sets_data = {
        "household": household_id,
        "selected_windows": selected_windows,
        "feature_sets": feature_sets,
    }

    with (output_directory / "feature_sets.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feature_sets_data, file, indent=2)

    run_info = {
        "household": household_id,
        "feature_files": {
            split_name: str(path)
            for split_name, path in feature_files.items()
        },
        "best_windows_file": str(best_windows_file),
        "target": target,
        "training_splits": list(FINAL_TRAINING_SPLITS),
        "selected_windows": selected_windows,
        "rf_params": RF_PARAMS,
        "rows_by_split": rows_by_split,
        "combined_rows": sum(rows_by_split.values()),
        "n_manual_features": len(manual_features),
        "n_tsfresh_features": int(
            (ranking["feature_type"] == "tsfresh").sum()
        ),
        "n_features": len(ranking),
        "feature_set_sizes": {
            name: len(features)
            for name, features in feature_sets.items()
        },
        "fit_seconds": fit_seconds,
        "fit_duration": str(
            timedelta(seconds=int(fit_seconds))
        ),
        "total_seconds": total_seconds,
        "total_duration": str(
            timedelta(seconds=int(total_seconds))
        ),
        "timestamp": pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with (output_directory / "run_info.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_info, file, indent=2)


def process_household(household_id: int) -> None:
    """Create the final feature ranking for one household."""
    total_start = time.perf_counter()
    target = get_target_name(HORIZON)
    best_windows_file = get_best_window_combination_file(household_id)
    output_directory = get_feature_ranking_directory(household_id)

    selected_windows = load_best_windows(
        path=best_windows_file,
        allowed_windows=WINDOWS,
    )

    print()
    print(f"=== Ranking features for household {household_id} ===")
    print(f"Selected windows: {selected_windows}")

    feature_files: dict[DatasetSplit, Path] = {}
    split_data: dict[
        DatasetSplit,
        tuple[pd.DataFrame, pd.Series, list[str]],
    ] = {}

    for split_name in FINAL_TRAINING_SPLITS:
        feature_file = get_feature_file(
            household_id=household_id,
            dataset_split=split_name,
            windows=WINDOWS,
            horizon=HORIZON,
        )
        feature_files[split_name] = feature_file
        split_data[split_name] = load_selected_split(
            path=feature_file,
            target=target,
            selected_windows=selected_windows,
            split_name=split_name.capitalize(),
        )

    validate_training_splits(split_data)

    rows_by_split = {
        split_name: len(split_data[split_name][0])
        for split_name in FINAL_TRAINING_SPLITS
    }
    manual_features = list(split_data[FINAL_TRAINING_SPLITS[0]][2])

    X = pd.concat(
        [
            split_data[split_name][0]
            for split_name in FINAL_TRAINING_SPLITS
        ],
        axis=0,
        ignore_index=True,
        copy=False,
    )
    y = pd.concat(
        [
            split_data[split_name][1]
            for split_name in FINAL_TRAINING_SPLITS
        ],
        axis=0,
        ignore_index=True,
        copy=False,
    )

    del split_data
    gc.collect()

    print(
        f"Combined: {len(X):,} rows, {len(X.columns):,} features"
    )

    model = RandomForestRegressor(**RF_PARAMS)

    fit_start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - fit_start

    ranking = create_feature_ranking(
        model=model,
        feature_columns=list(X.columns),
    )
    feature_sets = create_feature_sets(
        ranking=ranking,
        top_feature_counts=TOP_FEATURE_COUNTS,
    )

    total_seconds = time.perf_counter() - total_start

    save_results(
        output_directory=output_directory,
        household_id=household_id,
        feature_files=feature_files,
        best_windows_file=best_windows_file,
        target=target,
        selected_windows=selected_windows,
        ranking=ranking,
        feature_sets=feature_sets,
        manual_features=manual_features,
        rows_by_split=rows_by_split,
        fit_seconds=fit_seconds,
        total_seconds=total_seconds,
    )

    print("Top 10 features:")
    print(
        ranking.loc[
            :9,
            ["rank", "feature", "feature_type", "window", "importance"],
        ].to_string(index=False)
    )
    print(
        f"Fit duration: {timedelta(seconds=int(fit_seconds))}"
    )
    print(f"Results saved to: {output_directory}")

    del model
    del X
    del y
    del ranking
    del feature_sets
    gc.collect()


def main() -> None:
    """Rank selected features for every configured household."""
    if "test" in FINAL_TRAINING_SPLITS:
        raise ValueError(
            "FINAL_TRAINING_SPLITS must not contain the test split."
        )

    if set(FINAL_TRAINING_SPLITS) != {"train", "validation"}:
        raise ValueError(
            "FINAL_TRAINING_SPLITS must contain train and validation."
        )

    print("The test split is not used in this script.")

    for household_id in HOUSEHOLD_IDS:
        process_household(household_id)

    print()
    print("Feature ranking completed for all households.")


if __name__ == "__main__":
    main()
