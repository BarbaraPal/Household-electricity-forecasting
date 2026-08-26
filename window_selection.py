from __future__ import annotations

import gc
import itertools
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
from sklearn.metrics import mean_squared_error

from config import (
    HORIZON,
    HOUSEHOLD_IDS,
    WINDOWS,
    get_feature_file,
    get_target_name,
    get_window_ranking_directory,
    get_window_selection_directory,
)


NON_FEATURE_COLUMNS = {
    "unique_id",
    "ds",
    "target_time",
    "y",
    "index",
    "level_0",
}

RF_PARAMS = {
    "n_estimators": 50,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": 2,
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


def load_candidate_windows(
    selected_windows_file: Path,
    allowed_windows: Sequence[int],
) -> list[int]:
    """Load the highest-ranked candidate windows."""
    if not selected_windows_file.exists():
        raise FileNotFoundError(
            f"File not found: {selected_windows_file}"
        )

    with selected_windows_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # The current window-ranking script stores the candidates under
    # ``selected_windows``. ``candidate_windows`` is also accepted so that the
    # name can be made more explicit later without breaking this script.
    raw_windows = data.get(
        "candidate_windows",
        data.get("selected_windows"),
    )

    if raw_windows is None:
        raise ValueError(
            "Neither 'candidate_windows' nor 'selected_windows' is present "
            f"in {selected_windows_file}."
        )

    candidate_windows = [int(window) for window in raw_windows]

    if not candidate_windows:
        raise ValueError("At least one candidate window is required.")

    if len(candidate_windows) != len(set(candidate_windows)):
        raise ValueError(
            f"Candidate windows are not unique: {candidate_windows}"
        )

    unknown_windows = sorted(
        set(candidate_windows).difference(allowed_windows)
    )

    if unknown_windows:
        raise ValueError(
            "Candidate windows are not present in the configured WINDOWS: "
            f"{unknown_windows}."
        )

    return candidate_windows


def load_split(
    path: Path,
    target: str,
    candidate_windows: Sequence[int],
    split_name: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    dict[int, list[str]],
]:
    """Load one split and retain candidate-window and manual features."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    dataframe = pd.read_pickle(path)

    if not dataframe.columns.is_unique:
        duplicates = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate columns in {split_name}: {duplicates}"
        )

    if target not in dataframe.columns:
        raise ValueError(f"{target} is missing from {split_name}.")

    manual_features = find_manual_features(dataframe)

    if not manual_features:
        raise ValueError(f"No manual features in {split_name}.")

    features_by_window: dict[int, list[str]] = {}

    for window in candidate_windows:
        prefix = f"r{window}_"
        window_features = [
            column
            for column in dataframe.columns
            if str(column).startswith(prefix)
        ]

        if not window_features:
            raise ValueError(
                f"No columns with prefix '{prefix}' in {split_name}."
            )

        features_by_window[window] = window_features

    tsfresh_features = [
        feature
        for window in candidate_windows
        for feature in features_by_window[window]
    ]

    feature_columns = [
        *manual_features,
        *tsfresh_features,
    ]

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            f"Feature names are not unique in {split_name}."
        )

    X = dataframe.loc[:, feature_columns].copy()
    y = pd.to_numeric(
        dataframe[target],
        errors="raise",
    ).astype(np.float32)

    del dataframe
    gc.collect()

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        ).astype(np.float32)

    invalid_columns = [
        column
        for column in X.columns
        if not np.isfinite(X[column].to_numpy(copy=False)).all()
    ]

    if invalid_columns:
        raise ValueError(
            f"{split_name} contains NaN or infinite values in columns: "
            f"{invalid_columns[:20]}"
        )

    if not np.isfinite(y.to_numpy(copy=False)).all():
        raise ValueError(
            f"{split_name} target contains NaN or infinite values."
        )

    print(f"{split_name} rows: {len(X):,}")
    print(
        f"{split_name} manual features: {len(manual_features):,}"
    )
    print(
        f"{split_name} candidate TSFresh features: "
        f"{len(tsfresh_features):,}"
    )

    return X, y, manual_features, features_by_window


def validate_compatibility(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    train_manual_features: list[str],
    validation_manual_features: list[str],
    train_features_by_window: dict[int, list[str]],
    validation_features_by_window: dict[int, list[str]],
) -> None:
    """Verify that training and validation feature schemas are identical."""
    if train_manual_features != validation_manual_features:
        raise ValueError(
            "Training and validation manual features are not identical."
        )

    if train_features_by_window != validation_features_by_window:
        raise ValueError(
            "Training and validation TSFresh features are not identical."
        )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns are not identical."
        )


def create_combinations(
    candidate_windows: Sequence[int],
) -> list[tuple[int, ...]]:
    """Create every non-empty combination of candidate windows."""
    combinations = [
        combination
        for size in range(1, len(candidate_windows) + 1)
        for combination in itertools.combinations(candidate_windows, size)
    ]

    expected_count = 2 ** len(candidate_windows) - 1

    if len(combinations) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} combinations, but created "
            f"{len(combinations)}."
        )

    return combinations


def evaluate_combinations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    manual_features: list[str],
    features_by_window: dict[int, list[str]],
    combinations: list[tuple[int, ...]],
) -> pd.DataFrame:
    """Train on the training split and evaluate every combination."""
    results = []

    for number, combination in enumerate(combinations, start=1):
        tsfresh_features = [
            feature
            for window in combination
            for feature in features_by_window[window]
        ]

        model_features = [
            *manual_features,
            *tsfresh_features,
        ]

        X_train_subset = X_train.loc[:, model_features]
        X_validation_subset = X_validation.loc[:, model_features]

        model = RandomForestRegressor(**RF_PARAMS)

        fit_start = time.perf_counter()
        model.fit(X_train_subset, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predictions = model.predict(X_validation_subset)

        rmse = float(
            np.sqrt(mean_squared_error(y_validation, predictions))
        )

        results.append(
            {
                "windows": list(combination),
                "n_windows": len(combination),
                "n_manual_features": len(manual_features),
                "n_tsfresh_features": len(tsfresh_features),
                "n_features": len(model_features),
                "validation_rmse": rmse,
                "fit_seconds": fit_seconds,
            }
        )

        print(
            f"[{number:02d}/{len(combinations)}] "
            f"windows={list(combination)}, "
            f"features={len(model_features):,}, "
            f"RMSE={rmse:.6f}, "
            f"fit={timedelta(seconds=int(fit_seconds))}"
        )

        del X_train_subset
        del X_validation_subset
        del model
        del predictions
        gc.collect()

    results_dataframe = (
        pd.DataFrame(results)
        .sort_values(
            ["validation_rmse", "n_features"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    results_dataframe.insert(
        0,
        "rank",
        np.arange(1, len(results_dataframe) + 1),
    )

    return results_dataframe


def save_results(
    output_directory: Path,
    train_file: Path,
    validation_file: Path,
    selected_windows_file: Path,
    target: str,
    results: pd.DataFrame,
    candidate_windows: list[int],
    total_seconds: float,
    train_rows: int,
    validation_rows: int,
) -> None:
    """Save validation results and the best window combination."""
    output_directory.mkdir(parents=True, exist_ok=True)

    results.to_pickle(
        output_directory / "window_combination_results.pkl"
    )

    csv_results = results.copy()
    csv_results["windows"] = csv_results["windows"].apply(
        lambda windows: "_".join(str(window) for window in windows)
    )
    csv_results.to_csv(
        output_directory / "window_combination_results.csv",
        index=False,
    )

    best = results.iloc[0]
    best_windows = [int(window) for window in best["windows"]]

    best_result = {
        "best_windows": best_windows,
        "validation_rmse": float(best["validation_rmse"]),
        "n_features": int(best["n_features"]),
        "n_manual_features": int(best["n_manual_features"]),
        "n_tsfresh_features": int(best["n_tsfresh_features"]),
    }

    with (output_directory / "best_window_combination.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_result, file, indent=2)

    run_info = {
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "selected_windows_file": str(selected_windows_file),
        "target": target,
        "candidate_windows": candidate_windows,
        "n_combinations": len(results),
        "rf_params": RF_PARAMS,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "best_result": best_result,
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
    """Select the best window combination for one household."""
    total_start = time.perf_counter()
    target = get_target_name(HORIZON)

    train_file = get_feature_file(
        household_id=household_id,
        dataset_split="train",
        windows=WINDOWS,
        horizon=HORIZON,
    )
    validation_file = get_feature_file(
        household_id=household_id,
        dataset_split="validation",
        windows=WINDOWS,
        horizon=HORIZON,
    )
    selected_windows_file = (
        get_window_ranking_directory(household_id)
        / "selected_windows.json"
    )
    output_directory = get_window_selection_directory(household_id)

    print()
    print(
        f"=== Selecting windows for household {household_id} ==="
    )
    print(f"Training file: {train_file}")
    print(f"Validation file: {validation_file}")

    candidate_windows = load_candidate_windows(
        selected_windows_file=selected_windows_file,
        allowed_windows=WINDOWS,
    )

    print(f"Candidate windows: {candidate_windows}")
    print()

    (
        X_train,
        y_train,
        train_manual_features,
        train_features_by_window,
    ) = load_split(
        path=train_file,
        target=target,
        candidate_windows=candidate_windows,
        split_name="Training",
    )

    print()

    (
        X_validation,
        y_validation,
        validation_manual_features,
        validation_features_by_window,
    ) = load_split(
        path=validation_file,
        target=target,
        candidate_windows=candidate_windows,
        split_name="Validation",
    )

    validate_compatibility(
        X_train=X_train,
        X_validation=X_validation,
        train_manual_features=train_manual_features,
        validation_manual_features=validation_manual_features,
        train_features_by_window=train_features_by_window,
        validation_features_by_window=validation_features_by_window,
    )

    combinations = create_combinations(candidate_windows)

    print()
    print(
        f"Evaluating {len(combinations)} window combinations:"
    )

    results = evaluate_combinations(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        manual_features=train_manual_features,
        features_by_window=train_features_by_window,
        combinations=combinations,
    )

    total_seconds = time.perf_counter() - total_start

    save_results(
        output_directory=output_directory,
        train_file=train_file,
        validation_file=validation_file,
        selected_windows_file=selected_windows_file,
        target=target,
        results=results,
        candidate_windows=candidate_windows,
        total_seconds=total_seconds,
        train_rows=len(X_train),
        validation_rows=len(X_validation),
    )

    print()
    print("Validation ranking:")
    print(
        results[
            [
                "rank",
                "windows",
                "n_features",
                "validation_rmse",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )

    best = results.iloc[0]

    print()
    print(f"Best window combination: {best['windows']}")
    print(
        f"Best validation RMSE: {best['validation_rmse']:.6f}"
    )
    print(
        "Total duration: "
        f"{timedelta(seconds=int(total_seconds))}"
    )
    print(f"Results saved to: {output_directory}")

    del X_train
    del y_train
    del X_validation
    del y_validation
    del train_manual_features
    del validation_manual_features
    del train_features_by_window
    del validation_features_by_window
    del combinations
    del results
    gc.collect()


def main() -> None:
    """Select the best window combination for all households."""
    for household_id in HOUSEHOLD_IDS:
        process_household(household_id)


if __name__ == "__main__":
    main()
