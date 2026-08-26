from __future__ import annotations

import gc
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    FEATURE_SET_ORDER,
    FINAL_EVALUATION_DIRECTORY,
    FINAL_TRAINING_SPLITS,
    HORIZON,
    HOUSEHOLD_IDS,
    WINDOWS,
    DatasetSplit,
    get_feature_file,
    get_feature_sets_file,
    get_final_evaluation_directory,
    get_target_name,
)


TARGET = get_target_name(HORIZON)

# For H = 96, lag_96 at the target time is the last value available at the
# forecast origin. It is therefore used as the 24-hour naive reference forecast.
REFERENCE_FEATURE = "lag_96"

RF_PARAMS = {
    "n_estimators": 200,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": 2,
}

KEY_COLUMNS = ["unique_id", "ds"]


def load_feature_sets(
    household_id: int,
) -> tuple[list[int], dict[str, list[str]]]:
    """Load and validate the feature sets produced by feature_rank.py."""
    path = get_feature_sets_file(household_id)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "selected_windows" not in data:
        raise ValueError(f"'selected_windows' is missing from {path}")

    if "feature_sets" not in data:
        raise ValueError(f"'feature_sets' is missing from {path}")

    selected_windows = [
        int(window) for window in data["selected_windows"]
    ]

    if not selected_windows:
        raise ValueError(f"'selected_windows' is empty in {path}")

    if len(selected_windows) != len(set(selected_windows)):
        raise ValueError(
            f"'selected_windows' contains duplicates in {path}"
        )

    invalid_windows = sorted(set(selected_windows).difference(WINDOWS))

    if invalid_windows:
        raise ValueError(
            f"Unknown selected windows in {path}: {invalid_windows}"
        )

    raw_feature_sets = data["feature_sets"]

    if not isinstance(raw_feature_sets, dict):
        raise ValueError(f"'feature_sets' must be a dictionary in {path}")

    expected_names = list(FEATURE_SET_ORDER)
    missing_names = [
        name for name in expected_names if name not in raw_feature_sets
    ]
    unexpected_names = sorted(
        set(raw_feature_sets).difference(expected_names)
    )

    if missing_names:
        raise ValueError(
            f"Missing feature sets in {path}: {missing_names}"
        )

    if unexpected_names:
        raise ValueError(
            f"Unexpected feature sets in {path}: {unexpected_names}"
        )

    feature_sets: dict[str, list[str]] = {}

    for name in expected_names:
        features = raw_feature_sets[name]

        if not isinstance(features, list) or not features:
            raise ValueError(
                f"Feature set '{name}' is empty or invalid in {path}"
            )

        feature_names = [str(feature) for feature in features]

        if len(feature_names) != len(set(feature_names)):
            raise ValueError(
                f"Feature set '{name}' contains duplicate features"
            )

        feature_sets[name] = feature_names

    all_features = set(feature_sets["all"])

    for name, features in feature_sets.items():
        unknown_features = set(features).difference(all_features)

        if unknown_features:
            raise ValueError(
                f"Feature set '{name}' contains features that are not "
                f"listed in 'all': {sorted(unknown_features)[:20]}"
            )

    return selected_windows, feature_sets


def validate_source_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Validate identifiers and the target in a combined feature file."""
    if not df.columns.is_unique:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns in {path}: {duplicates}")

    required_columns = [*KEY_COLUMNS, TARGET]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path}: {missing_columns}"
        )

    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            f"Duplicate combinations of {KEY_COLUMNS} in {path}"
        )


def find_invalid_columns(df: pd.DataFrame) -> list[str]:
    """Return columns containing NaN or infinite values."""
    return [
        column
        for column in df.columns
        if not np.isfinite(df[column].to_numpy(copy=False)).all()
    ]


def load_split(
    household_id: int,
    dataset_split: DatasetSplit,
    required_features: Sequence[str],
    include_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """Load one configured split and retain only required model columns."""
    path = get_feature_file(
        household_id=household_id,
        dataset_split=dataset_split,
        windows=WINDOWS,
        horizon=HORIZON,
    )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_pickle(path)
    validate_source_dataframe(df, path)
    df["ds"] = pd.to_datetime(df["ds"])

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features in {path}: {missing_features[:20]}"
        )

    metadata: pd.DataFrame | None = None

    if include_metadata:
        metadata_columns = [REFERENCE_FEATURE, "target_time"]
        missing_metadata = [
            column
            for column in metadata_columns
            if column not in df.columns
        ]

        if missing_metadata:
            raise ValueError(
                f"Missing test metadata in {path}: {missing_metadata}"
            )

    df = df.set_index(KEY_COLUMNS, verify_integrity=True)

    X = (
        df.loc[:, list(required_features)]
        .apply(pd.to_numeric, errors="raise")
        .astype(np.float32)
    )
    y = pd.to_numeric(df[TARGET], errors="raise").astype(np.float32)

    if include_metadata:
        metadata = pd.DataFrame(index=df.index)
        metadata["reference_prediction"] = pd.to_numeric(
            df[REFERENCE_FEATURE], errors="raise"
        ).astype(np.float32)
        metadata["target_time"] = pd.to_datetime(df["target_time"])

    del df
    gc.collect()

    invalid_columns = find_invalid_columns(X)

    if invalid_columns:
        raise ValueError(
            f"{dataset_split} contains NaN or infinite values in: "
            f"{invalid_columns[:20]}"
        )

    if not np.isfinite(y.to_numpy(copy=False)).all():
        raise ValueError(
            f"{dataset_split} target contains NaN or infinite values"
        )

    if metadata is not None:
        reference_values = metadata[
            "reference_prediction"
        ].to_numpy(copy=False)

        if not np.isfinite(reference_values).all():
            raise ValueError(
                f"{dataset_split} reference predictions contain NaN or "
                "infinite values"
            )

    print(
        f"  {dataset_split}: {len(X):,} rows, "
        f"{len(X.columns):,} available features"
    )

    return X, y, metadata


def validate_splits(
    split_features: dict[DatasetSplit, pd.DataFrame],
    X_test: pd.DataFrame,
) -> None:
    """Check column identity, row separation, and chronological order."""
    ordered_splits = list(FINAL_TRAINING_SPLITS)
    reference_columns = list(split_features[ordered_splits[0]].columns)

    for split_name in ordered_splits[1:]:
        if list(split_features[split_name].columns) != reference_columns:
            raise ValueError(
                "Final-training feature columns are not identical across "
                "splits"
            )

    if list(X_test.columns) != reference_columns:
        raise ValueError(
            "Final-training and test feature columns are not identical"
        )

    all_splits: list[tuple[str, pd.DataFrame]] = [
        (split_name, split_features[split_name])
        for split_name in ordered_splits
    ]
    all_splits.append(("test", X_test))

    for first_index, (first_name, first_df) in enumerate(all_splits):
        for second_name, second_df in all_splits[first_index + 1 :]:
            overlap = first_df.index.intersection(second_df.index)

            if len(overlap) > 0:
                raise ValueError(
                    f"{first_name} and {second_name} rows overlap"
                )

    for (first_name, first_df), (second_name, second_df) in zip(
        all_splits,
        all_splits[1:],
    ):
        first_times = first_df.index.get_level_values("ds")
        second_times = second_df.index.get_level_values("ds")

        if first_times.max() >= second_times.min():
            raise ValueError(
                f"{first_name} rows are not strictly before "
                f"{second_name} rows"
            )


def evaluate_feature_sets(
    X_train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metadata: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every configured feature set and evaluate it on the test set."""
    results: list[dict[str, object]] = []
    prediction_results: list[pd.DataFrame] = []

    reference_predictions = test_metadata[
        "reference_prediction"
    ].to_numpy(copy=False)
    reference_rmse = float(
        np.sqrt(mean_squared_error(y_test, reference_predictions))
    )

    if not np.isfinite(reference_rmse) or reference_rmse <= 0:
        raise ValueError(
            "The reference RMSE must be a positive finite value"
        )

    actual_values = y_test.to_numpy(copy=False)

    for number, (name, features) in enumerate(
        feature_sets.items(),
        start=1,
    ):
        print(
            f"  [{number:02d}/{len(feature_sets)}] "
            f"{name}: {len(features):,} features"
        )

        model = RandomForestRegressor(**RF_PARAMS)
        X_fit = X_train_validation.loc[:, features]
        X_evaluation = X_test.loc[:, features]

        fit_start = time.perf_counter()
        model.fit(X_fit, y_train_validation)
        fit_seconds = time.perf_counter() - fit_start

        prediction_start = time.perf_counter()
        predictions = model.predict(X_evaluation)
        prediction_seconds = time.perf_counter() - prediction_start

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        bias = float(np.mean(actual_values - predictions))
        relative_rmse = float(rmse / reference_rmse)

        results.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "reference_rmse": reference_rmse,
                "rrmse": relative_rmse,
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
            }
        )

        current_predictions = test_metadata.reset_index().copy()
        current_predictions["feature_set"] = name
        current_predictions["actual"] = actual_values
        current_predictions["prediction"] = predictions
        current_predictions["error"] = (
            current_predictions["actual"]
            - current_predictions["prediction"]
        )
        current_predictions["absolute_error"] = (
            current_predictions["error"].abs()
        )
        current_predictions["squared_error"] = (
            current_predictions["error"] ** 2
        )
        current_predictions["reference_error"] = (
            current_predictions["actual"]
            - current_predictions["reference_prediction"]
        )
        current_predictions["reference_squared_error"] = (
            current_predictions["reference_error"] ** 2
        )
        prediction_results.append(current_predictions)

        print(
            f"      MAE={mae:.6f}, RMSE={rmse:.6f}, "
            f"Bias={bias:.6f}, rRMSE={relative_rmse:.6f}, "
            f"fit={timedelta(seconds=int(fit_seconds))}"
        )

        del model
        del X_fit
        del X_evaluation
        del predictions
        del current_predictions
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df["rmse_rank"] = (
        results_df["rmse"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    results_df = results_df.sort_values(
        ["rmse", "n_features"],
        ascending=[True, True],
    ).reset_index(drop=True)

    predictions_df = pd.concat(
        prediction_results,
        axis=0,
        ignore_index=True,
        copy=False,
    )

    return results_df, predictions_df


def save_household_results(
    household_id: int,
    selected_windows: list[int],
    feature_sets: dict[str, list[str]],
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    split_rows: dict[str, int],
    total_seconds: float,
) -> None:
    """Save detailed final-test outputs for one household."""
    output_directory = get_final_evaluation_directory(household_id)
    output_directory.mkdir(parents=True, exist_ok=True)

    results.to_pickle(output_directory / "final_test_results.pkl")
    results.to_csv(
        output_directory / "final_test_results.csv",
        index=False,
    )
    predictions.to_pickle(output_directory / "test_predictions.pkl")
    predictions.to_csv(
        output_directory / "test_predictions.csv",
        index=False,
    )

    run_info = {
        "household": household_id,
        "target": TARGET,
        "horizon": HORIZON,
        "training_splits": list(FINAL_TRAINING_SPLITS),
        "evaluation_split": "test",
        "metrics": ["mae", "rmse", "bias", "rrmse"],
        "reference_model": (
            "24-hour naive forecast: target_t96_hat = lag_96 = y_t"
        ),
        "selected_windows": selected_windows,
        "rf_params": RF_PARAMS,
        "feature_set_order": list(feature_sets),
        "feature_set_sizes": {
            name: len(features)
            for name, features in feature_sets.items()
        },
        "split_rows": split_rows,
        "train_validation_rows": sum(
            split_rows[split_name]
            for split_name in FINAL_TRAINING_SPLITS
        ),
        "test_rows": split_rows["test"],
        "feature_sets_file": str(get_feature_sets_file(household_id)),
        "test_feature_file": str(
            get_feature_file(
                household_id=household_id,
                dataset_split="test",
                windows=WINDOWS,
                horizon=HORIZON,
            )
        ),
        "total_seconds": total_seconds,
        "total_duration": str(
            timedelta(seconds=int(total_seconds))
        ),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with (output_directory / "run_info.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_info, file, indent=2)


def process_household(household_id: int) -> pd.DataFrame:
    """Run final evaluation for one household."""
    total_start = time.perf_counter()
    selected_windows, feature_sets = load_feature_sets(household_id)
    required_features = feature_sets["all"]

    print()
    print(f"Household {household_id}")
    print(f"  Selected windows: {selected_windows}")
    print(f"  Feature sets: {list(feature_sets)}")

    split_features: dict[DatasetSplit, pd.DataFrame] = {}
    split_targets: dict[DatasetSplit, pd.Series] = {}

    for split_name in FINAL_TRAINING_SPLITS:
        X_split, y_split, _ = load_split(
            household_id=household_id,
            dataset_split=split_name,
            required_features=required_features,
        )
        split_features[split_name] = X_split
        split_targets[split_name] = y_split

    X_test, y_test, test_metadata = load_split(
        household_id=household_id,
        dataset_split="test",
        required_features=required_features,
        include_metadata=True,
    )

    if test_metadata is None:
        raise RuntimeError("Test metadata were not created")

    validate_splits(
        split_features=split_features,
        X_test=X_test,
    )

    split_rows = {
        split_name: len(split_features[split_name])
        for split_name in FINAL_TRAINING_SPLITS
    }
    split_rows["test"] = len(X_test)

    X_train_validation = pd.concat(
        [split_features[name] for name in FINAL_TRAINING_SPLITS],
        axis=0,
        copy=False,
    )
    y_train_validation = pd.concat(
        [split_targets[name] for name in FINAL_TRAINING_SPLITS],
        axis=0,
        copy=False,
    )

    del split_features
    del split_targets
    gc.collect()

    print(
        f"  {' + '.join(FINAL_TRAINING_SPLITS)}: "
        f"{len(X_train_validation):,} rows"
    )

    results, predictions = evaluate_feature_sets(
        X_train_validation=X_train_validation,
        y_train_validation=y_train_validation,
        X_test=X_test,
        y_test=y_test,
        test_metadata=test_metadata,
        feature_sets=feature_sets,
    )

    results.insert(0, "household", household_id)
    predictions.insert(0, "household", household_id)
    total_seconds = time.perf_counter() - total_start

    save_household_results(
        household_id=household_id,
        selected_windows=selected_windows,
        feature_sets=feature_sets,
        results=results,
        predictions=predictions,
        split_rows=split_rows,
        total_seconds=total_seconds,
    )

    print()
    print("  Test results:")
    print(
        results[
            [
                "rmse_rank",
                "feature_set",
                "n_features",
                "mae",
                "rmse",
                "bias",
                "reference_rmse",
                "rrmse",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )
    print(
        f"  Total duration: "
        f"{timedelta(seconds=int(total_seconds))}"
    )
    print(
        f"  Results saved to: "
        f"{get_final_evaluation_directory(household_id)}"
    )

    del X_train_validation
    del y_train_validation
    del X_test
    del y_test
    del test_metadata
    del predictions
    gc.collect()

    return results


def save_combined_results(results: pd.DataFrame) -> None:
    """Save per-household results and an aggregate feature-set summary."""
    FINAL_EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results.to_pickle(
        FINAL_EVALUATION_DIRECTORY / "all_households_results.pkl"
    )
    results.to_csv(
        FINAL_EVALUATION_DIRECTORY / "all_households_results.csv",
        index=False,
    )

    summary = (
        results.groupby("feature_set", as_index=False)
        .agg(
            n_households=("household", "nunique"),
            mean_n_features=("n_features", "mean"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            median_rmse=("rmse", "median"),
            mean_bias=("bias", "mean"),
            std_bias=("bias", "std"),
            mean_reference_rmse=("reference_rmse", "mean"),
            mean_rrmse=("rrmse", "mean"),
            std_rrmse=("rrmse", "std"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .sort_values("mean_rmse", ascending=True)
        .reset_index(drop=True)
    )

    summary.to_pickle(
        FINAL_EVALUATION_DIRECTORY / "feature_set_summary.pkl"
    )
    summary.to_csv(
        FINAL_EVALUATION_DIRECTORY / "feature_set_summary.csv",
        index=False,
    )

    print()
    print("Summary across processed households:")
    print(summary.to_string(index=False))


def validate_configuration() -> None:
    """Prevent accidental leakage and inconsistent feature-set settings."""
    if "test" in FINAL_TRAINING_SPLITS:
        raise ValueError(
            "FINAL_TRAINING_SPLITS must not contain the test split."
        )

    if set(FINAL_TRAINING_SPLITS) != {"train", "validation"}:
        raise ValueError(
            "FINAL_TRAINING_SPLITS must contain train and validation."
        )

    if not HOUSEHOLD_IDS:
        raise ValueError("HOUSEHOLD_IDS must contain at least one value.")

    if len(HOUSEHOLD_IDS) != len(set(HOUSEHOLD_IDS)):
        raise ValueError("HOUSEHOLD_IDS contains duplicate identifiers.")

    if HORIZON != 96:
        raise ValueError(
            "The configured naive reference feature lag_96 requires "
            "HORIZON = 96."
        )


def main() -> None:
    validate_configuration()
    households = list(HOUSEHOLD_IDS)

    print(f"Households: {households}")
    print("Final models will be evaluated on the test sets.")
    print("Do not change the model configuration after viewing the results.")

    household_results = []

    for household_id in households:
        household_results.append(process_household(household_id))

    combined_results = pd.concat(
        household_results,
        axis=0,
        ignore_index=True,
        copy=False,
    )
    save_combined_results(combined_results)

    print()
    print("Final evaluation completed for all configured households.")


if __name__ == "__main__":
    main()
