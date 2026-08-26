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
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)
from sklearn.ensemble import RandomForestRegressor

from config import (
    HORIZON,
    HOUSEHOLD_IDS,
    WINDOW_RANKING_SPLIT,
    WINDOWS,
    get_feature_file,
    get_target_name,
    get_window_ranking_directory,
)


TOP_N_WINDOWS = 4

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


def is_tsfresh_feature(
    column: object,
) -> bool:
    """Return whether a column is a prefixed TSFresh feature."""
    return (
        re.match(
            r"^r\d+_",
            str(column),
        )
        is not None
    )


def load_data(
    data_file: Path,
    target: str,
    windows: Sequence[int],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    dict[int, list[str]],
    list[str],
]:
    """Load a training feature dataset and prepare X and y."""
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}"
        )

    dataframe = pd.read_pickle(
        data_file
    )

    if not dataframe.columns.is_unique:
        duplicates = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate columns: {duplicates}"
        )

    if target not in dataframe.columns:
        raise ValueError(
            f"Target column not found: {target}"
        )

    features_by_window: dict[
        int,
        list[str],
    ] = {}

    for window in windows:
        prefix = f"r{window}_"

        window_features = [
            column
            for column in dataframe.columns
            if str(column).startswith(prefix)
        ]

        if not window_features:
            raise ValueError(
                f"No TSFresh features with prefix "
                f"'{prefix}' were found."
            )

        features_by_window[
            window
        ] = window_features

    feature_counts = {
        window: len(features)
        for window, features
        in features_by_window.items()
    }

    if len(set(feature_counts.values())) != 1:
        raise ValueError(
            "Windows do not contain the same number "
            "of TSFresh features: "
            f"{feature_counts}. Summed importances "
            "would not be directly comparable."
        )

    all_target_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith(
            "target_t"
        )
    ]

    manual_features = []

    for column in dataframe.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        if column in all_target_columns:
            continue

        if is_tsfresh_feature(column):
            continue

        if (
            is_numeric_dtype(
                dataframe[column]
            )
            or is_bool_dtype(
                dataframe[column]
            )
        ):
            manual_features.append(
                column
            )

    if not manual_features:
        raise ValueError(
            "No manual features were found."
        )

    tsfresh_features = [
        feature
        for window in windows
        for feature
        in features_by_window[window]
    ]

    feature_columns = [
        *manual_features,
        *tsfresh_features,
    ]

    if len(feature_columns) != len(
        set(feature_columns)
    ):
        raise ValueError(
            "Feature names are not unique."
        )

    X = dataframe.loc[
        :,
        feature_columns,
    ].copy()

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
        if not np.isfinite(
            X[column].to_numpy(
                copy=False
            )
        ).all()
    ]

    if invalid_columns:
        raise ValueError(
            "Feature matrix contains NaN or "
            "infinite values in columns: "
            f"{invalid_columns[:20]}"
        )

    if not np.isfinite(
        y.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "Target contains NaN or "
            "infinite values."
        )

    print(f"Rows: {len(X):,}")

    print(
        f"Manual features: "
        f"{len(manual_features):,}"
    )

    print(
        f"TSFresh features: "
        f"{len(tsfresh_features):,}"
    )

    print(
        f"All model features: "
        f"{X.shape[1]:,}"
    )

    print(
        f"TSFresh features per window: "
        f"{feature_counts}"
    )

    return (
        X,
        y,
        features_by_window,
        manual_features,
    )


def calculate_rankings(
    X: pd.DataFrame,
    y: pd.Series,
    features_by_window: dict[
        int,
        list[str],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[int],
    float,
]:
    """
    Fit a random forest and calculate feature and window rankings.
    """
    model = RandomForestRegressor(
        **RF_PARAMS
    )

    start_time = time.perf_counter()

    model.fit(
        X,
        y,
    )

    fit_seconds = (
        time.perf_counter()
        - start_time
    )

    feature_to_window = {
        feature: window
        for window, features
        in features_by_window.items()
        for feature in features
    }

    feature_ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    feature_ranking["window"] = (
        pd.array(
            [
                feature_to_window.get(
                    feature,
                    pd.NA,
                )
                for feature
                in feature_ranking[
                    "feature"
                ]
            ],
            dtype="Int64",
        )
    )

    feature_ranking["source"] = (
        np.where(
            feature_ranking[
                "window"
            ].isna(),
            "manual",
            "tsfresh",
        )
    )

    feature_ranking = (
        feature_ranking
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(feature_ranking) + 1,
        ),
    )

    window_ranking = (
        feature_ranking
        .dropna(
            subset=["window"]
        )
        .groupby(
            "window",
            as_index=False,
        )
        .agg(
            importance=(
                "importance",
                "sum",
            ),
            n_features=(
                "feature",
                "size",
            ),
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    window_ranking[
        "window"
    ] = (
        window_ranking[
            "window"
        ].astype(int)
    )

    total_importance = (
        window_ranking[
            "importance"
        ].sum()
    )

    if total_importance <= 0:
        raise ValueError(
            "Total window importance "
            "is zero."
        )

    window_ranking[
        "relative_importance"
    ] = (
        window_ranking[
            "importance"
        ]
        / total_importance
    )

    window_ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(window_ranking) + 1,
        ),
    )

    selected_windows = (
        window_ranking
        .head(
            TOP_N_WINDOWS
        )["window"]
        .astype(int)
        .tolist()
    )

    del model
    gc.collect()

    return (
        feature_ranking,
        window_ranking,
        selected_windows,
        fit_seconds,
    )


def save_results(
    output_directory: Path,
    data_file: Path,
    target: str,
    windows: Sequence[int],
    feature_ranking: pd.DataFrame,
    window_ranking: pd.DataFrame,
    selected_windows: list[int],
    fit_seconds: float,
    X: pd.DataFrame,
    manual_features: list[str],
) -> None:
    """
    Save feature ranking, window ranking, and run information.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_ranking.to_pickle(
        output_directory
        / "feature_ranking.pkl"
    )

    feature_ranking.to_csv(
        output_directory
        / "feature_ranking.csv",
        index=False,
    )

    window_ranking.to_pickle(
        output_directory
        / "window_ranking.pkl"
    )

    window_ranking.to_csv(
        output_directory
        / "window_ranking.csv",
        index=False,
    )

    with (
        output_directory
        / "selected_windows.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selected_windows": (
                    selected_windows
                ),
            },
            file,
            indent=2,
        )

    manual_importance = float(
        feature_ranking.loc[
            (
                feature_ranking[
                    "source"
                ]
                == "manual"
            ),
            "importance",
        ].sum()
    )

    run_info = {
        "data_file": str(
            data_file
        ),
        "target": target,
        "windows": list(
            windows
        ),
        "rf_params": RF_PARAMS,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "n_manual_features": (
            len(manual_features)
        ),
        "manual_importance": (
            manual_importance
        ),
        "selected_windows": (
            selected_windows
        ),
        "fit_seconds": (
            fit_seconds
        ),
        "fit_duration": str(
            timedelta(
                seconds=int(
                    fit_seconds
                )
            )
        ),
        "timestamp": (
            pd.Timestamp.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
    }

    with (
        output_directory
        / "run_info.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_info,
            file,
            indent=2,
        )


def process_household(
    household_id: int,
) -> None:
    """Calculate and save window rankings for one household."""
    target = get_target_name(
        HORIZON
    )

    data_file = get_feature_file(
        household_id=household_id,
        dataset_split=(
            WINDOW_RANKING_SPLIT
        ),
        windows=WINDOWS,
        horizon=HORIZON,
    )

    output_directory = (
        get_window_ranking_directory(
            household_id
        )
    )

    print()

    print(
        "=== Ranking windows for "
        f"household {household_id} ==="
    )

    print(
        f"Input file: {data_file}"
    )

    (
        X,
        y,
        features_by_window,
        manual_features,
    ) = load_data(
        data_file=data_file,
        target=target,
        windows=WINDOWS,
    )

    (
        feature_ranking,
        window_ranking,
        selected_windows,
        fit_seconds,
    ) = calculate_rankings(
        X=X,
        y=y,
        features_by_window=(
            features_by_window
        ),
    )

    save_results(
        output_directory=(
            output_directory
        ),
        data_file=data_file,
        target=target,
        windows=WINDOWS,
        feature_ranking=(
            feature_ranking
        ),
        window_ranking=(
            window_ranking
        ),
        selected_windows=(
            selected_windows
        ),
        fit_seconds=fit_seconds,
        X=X,
        manual_features=(
            manual_features
        ),
    )

    print()
    print("Window ranking:")

    print(
        window_ranking.to_string(
            index=False
        )
    )

    print()

    print(
        f"Selected windows: "
        f"{selected_windows}"
    )

    print(
        "Fit duration: "
        f"{timedelta(seconds=int(fit_seconds))}"
    )

    print(
        f"Results saved to: "
        f"{output_directory}"
    )

    del X
    del y
    del features_by_window
    del manual_features
    del feature_ranking
    del window_ranking
    gc.collect()


def main() -> None:
    """Rank rolling windows for every configured household."""
    if WINDOW_RANKING_SPLIT != "train":
        raise ValueError(
            "WINDOW_RANKING_SPLIT must be "
            "'train' to avoid using validation "
            "or test data for window ranking."
        )

    for household_id in HOUSEHOLD_IDS:
        process_household(
            household_id
        )


if __name__ == "__main__":
    main()