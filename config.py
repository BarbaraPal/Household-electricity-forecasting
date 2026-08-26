from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, cast


DatasetSplit = Literal["train", "validation", "test"]


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
FEATURE_DIRECTORY = PROJECT_ROOT / "data" / "features"
WINDOW_RANKING_DIRECTORY = PROJECT_ROOT / "data" / "window_ranking"
WINDOW_SELECTION_DIRECTORY = PROJECT_ROOT / "data" / "window_selection"
FEATURE_RANKING_DIRECTORY = PROJECT_ROOT / "data" / "feature_ranking"
FINAL_EVALUATION_DIRECTORY = PROJECT_ROOT / "data" / "final_evaluation"
RESULT_ANALYSIS_DIRECTORY = PROJECT_ROOT / "data" / "result_analysis"

FEATURE_ENGINEERING_RUN_INFO_FILE = (
    PROJECT_ROOT / "data" / "feature_engineer_runs.json"
)

# The public example processes household 1. Add the remaining identifiers when
# running the private full experiment, for example: (1, 2, 3, 4).
HOUSEHOLD_IDS = (1,)

# Feature engineering is run separately for every chronological subset.
FEATURE_ENGINEERING_SPLITS: tuple[DatasetSplit, ...] = (
    "train",
    "validation",
    "test",
)

# Preliminary window ranking must use only the training subset.
WINDOW_RANKING_SPLIT: DatasetSplit = "train"

# Final feature ranking and final model training use both subsets. The test
# subset remains reserved exclusively for final evaluation.
FINAL_TRAINING_SPLITS: tuple[DatasetSplit, ...] = (
    "train",
    "validation",
)

FREQUENCY = "15min"
HORIZON = 96

WINDOWS = (
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    96,
    672,
)

TOP_FEATURE_COUNTS = (
    10,
    24,
    50,
    100,
    200,
)

# Feature sets evaluated by the final models. Their order is kept identical in
# saved predictions and printed output.
FEATURE_SET_ORDER = tuple(
    f"top_{feature_count}"
    for feature_count in TOP_FEATURE_COUNTS
) + ("all",)

# The period boundaries are inclusive during preprocessing. The one-day gaps
# between consecutive subsets are intentional.
SPLIT_PERIODS: dict[DatasetSplit, tuple[str, str]] = {
    "train": (
        "2024-01-01 00:15:00",
        "2025-06-29 23:45:00",
    ),
    "validation": (
        "2025-07-01 00:00:00",
        "2025-09-29 23:45:00",
    ),
    "test": (
        "2025-10-01 00:00:00",
        "2026-01-01 00:15:00",
    ),
}


def validate_split(dataset_split: str) -> DatasetSplit:
    """Validate and normalize a dataset-split name."""
    aliases = {
        "val": "validation",
    }

    normalized_split = aliases.get(dataset_split, dataset_split)

    if normalized_split not in SPLIT_PERIODS:
        raise ValueError(
            "dataset_split must be 'train', 'validation', or 'test'."
        )

    return cast(DatasetSplit, normalized_split)


def get_dataset_name(household_id: int) -> str:
    """Return the standard dataset name for one household."""
    if household_id <= 0:
        raise ValueError("household_id must be a positive integer.")

    return f"df_{household_id}"


def get_target_name(horizon: int = HORIZON) -> str:
    """Return the target-column name for a forecast horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    return f"target_t{horizon}"


def get_windows_text(windows: Sequence[int] = WINDOWS) -> str:
    """Return rolling-window lengths formatted for a filename."""
    if not windows:
        raise ValueError("windows must contain at least one value.")

    return "_".join(str(window) for window in windows)


def get_split_period(dataset_split: str) -> tuple[str, str]:
    """Return the inclusive processing period for a dataset split."""
    normalized_split = validate_split(dataset_split)
    return SPLIT_PERIODS[normalized_split]


def get_processed_file(
    household_id: int,
    dataset_split: str,
) -> Path:
    """Return the processed CSV path for one household and split."""
    dataset_name = get_dataset_name(household_id)
    normalized_split = validate_split(dataset_split)

    return (
        PROCESSED_DATA_DIRECTORY
        / f"{dataset_name}_{normalized_split}.csv"
    )


def get_feature_file(
    household_id: int,
    dataset_split: str,
    windows: Sequence[int] = WINDOWS,
    horizon: int = HORIZON,
) -> Path:
    """Return the combined feature-file path."""
    dataset_name = get_dataset_name(household_id)
    normalized_split = validate_split(dataset_split)
    windows_text = get_windows_text(windows)
    target_name = get_target_name(horizon)

    return (
        FEATURE_DIRECTORY
        / (
            f"{dataset_name}_{normalized_split}_features_"
            f"{windows_text}_{target_name}.pkl"
        )
    )


def get_window_ranking_directory(household_id: int) -> Path:
    """Return the window-ranking output directory for one household."""
    return WINDOW_RANKING_DIRECTORY / get_dataset_name(household_id)


def get_window_selection_directory(household_id: int) -> Path:
    """Return the window-selection output directory for one household."""
    return WINDOW_SELECTION_DIRECTORY / get_dataset_name(household_id)


def get_best_window_combination_file(household_id: int) -> Path:
    """Return the selected window-combination file for one household."""
    return (
        get_window_selection_directory(household_id)
        / "best_window_combination.json"
    )


def get_feature_ranking_directory(household_id: int) -> Path:
    """Return the final feature-ranking output directory."""
    return FEATURE_RANKING_DIRECTORY / get_dataset_name(household_id)


def get_feature_sets_file(household_id: int) -> Path:
    """Return the final feature-set definition file."""
    return get_feature_ranking_directory(household_id) / "feature_sets.json"


def get_final_evaluation_directory(household_id: int) -> Path:
    """Return the final-evaluation output directory for one household."""
    return FINAL_EVALUATION_DIRECTORY / get_dataset_name(household_id)


def get_all_households_results_file() -> Path:
    """Return the combined final-test results file."""
    return FINAL_EVALUATION_DIRECTORY / "all_households_results.csv"


def get_result_analysis_table_directory() -> Path:
    """Return the directory for result-analysis tables."""
    return RESULT_ANALYSIS_DIRECTORY / "tables"


def get_result_analysis_figure_directory() -> Path:
    """Return the directory for result-analysis figures."""
    return RESULT_ANALYSIS_DIRECTORY / "figures"
