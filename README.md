# Household Electricity Consumption Forecasting

This repository contains the code used in a master's thesis on short-term household electricity consumption forecasting. The study investigates how the construction and selection of input features affect forecasting accuracy and model training time.

Electricity consumption is forecast separately for each household using a random forest regressor. The forecasting horizon is 96 fifteen-minute intervals, corresponding to 24 hours ahead.

## Data

The original dataset was provided by Elektro Maribor and contains electricity consumption measurements recorded at 15-minute intervals between January 2024 and January 2026.

Due to privacy restrictions, the complete dataset cannot be published. The repository includes a sample dataset for one household for which permission for public release was obtained. The sample data can be found in:

```text
data/dfs/
```

The target variable is electricity consumption measured in kilowatt-hours (kWh).

## Methodology

For each time point \(t\), the model predicts electricity consumption
96 time steps ahead:

$$
y_{t+96}.
$$

The input data contain three groups of features:

- lagged electricity consumption values;
- calendar features calculated for the target time;
- automatically extracted time-series features calculated with `tsfresh`.

Calendar features include cyclical representations of the hour, day of the week, day of the year and month, as well as indicators for weekends, Sundays and holidays in Slovenia.

The `tsfresh` features are calculated from nine rolling windows, ranging
from 4 observations (one hour) to 672 observations (seven days). The
window lengths are 4, 8, 12, 16, 24, 32, 48, 96 and 672 observations.

The selection procedure is performed separately for each household:

1. The nine rolling-window lengths are ranked using aggregated random forest feature importances.
2. The four highest-ranked window lengths are retained.
3. All 15 non-empty combinations of these four window lengths are evaluated on the validation set.
4. Features from the selected windows are ranked using random forest feature importances.
5. Final models are trained using the Top 10, Top 24, Top 50, Top 100, Top 200 and complete feature sets.

The test set is used only for the final evaluation.

## Repository structure

```text
.
├── config.py
├── data_preprocessing.py
├── feature_engineer.py
├── window_rank.py
├── window_selection.py
├── feature_rank.py
├── final_evaluation.py
├── data/
│   └── dfs/
├── LICENSE
└── README.md
```

The scripts have the following purposes:

| File | Description |
|---|---|
| `config.py` | Defines paths, dataset splits, forecasting horizon, rolling-window lengths and experiment settings. |
| `data_preprocessing.py` | Loads, validates and chronologically divides the electricity consumption data into training, validation and test sets. |
| `feature_engineer.py` | Constructs lagged and calendar features and extracts rolling-window features using `tsfresh`. |
| `window_rank.py` | Produces the preliminary ranking of rolling-window lengths. |
| `window_selection.py` | Evaluates combinations of the four highest-ranked window lengths on the validation set. |
| `feature_rank.py` | Ranks the available input features and constructs feature sets of different sizes. |
| `final_evaluation.py` | Trains the final models and evaluates them on the test set. |

## Data splitting

The data are divided chronologically to preserve their temporal ordering:

- training period: data before July 2025;
- validation period: July–September 2025;
- test period: October–December 2025.

The assignment of observations to the three sets is based on the timestamp of the target value. A gap corresponding to the 24-hour forecasting horizon is maintained between consecutive sets to prevent information leakage.

The exact temporal boundaries are defined in `config.py`.

## Installation

Python 3.10 or a newer compatible version is recommended.

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Running the pipeline

Run the scripts from the repository root in the following order:

```bash
python data_preprocessing.py
python feature_engineer.py
python window_rank.py
python window_selection.py
python feature_rank.py
python final_evaluation.py
```

The experiment settings, input paths and output paths can be modified in `config.py`.

Feature extraction with large rolling windows, particularly the window of 672 observations, can require substantial memory and processing time.

## Evaluation metrics

The final models are evaluated using:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- relative root mean squared error (rRMSE);
- forecast bias;
- Pearson correlation coefficient;
- model training time.

The relative RMSE compares the random forest predictions with a 24-hour naive forecast that uses the most recently available value for the same target horizon.

## Reproducibility

Random forest models use a fixed random seed to ensure reproducible results. All feature-selection decisions are made using only the training and validation data. The test data do not influence rolling-window selection, feature ranking or model training.

Results may still depend on the versions of Python and the libraries used. The package versions listed in `requirements.txt` should therefore be used when reproducing the experiments.

## License

This project is available under the terms specified in the [LICENSE](LICENSE) file.

## Author

Barbara Pal  
Master's programme in Financial Mathematics  
Faculty of Mathematics and Physics, University of Ljubljana