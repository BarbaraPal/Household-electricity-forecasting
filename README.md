# Household Electricity Consumption Forecasting

This repository contains the code used in a master's thesis on short-term household electricity consumption forecasting. The study investigates how the construction and selection of input features affect forecasting accuracy and model training time.

Electricity consumption is forecast separately for each household using a random forest regressor. The forecasting horizon is 96 fifteen-minute intervals, corresponding to 24 hours ahead.

## Data

The dataset was provided by Elektro Maribor. The analysis uses electricity consumption measurements recorded at 15-minute intervals from 1 January 2024 00:15 to 1 January 2026 00:15.

Due to privacy restrictions, the complete dataset cannot be published. The repository includes a sample dataset for one household for which permission for public release was obtained. The sample data are available in:

```text
data/dfs/df_1.csv
```

The target variable is electricity consumption measured in kilowatt-hours (kWh). The preprocessing script restricts the input data to the analysis period and converts timestamps to fixed Central European Time (CET, UTC+1).

## Methodology

For each forecast-origin time $t$, the model predicts electricity consumption 96 time steps ahead:

$$
y_{t+96}
$$

The input data contain three groups of features:

- historical electricity consumption values;
- calendar features calculated for the target time;
- automatically extracted time-series features calculated with `tsfresh`.

The historical features represent electricity consumption 24 hours and seven days before the target time.

Calendar features include cyclical representations of the hour, day of the week, month and day of the year, as well as indicators for weekends, Sundays and holidays in Slovenia.

The `tsfresh` features are calculated from nine rolling windows. The window lengths are 4, 8, 12, 16, 24, 32, 48, 96 and 672 observations, corresponding to periods from one hour to seven days. With the package versions listed in `requirements.txt`, 777 `tsfresh` features are produced for each rolling-window length.

Missing or infinite values produced during feature extraction are replaced with zero using `impute_dataframe_zero()`.

The selection procedure is performed separately for each household:

1. The nine rolling-window lengths are ranked using aggregated random forest feature importances.
2. The four highest-ranked window lengths are retained.
3. All 15 non-empty combinations of these four window lengths are evaluated on the validation set.
4. Features from the selected windows are ranked using random forest feature importances.
5. Final models are trained using the Top 10, Top 24, Top 50, Top 100, Top 200 and complete feature sets.

The auxiliary random forest models used for window and feature ranking contain 50 trees. The final random forest models contain 200 trees. All models use `max_features="sqrt"` and `random_state=0`.

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
├── requirements.txt
├── data/
│   └── dfs/
│       └── df_1.csv
├── .gitignore
├── LICENSE
└── README.md
```

The scripts have the following purposes:

| File | Description |
|---|---|
| `config.py` | Defines shared paths, dataset splits, the forecasting horizon, rolling-window lengths and experiment settings. |
| `data_preprocessing.py` | Loads, validates and chronologically divides the electricity consumption data into training, validation and test sets. |
| `feature_engineer.py` | Constructs historical and calendar features and extracts rolling-window features using `tsfresh`. |
| `window_rank.py` | Produces the preliminary ranking of rolling-window lengths. |
| `window_selection.py` | Evaluates combinations of the four highest-ranked rolling-window lengths on the validation set. |
| `feature_rank.py` | Ranks the available input features and constructs feature sets of different sizes. |
| `final_evaluation.py` | Trains the final models and evaluates them on the test set. |

## Data splitting

The data are divided chronologically according to the timestamp of the target value:

- training targets: before 30 June 2025 00:00;
- validation targets: from 1 July 2025 00:00 to before 30 September 2025 00:00;
- test targets: from 1 October 2025 00:00 onward.

A 24-hour gap is maintained between consecutive sets to preserve their temporal separation.

Feature engineering is performed separately within the training, validation and test sets. Measurements from a preceding set are not used to construct rolling windows in the following set.

Because a complete rolling window of 672 observations and a target value 96 time steps ahead are required, the effective target periods are:

- training: 9 January 2024 00:00–29 June 2025 23:45;
- validation: 8 July 2025 23:45–29 September 2025 23:45;
- test: 8 October 2025 23:45–1 January 2026 00:15.

## Installation

Python 3.11 or a newer compatible version is required.

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
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

Shared experiment settings and generated-output paths are defined in `config.py`. The raw input file mapping and preprocessing period are configured at the end of `data_preprocessing.py`.

The public configuration processes household 1. Additional household identifiers can be added to `HOUSEHOLD_IDS` in `config.py` when the corresponding private datasets are available.

Feature extraction with large rolling windows, particularly the window of 672 observations, can require substantial memory and processing time.

Generated intermediate files and results are saved in subdirectories of `data/`.

## Evaluation metrics

The final models are evaluated using:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- relative root mean squared error (rRMSE);
- forecast bias;
- model training time.

The relative RMSE compares the RMSE of the random forest predictions with the RMSE of a 24-hour naive forecast. For the forecasting horizon of 96 time steps, the reference prediction is the last value available at the forecast-origin time:

$$
\widehat{y}^{(\mathrm{ref})}_{t+96}=y_t
$$

A value of rRMSE below one indicates that the random forest model outperforms the reference forecast.

## Reproducibility

Random forest models use a fixed random seed to support reproducible results. All window-selection and feature-selection decisions are made using only the training and validation data. The test data do not influence rolling-window selection, feature ranking or model training.

The package versions required to run the code are listed in `requirements.txt`.

The publicly available sample enables the complete experimental pipeline to be run for household 1. Reproducing the aggregate results for all 11 households requires the complete private dataset.

## License

This project is available under the terms specified in the [LICENSE](LICENSE) file.

## Author

Barbara Pal  
Master's Programme in Financial Mathematics  
Faculty of Mathematics and Physics, University of Ljubljana