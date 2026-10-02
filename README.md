[Română](README.ro.md)

# Energy Load Forecast

Hourly electricity load forecasting for Romania using historical data from the ENTSO-E Transparency Platform.

## Project overview

The project forecasts electricity load for Romania (`RO`) at hourly resolution with a 24-hour forecast horizon.

The final workflow includes:

- Naive baseline
- Seasonal Naive baseline
- LightGBM
- N-BEATS
- P10-P90 prediction intervals with LightGBM quantile models
- MAE, RMSE, and MAPE for point forecasts
- Pinball loss, empirical coverage, and interval width for prediction intervals
- Feature importance and residual/error analysis

The final experiment uses chronological train, validation, and test windows. The test window contains 24 hourly observations.

## Data

Historical electricity consumption is obtained from the ENTSO-E Transparency Platform.

The ingestion layer supports:

- ENTSO-E CSV data
- The ENTSO-E API through `entsoe-py`

Timestamps are normalized to UTC and the load data is resampled to hourly resolution.

The final experiment covers July 2025 through December 2025. Since the dataset contains only part of one year, yearly seasonality is treated as a limitation rather than estimated as a complete annual pattern.

## Project structure

```text
energy_load_forecast/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── README.ro.md
├── uv.lock
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── all_models_comparison_plot.png
│   ├── final_comparison_plot.png
│   ├── final_report.md
│   ├── final_report.ro.md
│   └── results.md
├── notebooks/
│   ├── 00_explore_raw_csv.ipynb
│   ├── 01_api_ingestion.ipynb
│   ├── 02_preprocessing_exploration.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_nbeats.ipynb
│   ├── 08_error_analysis.ipynb
│   └── 09_final_results.ipynb
├── src/
│   └── energy_load_forecast/
│       ├── __init__.py
│       ├── config.py
│       ├── error_analysis.py
│       ├── evaluation.py
│       ├── features.py
│       ├── ingestion.py
│       ├── interval_evaluation.py
│       ├── preprocessing.py
│       └── models/
└── tests/
    ├── test_baselines.py
    ├── test_error_analysis.py
    ├── test_evaluation.py
    ├── test_features.py
    ├── test_ingestion.py
    ├── test_interval_evaluation.py
    ├── test_lightgbm_model.py
    ├── test_nbeats_model.py
    └── test_preprocessing.py
```

The notebooks preserve the original analysis and experimentation history. The Python modules under `src/` contain the reusable implementation extracted from that work.

## Environment and installation

This project uses [uv](https://docs.astral.sh/uv/).

Create the environment and install the locked dependencies with:

```bash
uv sync
```

The project requires Python 3.14 or newer.

Development dependencies, including `pytest`, `ruff`, and `ipykernel`, are installed by `uv sync`.

## ENTSO-E API key

For API ingestion, create a local `.env` file based on `.env.example` and set your ENTSO-E API key:

```text
ENTSOE_API_KEY=your_key_here
```

Do not commit `.env` or the API key.

## Running tests

Run the full test suite with:

```bash
uv run pytest
```

The tests cover ingestion, preprocessing, feature engineering, baselines, model wrappers, evaluation, interval evaluation, and error analysis.

## Running the notebooks

Start Jupyter through the project environment:

```bash
uv run jupyter lab
```

The notebooks follow the project workflow:

1. `00_explore_raw_csv.ipynb` explores the raw CSV data.
2. `01_api_ingestion.ipynb` handles ENTSO-E API ingestion.
3. `02_preprocessing_exploration.ipynb` checks data cleaning and preprocessing.
4. `03_eda.ipynb` contains exploratory data analysis and seasonality analysis.
5. `04_baselines.ipynb` evaluates the required baseline models.
6. `05_feature_engineering.ipynb` builds calendar, lag, and rolling features.
7. `06_lightgbm.ipynb` develops the LightGBM model.
8. `07_nbeats.ipynb` develops the N-BEATS model.
9. `08_error_analysis.ipynb` analyzes forecast errors and model behavior.
10. `09_final_results.ipynb` runs the final comparison and generates the main deliverables.

## Final analysis

Run `notebooks/09_final_results.ipynb` after the environment and data are configured.

The final notebook:

1. loads or fetches the hourly Romanian load series;
2. prepares chronological train, validation, and test windows;
3. evaluates Naive and Seasonal Naive baselines;
4. trains and evaluates LightGBM;
5. evaluates the P10-P90 LightGBM prediction interval;
6. trains and evaluates N-BEATS;
7. writes the final metrics to `docs/results.md`;
8. saves comparison and diagnostic figures to `docs/`.

LightGBM test forecasting is recursive, so future actual load values are not used as future lag or rolling inputs.

N-BEATS training can take longer than the other models. Final N-BEATS results should be taken from a completed run of `09_final_results.ipynb`.

## Final results

The final test comparison is based on the same 24-hour test window for all models.

The generated metrics are stored in:

```text
docs/results.md
```

The main figures are stored in:

```text
docs/all_models_comparison_plot.png
docs/final_comparison_plot.png
```

The full written analysis is available in:

```text
docs/final_report.md
docs/final_report.ro.md
```

## Reproducibility

The repository includes `uv.lock`, so dependency versions are locked.

A typical setup from a fresh checkout is:

```bash
uv sync
uv run pytest
uv run jupyter lab
```

Then open `notebooks/09_final_results.ipynb` and run the notebook from top to bottom.

The final test horizon must contain exactly 24 hourly observations.
