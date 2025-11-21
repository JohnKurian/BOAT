# BOAT: Bayesian Optimization AutoML Time-series Framework

Professional, high-quality AutoML framework for time-series forecasting.

## Overview

BOAT is a framework designed to automate the process of selecting and tuning time-series forecasting models. It supports various algorithms including:
- **Deep Learning**: LSTM (Univariate & Multivariate), TCN, N-Beats
- **Machine Learning**: Random Forest, KNN (via sktime)
- **Statistical**: Theta Forecaster

## Architecture

The project is structured as a Python package `boat` with the following components:
- `boat.core`: Core engine for running AutoML pipelines.
- `boat.models`: Implementations of forecasting models.
- `boat.api`: Flask-based REST API for managing experiments.
- `boat.ui`: Streamlit-based user interface.
- `boat.utils`: Utility modules for database, metrics, etc.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Copy `.env.example` to `.env` and update the values if necessary.
   ```bash
   cp .env.example .env
   ```

## Usage

### Starting the Backend (API)
The API handles experiment creation and management.
```bash
python -m boat.api.app
```
The API will be available at `http://localhost:8000`.

### Starting the Frontend (UI)
The UI allows you to interact with the framework, create experiments, and visualize results.
```bash
streamlit run boat/ui/streamlit_app.py
```
The UI will be available at `http://localhost:8501`.

### Starting MLFlow
MLFlow is used for experiment tracking.
```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
```

## Development

### Running Tests
(To be implemented)
```bash
pytest tests/
```

## Legacy Code
Old implementation files have been moved to `legacy_code/` for reference.
