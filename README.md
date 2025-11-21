# 🚤 BOAT: Bayesian Optimization AutoML Time-series Framework

> **A reusable and user-extensible AutoML framework with native support for time-series data.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📖 Overview

**BOAT** is a state-of-the-art AutoML framework designed specifically for industrial time-series applications. While existing AutoML solutions excel at image, text, and tabular data, they often lack native support for the complex time-series data found in industrial settings (e.g., sensor readings, energy consumption).

BOAT bridges this gap by providing a robust, extensible platform that automates the selection and tuning of time-series forecasting models. It leverages **Bayesian Optimization** combined with **Monte Carlo Tree Search (LA-MCTS)** to efficiently navigate high-dimensional search spaces.

## ✨ Key Features

*   **🚀 High-Dimensional Optimization**: Utilizes a state-of-the-art Bayesian optimization-based algorithm (LA-MCTS) that performs efficiently even with large numbers of models and hyperparameters.
*   **🔥 Warm Starting**: Includes a meta-learning module that learns from previous datasets. It uses meta-features and cosine similarity to "warm start" the optimization in a promising region, significantly accelerating convergence.
*   **📈 Native Time-Series Support**: Built-in support for advanced time-series models including **LSTM**, **TCN**, **N-Beats**, and statistical methods like **Theta Forecaster**.
*   **🛠️ Extensible API**: A flexible Python API that allows users to define custom data loaders, preprocessing steps, model functions, and search spaces.
*   **🖥️ Interactive UI**: A centralized Streamlit-based dashboard for managing experiments, visualizing results, and deploying models.

## 🏗️ Architecture

The framework is structured as a modular Python package `boat`:

| Component | Description |
|-----------|-------------|
| **`boat.core`** | The heart of the framework, containing the **LA-MCTS** optimizer and the **Hyperspace Controller**. |
| **`boat.models`** | A repository of forecasting models, including Deep Learning (LSTM, TCN, N-Beats) and Sktime wrappers. |
| **`boat.api`** | A **Flask**-based REST API for experiment management and model deployment. |
| **`boat.ui`** | A **Streamlit** frontend for easy interaction and visualization. |
| **`boat.utils`** | Utility modules for data processing, metrics (TSFEL), and database interactions. |
| **MLOps Layer** | Built on **MLFlow** for experiment tracking, model versioning, and artifact storage. |

## 🤖 Supported Models

BOAT supports a wide range of algorithms out-of-the-box:

*   **Deep Learning**:
    *   Long Short-Term Memory (LSTM) - Univariate & Multivariate
    *   Temporal Convolutional Networks (TCN)
    *   N-Beats
*   **Machine Learning (via Sktime)**:
    *   Random Forest Regressor
    *   K-Nearest Neighbors (KNN)
*   **Statistical**:
    *   Theta Forecaster

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB (for experiment metadata)
- MLFlow (for tracking)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/BOAT.git
    cd BOAT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Copy the example environment file and update the values (e.g., MongoDB URI).
    ```bash
    cp .env.example .env
    ```

### Running the Framework

1.  **Start MLFlow Server:**
    ```bash
    mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
    ```

2.  **Start the Backend API:**
    ```bash
    python -m boat.api.app
    ```
    *API running at `http://localhost:8000`*

3.  **Start the User Interface:**
    ```bash
    streamlit run boat/ui/streamlit_app.py
    ```
    *UI running at `http://localhost:8501`*

## 📚 Citation

If you use BOAT in your research, please cite our paper:

```bibtex
@inproceedings{kurian2021boat,
  title={BOAT: A Bayesian Optimization AutoML Time-series Framework for Industrial Applications},
  author={Kurian, John Joy and Dix, Marcel and Amihai, Ido and Ceusters, Glenn and Prabhune, Ajinkya},
  booktitle={IEEE International Conference on Industrial Informatics (INDIN)},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by ABB Corporate Research Center, Industrial Data Analytics.*
