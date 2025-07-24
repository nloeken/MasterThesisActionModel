# Master Thesis: Context-Aware Modeling of Player Decisions in Football Using Machine Learning and Expected Threat Analysis

This repository contains the code and data processing pipeline explained in the corresponding thesis.

## Overview

The goal of this project is to build a machine learning model that predicts the next action in a football game using a combination of event data (e.g., pass, shot, dribbling) and respective positional tracking data, both optained from Statsbomb Open data.  
The core model is implemented using **XGBoost**, a powerful gradient boosting framework.

## Data

All data is retrieved from the Statsbomb open data repository.
It can be downloaded [here](https://github.com/statsbomb/open-data) and should be stored locally. The path should be configured in the `config` file.

## Methods & Tools

- Python 3.10+
- XGBoost, SHAP
- Statsbombpy
- Pandas, NumPy
- Scikit-learn
- Matplotlib, mplsoccer (for visualizations)
- Tqdm

## Project Structure

```
XGBOOST/
├── plot_events.py              # exploratory data analysis: plot events
├── config.py                   # configuration file for data sources
├── utils.py                    # helper functions 
├── data_loading.py             # loading raw data
├── data_preprocessing.py       # preprocessing raw data
├── feature_engineering.py      # define and calculate features
├── model_training.py/          # train XGBoost model
├── model_evaluation.py         # evaluate model performance
├── requirements.txt            # Python dependencies
├── run_all.py                  # Executes whole model code
└── README.md             
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nloeken/xgboost.git
```
### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```



