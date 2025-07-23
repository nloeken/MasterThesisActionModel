# Master Thesis: Context-Aware Modeling of Player Decisions in Football Using Machine Learning and Expected Threat Analysis

This repository contains the code and data processing pipeline explained in the corresponding thesis.

## Overview

The goal of this project is to build a machine learning model that predicts the next action in a football game using a combination of event data (e.g., pass, shot, dribbling) and respective positional tracking data, both optained from Statsbomb Open data.  
The core model is implemented using **XGBoost**, a powerful gradient boosting framework.

---

## Methods & Tools

- Python 3.10+
- XGBoost, SHAP
- Statsbombpy
- Pandas, NumPy
- Scikit-learn
- Matplotlib, mplsoccer (for visualizations)
- Tqdm

## Project Structure

 <pre> ```text project-root/ ├── main.py # Main training/evaluation script ├── data/ # Raw or preprocessed event/position data ├── models/ # Saved models ├── utils/ # Helper functions and scripts ├── notebooks/ # (Optional) Jupyter Notebooks for EDA or prototyping ├── requirements.txt # Python dependencies └── README.md # This file ``` </pre> 

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



