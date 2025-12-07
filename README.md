# EV
This app is a full end‑to‑end data science project for EV battery health.

You can:

Work with 3 built‑in synthetic datasets: Urban, Highway, Mixed.
Upload multiple CSV files → each becomes Upload_1, Upload_2, etc.
Analyse them individually, side‑by‑side, or all combined.
We intentionally include:

Missing data (MCAR + MAR)
Feature engineering (thermal, energy, stress)
Encoding (numeric, categorical, text → TF‑IDF)
Classical models (Linear Regression, RandomForest, GradientBoosting)
Advanced models (deep neural network MLP, XGBoost)
Time series forecasting (SOH)
Rich visualizations: histograms, box/violin, scatter, 3D scatter, scatter matrix, PCA, correlation & missingness heatmaps, etc.
Each tab matches the course topics (IDA/EDA, Missingness, Encoding, Regression, SVD/PCA, Time Series, NLP) and the final project rubric.

Quick tab guide
📖 Introduction: story & tab descriptions.
🏠 Summary: KPIs + dataset mix + key plots.
📦 Data Overview: table, data types, stats, per‑dataset summary.
📊 EDA & Viz Gallery: all the classic EDA plots.
🧩 Missingness Lab: missing patterns + imputation comparison.
🔁 Encoding & Classical Models: before/after encoding + RF/GB/LR + RF tuning.
🧠 Deep Learning & Ensembles: neural net (MLP) + XGBoost (if installed).
🔮 Predictions & Forecasting: RUL and SOH time‑series forecast.
🌍 Insights & Rubric: real‑world conclusions & rubric mapping.
💾 Export: download cleaned & engineered data for GitHub.


---

# Robust EV Battery SOH & RUL: A Missing-Data–Aware Analytics and Visualization Framework

## 📖 Project Overview

This project presents a comprehensive framework for **analyzing and forecasting the health of electric vehicle (EV) batteries**, focusing on **State of Health (SOH)** and **Remaining Useful Life (RUL)**.
The framework is **missing-data–aware**, supporting multiple datasets with advanced preprocessing, visualization, and predictive modeling. It is designed as a **user-friendly Streamlit web application** for interactive exploration, comparison, and insights generation.

---

## 🚀 Key Features

### 1. Multi-Dataset System

* Supports **three or more datasets**, including Urban, Highway, and Mixed usage cycles.
* Allows **single, multi, or combined dataset selection**.
* Automatically merges datasets and handles missing data.

### 2. Data Collection & Preparation

* Advanced **data cleaning and preprocessing**.
* Handles **numeric, categorical, and text features**.
* Implements **feature encoding**:

  * Numeric → scaling and imputation
  * Categorical → one-hot encoding
  * Text → TF-IDF vectorization
* Shows **raw vs encoded data tables** for complete transparency.

### 3. Exploratory Data Analysis (EDA) & Visualization

* **Over 10 different types of plots**:

  * Histograms, scatter plots, boxplots, violin plots
  * Parallel coordinates, correlation heatmaps
  * Multi-dataset comparison plots
* **Interactive visualizations** via Plotly for detailed exploration.
* Each plot includes **description and interpretation** of insights.

### 4. Data Processing & Feature Engineering

* Implements multiple **imputation methods** for missing data.
* Generates **derived features** and transformations.
* Visualizes **pre- and post-encoding feature distributions**.

### 5. Classical & Advanced Models

* Classical ML: **Linear Regression, Random Forest, Gradient Boosting**.
* Advanced ML:

  * Small **Keras MLP neural network** for regression.
  * Random Forest **hyperparameter tuning** with GridSearchCV.
* Shows **model performance comparison**:

  * Regression → MAE, R²
  * Classification → Accuracy
  * Includes **prediction vs actual plots** and **confusion matrices**.
* Neural network architecture **visualized diagrammatically**.

### 6. Real-World Insights

* Predictive models provide **SOH/RUL forecasts**.
* Recommendations for battery usage and maintenance.
* Tabular and visual summaries of **model outputs and forecasts**.

### 7. Streamlit App Features

* Multi-tab layout:

  1. **Introduction** – project explanation & tab guide
  2. **Data Overview** – dataset preview and summary statistics
  3. **EDA & Visualization** – plots and insights
  4. **Feature Engineering & Encoding** – shows pre/post encoding
  5. **Classical Models** – ML training, comparison, RF tuning
  6. **Advanced Models** – neural networks, ensemble models
  7. **Forecast / End Results** – predicted SOH/RUL
* **Caching** for faster loading.
* Fully interactive **sidebar for dataset selection**.

---



## 🧩 How to Use

* Select one or multiple datasets in the **sidebar**.
* Navigate through tabs to:

  * Inspect raw and encoded data
  * Explore visualizations
  * Train and compare classical models
  * Evaluate advanced neural networks
  * Review forecasts and recommendations
* Hover over plots for detailed information; all plots include captions describing insights.

---

## 📈 Visualizations & Model Insights

* Multi-colored, publication-quality plots.
* Comparative analysis of **Urban vs Highway vs Mixed** cycles.
* Neural network architecture diagrams.
* Model performance dashboards for **classic vs advanced models**.
* Parallel coordinates plots for **cycle-level comparisons**.
* Feature importance and hyperparameter tuning visualizations.

---



## 📌 Notes

* Designed for **robust missing-data handling**.
* Works with **any uploaded dataset** .
* Can be easily extended to other battery types or domains.

---



