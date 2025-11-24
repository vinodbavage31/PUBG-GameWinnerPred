# 🏆 Game Winner Prediction Project: Leveraging ML to Forecast Competitive Match Outcomes

## 💡 Project Overview

This project focuses on building a robust predictive model to forecast the outcome of competitive games (specifically using a large **PUBG match dataset**), leveraging detailed player and match statistics. The goal is to build machine learning models that can accurately predict the winning outcome (`winPlacePerc`).

## 🎯 Objectives

*   **Predict the Winner:** Develop a high-accuracy predictive model to forecast the winning team or player (`winPlacePerc`).
*   **Feature Analysis:** Analyze which in-game features have the most significant impact on winning outcomes.
*   **Model Comparison:** Implement and compare multiple machine learning models to select the most accurate and production-ready one.

## 📊 Dataset & Target Variable

*   **Source:** Large-scale competitive game match dataset (over **4.4 million rows**).
*   **Features:** Player statistics (`kills`, `damageDealt`, `assists`, `heals`, etc.) and Match metadata.
*   **Target Variable:** `winPlacePerc` (final placement percentage).

## 🛠️ Requirements & Setup

This project requires a Python 3.x environment with libraries including **pandas, numpy, scikit-learn, xgboost, and lightgbm**. All development was performed in **Jupyter Notebooks**.

## 🚀 Project Pipeline & Structure

The project was executed through a structured, multi-step process:

1.  **Data Loading & Cleaning:** Load dataset and handle missing/inconsistent values.
2.  **Exploratory Data Analysis (EDA):** Visualize key features and identify patterns.
3.  **Feature Engineering:** Create high-value features like `totalDistance`, `headshot_rate`, and `killsPerDistance`.
4.  **Model Building & Evaluation:** Train initial models and evaluate performance.
5.  **Hyperparameter Tuning:** Optimize models using **RandomizedSearchCV**.
6.  **Model Comparison & Recommendation:** Compare five models and select the best for production.

---

## 🚧 Project Report: Challenges Faced and Resolution Strategies

This section details the critical challenges encountered and the robust strategies applied.

### Section 1: Data Challenges and Resolution Strategies

*   **Challenge: Large Dataset Size & Memory Management**
    *   **Resolution:** Used **Data Type Optimization (Downcasting)** and **Downsampling** (200k–750k rows) for hyperparameter tuning to manage the 4.4M+ records efficiently.
*   **Challenge: Outliers and Data Consistency**
    *   **Resolution:** Applied **IQR filtering** to remove atypical matches and dropped rows with missing target values (`winPlacePerc`).
*   **Challenge: Feature Engineering for Predictive Power**
    *   **Resolution:** Created composite features like `totalDistance` and skill-indicators like `headshot_rate` and `killsPerDistance`.
*   **Challenge: Preprocessing Heterogeneous Features**
    *   **Resolution:** Implemented a **`ColumnTransformer` with Pipelines** to apply `StandardScaler` (numerical) and `OneHotEncoder` (categorical) consistently.

### Section 2: Model Training & Evaluation

*   **Challenge: Long Training Times for Complex Models**
    *   **Resolution:** Used **`RandomizedSearchCV`** over GridSearch and controlled model complexity (`max_depth`, `n_estimators`).
*   **Challenge: Ensuring Fair Model Comparison**
    *   **Resolution:** Ensured consistency by using **Unified Pipelines** and the **exact same `train_test_split`** across all models.

---

## 📈 Model Comparison & Recommendation

Five regression models were evaluated based on performance metrics (R², RMSE, MAE) and training efficiency.

### Detailed Performance Metrics Summary

*   **Best R² Score (0.9334) & Lowest RMSE (0.0791):** **XGBoost Regressor**
*   **Fastest Training Time (322.18s):** LightGBM Regressor
*   **Models Compared:** XGBoost Regressor, Random Forest Regressor, LightGBM Regressor, Linear Regression, and Ridge Regression.

### 🌟 Final Recommendation

**The XGBoost Regressor is recommended for production deployment.**

**Justification:**
XGBoost delivered the highest predictive accuracy (R²: **0.9334**, RMSE: **0.0791**). This superior performance was deemed critical to the task, justifying the moderate training time (571.20s) over the faster, but slightly less accurate, LightGBM.

### Key Performance Snapshot:
*   **XGBoost:** R²: 0.9334 | RMSE: 0.0791 | Training Time: 571.20s
*   **Random Forest:** R²: 0.9331 | RMSE: 0.0793 | Training Time: 4975.40s
*   **Linear/Ridge:** R²: 0.8521 | RMSE: 0.1179 | Training Time: ~10s

## 🔑 Key Insights for Winning Strategy

The feature importance analysis provided actionable insights into factors that most influence the final game outcome:

1.  **Survival is King:** Features related to survival time and resource management (`healsAndBoosts`, `walkDistance`) were consistently among the most important predictors.
2.  **Efficient Aggression:** Metrics like **`killsPerDistance`** (capturing combat efficiency) and `headshot_rate` were more impactful than raw `kills` alone.
3.  **Late-Game Movement:** Aggregated movement metrics (`totalDistance`) proved critical, highlighting the importance of strategic positioning.
