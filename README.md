# 🏆 Game Winner Prediction Project: Leveraging ML to Forecast Competitive Match Outcomes

## 💡 Project Overview

This project focuses on building a robust predictive model to forecast the outcome of competitive games (specifically using a large **PUBG match dataset**), leveraging detailed player and match statistics. The primary goal is to move beyond simple statistics and develop machine learning models that can accurately predict the winning outcome (`winPlacePerc`).

The process involved extensive data wrangling, feature engineering, and a rigorous comparison of multiple machine learning models to identify the most accurate and production-ready solution.

## 🎯 Objectives

*   **Predict the Winner:** Develop a high-accuracy predictive model to forecast the winning team or player (`winPlacePerc`).
*   **Feature Analysis:** Analyze which in-game features (kills, damage dealt, survival time, etc.) have the most significant impact on winning outcomes.
*   **Model Comparison:** Implement and compare multiple machine learning models (e.g., Random Forest, XGBoost) to select the most accurate, robust, and efficient one for production deployment.

## 📊 Dataset & Target Variable

*   **Source:** Large-scale competitive game match dataset (over 4.4 million rows).
*   **Features:** Detailed match and player statistics, including:
    *   **Player Statistics:** `kills`, `damageDealt`, `assists`, `heals`, `boosts`, `walkDistance`, `rideDistance`, etc.
    *   **Match Metadata:** `matchId`, `groupId`, `gameMode`.
*   **Target Variable:**
    *   `winPlacePerc`: A continuous variable representing the final placement percentage (used for prediction).

## 🛠️ Requirements & Setup

This project requires a Python 3.x environment. All dependencies can typically be installed using `pip`.

```bash
# Recommended libraries for installation
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

The core development and analysis were performed in **Jupyter Notebooks**.

## 🚀 Project Pipeline & Structure

The project was executed through a structured, multi-step process, focusing on maximizing data quality and model performance.

| Task | Objective | Key Steps |
| :--- | :--- | :--- |
| **1. Data Analysis (EDA)** | Understand data structure and key relationships. | Load, inspect, handle missing/inconsistent values, visualize feature distributions (`kills`, `damageDealt`, `win distribution`). |
| **2. Feature Engineering** | Create features with high predictive power. | Calculate `totalDistance`, `headshot_rate`, `killsPerDistance`, `healsAndBoosts`. |
| **3. Model Building** | Train an initial high-performance model. | Preprocess (scaling, encoding), split data, train initial models (e.g., Random Forest, XGBoost). |
| **4. Hyperparameter Tuning** | Optimize model parameters for best performance. | Utilize **RandomizedSearchCV** on downsampled data. |
| **5. Model Comparison** | Evaluate and compare multiple model architectures. | Train and evaluate **Random Forest, XGBoost, and LightGBM**. |
| **6. Final Recommendation** | Select the best model for production. | Justify selection based on accuracy, speed, and robustness. |

## 🚧 Project Report: Challenges Faced and Resolution Strategies

Working with a large-scale, real-world competitive gaming dataset presented unique challenges. The following section details the primary obstacles and the techniques applied to overcome them, ensuring a robust and reliable predictive solution.

### Section 1: Data Challenges and Resolution Strategies

| Challenge | Description | Techniques Applied & Justification |
| :--- | :--- | :--- |
| **1. Large Dataset Size & Memory** | The 4.4M+ row dataset caused significant memory exhaustion and extremely long runtimes. | **Data Type Optimization (Downcasting):** Converted `float64`/`int64` to smaller types (`float32`, `int16`) to drastically reduce memory footprint. **Downsampling:** Used 200k–750k rows for initial analysis and hyperparameter tuning to make the process feasible. |
| **2. Outliers and Data Consistency** | Presence of "abnormal" matches (e.g., very few players) and missing target values. | **Outlier Match Removal:** Applied IQR filtering on player count features (`numPlayers`) to drop atypical matches that would skew model learning. **Target Handling:** Dropped rows with missing `winPlacePerc` to ensure models only train on valid data. |
| **3. Feature Engineering for Power** | Raw features lacked direct predictive correlation with the target. | **Combined Distance:** Created `totalDistance = walkDistance + rideDistance + swimDistance` for a holistic movement metric. **Skill Indicators:** Engineered `headshot_rate` and `killsPerDistance` to capture player efficiency and skill. |
| **4. Preprocessing Heterogeneous Features** | Handling the mix of numerical (need scaling) and categorical (need encoding) features. | **`ColumnTransformer` with Pipelines:** Applied `StandardScaler` to numerical features and `OneHotEncoder(handle_unknown="ignore")` to categorical features. This ensures scaling prevents feature domination and encoding supports ML models without crashing on unseen categories in production. |

### Section 2: Model Training & Evaluation Challenges

| Challenge | Description | Techniques Applied & Justification |
| :--- | :--- | :--- |
| **1. Long Training Times** | Complex ensemble models (RF, XGBoost) are inherently slow on millions of data points. | **`RandomizedSearchCV`:** Used for hyperparameter tuning to efficiently explore the search space, replacing the slower `GridSearchCV`. **Controlled Model Complexity:** Fixed `max_depth` and capped `n_estimators` to prevent overly deep, slow, and memory-heavy tree constructions. |
| **2. Ensuring Fair Model Comparison** | Need to guarantee models are compared on identical ground. | **Unified Pipelines:** Each model was wrapped in an identical preprocessor + regressor pipeline for consistent data handling. **Consistent Split:** The exact same `train_test_split` was reused for every model evaluation. |
| **3. Selecting the "Best" Model** | The "best" model is not always the one with the highest R² but the one best suited for production. | **Multi-Criteria Evaluation:** Compared models based on R², MAE, **training time**, **prediction speed**, and complexity. **Recommendation Rationale:** While XGBoost and Random Forest performed well, **LightGBM** was ultimately preferred due to its superior balance of high accuracy and low training/inference latency, making it ideal for scalable production deployment. |

## 📈 Model Comparison & Recommendation

Multiple regression models were trained and evaluated on standardized metrics (R², RMSE, MAE). The final recommendation prioritized a balance of predictive accuracy and operational efficiency.

| Model | R² Score (Example) | RMSE (Example) | Key Advantage | Production Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Low-Medium | High | Fast to train, highly interpretable. | Low. Too simplistic for complex data. |
| **Random Forest** | High | Low | Robust, captures non-linearity well. | Medium. Slow training and high memory use. |
| **XGBoost** | Very High | Very Low | State-of-the-art predictive performance. | High. Fast inference, but training is slower than LightGBM. |
| **LightGBM** | Very High | Very Low | **Highest speed-to-accuracy ratio.** | **Recommended.** Excellent speed, memory efficiency, and accuracy. |

### 🌟 Final Recommendation

**LightGBM (Light Gradient Boosting Machine)** is recommended for the production environment. Its use of Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) allows it to maintain superior predictive accuracy while offering significantly faster training times and lower memory consumption compared to XGBoost and Random Forest on large datasets.

## 🔑 Key Insights for Winning Strategy

The feature importance analysis (derived from the best models) provided actionable insights into factors that most influence the final game outcome:

1.  **Survival is King:** Features related to survival time and resource management (`healsAndBoosts`, `walkDistance`, `survivalTime`) were consistently among the most important predictors.
2.  **Efficient Aggression:** Metrics like **`killsPerDistance`** (capturing combat efficiency) and `headshot_rate` were more impactful than raw `kills` alone, indicating the *quality* of engagement matters more than the quantity.
3.  **Late-Game Movement:** Aggregated movement metrics (`totalDistance`) proved critical, highlighting the importance of strategic positioning throughout the match.
