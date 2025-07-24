models = [
  {
    "Model": "Multiple Linear Regression (MSE loss)",
    "Approach": "multivariate regression + 5-Fold cv + one-hot encoding",
    "MAE": 7.5616,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)


# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize linear regression model
model = LinearRegression()

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores across folds
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

# Convert to positive MAE values
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: Multiple Linear Regression (MSE loss)")
print("Approach: multivariate regression + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")
"""
  },
  {
    "Model": "Multiple Linear Regression (MSE loss + High VIF columns dropped)",
    "Approach": "Multivariate regression + 5-Fold CV + one-hot encoding",
    "MAE": 7.624,
    "Code": ""
  },
  {
    "Model": "Quantile Regression (MAE loss)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.5496,
    "Code": ""
  },
  {
    "Model": "Quantile Regression (MAE loss High VIF columns dropped)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.7069,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2",
    "MAE": 28.8104,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2 + high VIF columns dropped",
    "MAE": 33.3695,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3",
    "MAE": 18.0693,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped",
    "MAE": 18.8222,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4",
    "MAE": 16.7219,
    "Code": ""
  },
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped",
    "MAE": 17.5248,
    "Code": ""
  },
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler",
    "MAE": 8.4182,
    "Code": ""
  },
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped",
    "MAE": 8.5306,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.0474,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}",
    "MAE": 7.8887,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}",
    "MAE": 7.9547,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}",
    "MAE": 7.8615,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 4, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 30}",
    "MAE": 7.8275,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 0.5, 'max_depth': 30}",
    "MAE": 7.8671,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 10}",
    "MAE": 7.812,
    "Code": ""
  },
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}",
    "MAE": 7.8142,
    "Code": ""
  },
  {
    "Model": "XGBoost Regressor",
    "Approach": "Full-feature regression + OneHotEncoding + 5-Fold CV",
    "MAE": 8.7712,
    "Code": ""
  },
  {
    "Model": "XGBoost Regressor",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.8, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.8})",
    "MAE": 7.8803,
    "Code": ""
  },
  {
    "Model": "XGBoost Regressor(Tuned)",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.9, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.9})",
    "MAE": 7.9463,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.3525,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.9, 'regressor__num_leaves': 70, 'regressor__n_estimators': 100, 'regressor__max_depth': -1, 'regressor__learning_rate': 0.03, 'regressor__colsample_bytree': 1.0})",
    "MAE": 7.9638,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.8, 'regressor__num_leaves': 70, 'regressor__n_estimators': 500, 'regressor__min_child_samples': 30, 'regressor__max_depth': 3, 'regressor__learning_rate': 0.01, 'regressor__colsample_bytree': 0.7})",
    "MAE": 7.8892,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with BayesSearchCV (params: OrderedDict({'regressor__colsample_bytree': 1.0, 'regressor__learning_rate': 0.012614141235943423, 'regressor__max_depth': 3, 'regressor__min_child_samples': 44, 'regressor__n_estimators': 540, 'regressor__num_leaves': 20, 'regressor__reg_alpha': 0.0, 'regressor__reg_lambda': 0.22975045403226968, 'regressor__subsample': 1.0}))",
    "MAE": 7.8728,
    "Code": ""
  },
  {
    "Model": "Ridge Regression",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization",
    "MAE": 7.5447,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Tuned alpha=79.0604 using GridSearchCV",
    "MAE": 7.4468,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization alpha: 100.0",
    "MAE": 7.448,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection and polynomial features with 5-Fold CV Alpha: 10.0 Number of Features: 10",
    "MAE": 7.7575,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Tuned regression with 5-Fold CV Best Alpha: 112.8838",
    "MAE": 7.4505,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4408,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + 5-Fold CV and L1+L2 Regularization",
    "MAE": 7.4618,
    "Code": ""
  },
  {
    "Model": "Lasso Regression (tuned)",
    "Approach": "Tuned regression + 5-Fold CV Best Alpha: 0.07847599703514611",
    "MAE": 7.4711,
    "Code": ""
  },
  {
    "Model": "Lasso Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4746,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + Repeated 5-Fold CV + GridSearch on alpha + L1 ratio",
    "MAE": 7.4457,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha and L1 ratio selected via grid search",
    "MAE": 7.4457,
    "Code": ""
  },
  {
    "Model": "ExtraTrees Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.7777,
    "Code": ""
  },
  {
    "Model": "HistGradientBoosting Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.9384,
    "Code": ""
  },
  {
    "Model": "NGBoost Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 8.0045,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Ridge meta-model",
    "MAE": 7.5258,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + XGBoost + LightGBM)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + CatBoost meta-model",
    "MAE": "",
    "Code": ""
  },
  {
    "Model": "Stacking Regressor (Ridge + ElasticNet + XGBoost + LightGBM + CatBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Stacking",
    "MAE": 7.5193,
    "Code": ""
  },
  {
    "Model": "StackNet GradientBoosting Stacking (Layer 1 + Layer 2)",
    "Approach": "Layer 1: Ridge + Lasso + XGBoost; Layer 2: GradientBoostingRegressor; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 8.1737,
    "Code": ""
  },
  {
    "Model": "Voting Ensemble (Tree-Based Only)",
    "Approach": "RandomForest + ExtraTrees + LightGBM + XGBoost; Weighted Voting; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.9487,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "Diverse Feature Sets + OneHot + RobustScaler + Repeated 5-Fold CV + Linear meta-model",
    "MAE": 7.5496,
    "Code": ""
  },
  {
    "Model": "Bootstrap Aggregated XGBoost",
    "Approach": "10 Bootstrapped XGBoost Models + Averaged Predictions + OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.859,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Weighted Voting",
    "MAE": 7.4714,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.4702,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV",
    "MAE": 7.1206,
    "Code": ""
  },
  {
    "Model": "Stacking Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Stacking",
    "MAE": 7.4405,
    "Code": ""
  },
  {
    "Model": "Full Ensemble (Linear + Tree Models)",
    "Approach": "OneHot + RobustScaler + KFold CV + Stacking",
    "MAE": 7.5555,
    "Code": ""
  }
]
