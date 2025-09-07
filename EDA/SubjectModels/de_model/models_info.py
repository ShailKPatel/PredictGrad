models = [
  # Multiple Linear Regression (MSE loss)
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
  # Multiple Linear Regression (MSE loss + High VIF columns dropped)
  {
    "Model": "Multiple Linear Regression (MSE loss + High VIF columns dropped)",
    "Approach": "Multivariate regression + 5-Fold CV + one-hot encoding",
    "MAE": 7.624,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)
# Without dropping High VIF columns: MAE: 6.6619

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

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
print("Model: Multiple Linear Regression (MSE loss + High VIF columns dropped)")
print("Approach: Multivariate regression + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Quantile Regression (MAE loss)
  {
    "Model": "Quantile Regression (MAE loss)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.5496,
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

# Add intercept manually
X = sm.add_constant(X)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

# Fit Quantile Regression (MAE = q=0.5) on each fold
for train_index, test_index in kf.split(X):
    # Ensure input is float type to prevent dtype=object errors
    X_train = X.iloc[train_index].astype(float)
    X_test = X.iloc[test_index].astype(float)
    y_train = y.iloc[train_index].astype(float)
    y_test = y.iloc[test_index].astype(float)

    # Fit Quantile Regression model (q=0.5 corresponds to MAE minimization)
    model = sm.QuantReg(y_train, X_train)
    result = model.fit(q=0.5)

    # Predict and calculate fold MAE
    preds = result.predict(X_test)
    fold_mae = np.mean(np.abs(y_test - preds))
    mae_scores.append(fold_mae)

mean_mae = np.mean(mae_scores)

# Print and log
print("Model: Quantile Regression (MAE loss)")
print("Approach: q=0.5 + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Quantile Regression (MAE loss High VIF columns dropped)
  {
    "Model": "Quantile Regression (MAE loss High VIF columns dropped)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.7069,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")
# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Add intercept manually
X = sm.add_constant(X)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

# Fit Quantile Regression (MAE = q=0.5) on each fold
for train_index, test_index in kf.split(X):
    # Ensure input is float type to prevent dtype=object errors
    X_train = X.iloc[train_index].astype(float)
    X_test = X.iloc[test_index].astype(float)
    y_train = y.iloc[train_index].astype(float)
    y_test = y.iloc[test_index].astype(float)

    # Fit Quantile Regression model (q=0.5 corresponds to MAE minimization)
    model = sm.QuantReg(y_train, X_train)
    result = model.fit(q=0.5)

    # Predict and calculate fold MAE
    preds = result.predict(X_test)
    fold_mae = np.mean(np.abs(y_test - preds))
    mae_scores.append(fold_mae)

mean_mae = np.mean(mae_scores)

# Print and log
print("Model: Quantile Regression (MAE loss, High VIF columns dropped)")
print("Approach: q=0.5 + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 2)
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2",
    "MAE": 28.8104,
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

# Initialize polynomial regression (order 2)
polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 2)")
print(
    "Approach: Full-feature polynomial regression (degree 2) with 5-Fold CV and one-hot encoding"
)
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 2) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2 + high VIF columns dropped",
    "MAE": 33.3695,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 2)
polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 2)")
print("Approach: 5-Fold CV + one-hot encoding + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 3)
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3",
    "MAE": 18.0693,
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

# Initialize polynomial regression (order 3)
polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 3)")
print("Approach: Full-feature polynomial regression + 5-Fold CV + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 3) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped",
    "MAE": 18.8222,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")


# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 3)
polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 3)")
print("Approach: 5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 3)",
            "Approach": "5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Polynomial Regression (Order 4) 
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4",
    "MAE": 16.7219,
    "Code": """# One-hot encode categorical columns and drop the first column of each
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

# Initialize polynomial regression (order 4)
polyreg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 4)")
print("Approach: 5-Fold CV + one-hot encoding + degree 4")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 4)",
            "Approach": "5-Fold CV + one-hot encoding + degree 4",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Polynomial Regression (Order 4) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped",
    "MAE": 17.5248,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 4)
polyreg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 4)")
print("Approach: 5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 4)",
            "Approach": "5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Support Vector Regression (RBF)
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler",
    "MAE": 8.4182,
    "Code": """# One-hot encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)


# Define features and target
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Build pipeline: Standardize -> SVR
svr_pipeline = make_pipeline(
    StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
)

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    svr_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Support Vector Regression (RBF)")
print("Approach: 5-Fold CV + one-hot encoding + StandardScaler + RBF")
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Support Vector Regression (RBF)",
            "Approach": "5-Fold CV + one-hot encoding + StandardScaler",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Support Vector Regression (RBF) (high VIF columns dropped)
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped",
    "MAE": 8.5306,
    "Code": """# One-hot encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Drop high-VIF columns
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define features and target
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Build pipeline: Standardize -> SVR
svr_pipeline = make_pipeline(
    StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
)

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    svr_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Support Vector Regression (RBF)")
print(
    "Approach: 5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Support Vector Regression (RBF)",
            "Approach": "5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Random Forest Regressor
  {
    "Model": "Random Forest Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.0474,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Didn't drop columns with high internal correlation because tree based structures handel them well

# Define target and feature columns
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Define model pipeline (no preprocessor needed since categorical columns are already encoded)
model = Pipeline(
    steps=[
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Use 5-Fold CV with negative MAE
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Suppress specific sklearn warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    neg_mae_scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )

mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: Random Forest Regressor")
print("Approach: Full-feature regression with 5-Fold CV and OneHotEncoding")
print(f"MAE: {mean_mae:.4f}")

# Log results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor",
            "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}",
    "MAE": 7.8887,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                max_depth=20,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, "
    "'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, "
            "'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}",
    "MAE": 7.9547,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline with updated hyperparameters
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features=None,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, "
    "'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, "
            "'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}",
    "MAE": 7.8615,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline with the specified tuned hyperparameters
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, "
    "'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, "
            "'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 4, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 30}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 4, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 30}",
    "MAE": 7.8275,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]


# Preprocessing pipeline for encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Define parameter grid for RandomizedSearchCV
param_distributions = {
    "regressor__n_estimators": [100, 200, 500, 1000],
    "regressor__max_depth": [10, 20, 30, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": ["sqrt", "log2", None],  # Removed 'auto'
}


# Use 5-Fold CV for tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    scoring="neg_mean_absolute_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
)

# Fit the RandomizedSearchCV to the data
random_search.fit(X, y)

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_  # Convert back from neg MAE to MAE

# Print results to terminal
print("Model: Random Forest Regressor(tuned)")
print("Parameters:", best_params)
print(f"MAE: {best_score:.4f}")

# Log results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": best_params,
            "MAE": round(best_score, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 0.5, 'max_depth': 30}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 0.5, 'max_depth': 30}",
    "MAE": 7.8671,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define the tuned Random Forest model
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=30,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, "
    "'max_features': 0.5, 'max_depth': 30}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, "
            "'max_features': 0.5, 'max_depth': 30}",
            "MAE": round(mean_mae, 4),
        }
    ]
)
log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 10}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 10}",
    "MAE": 7.812,
    "Code": """# Handle outliers in target using IQR
target_col = "DE Theory"
Q1, Q3 = df[target_col].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[~((df[target_col] < Q1 - 1.5 * IQR) | (df[target_col] > Q3 + 1.5 * IQR))]

# Define target and features
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Full pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Hyperparameter tuning setup
param_distributions = {
    "regressor__n_estimators": [100, 200, 500],
    "regressor__max_depth": [10, 20, 30, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 3],
    "regressor__max_features": ["sqrt", 0.5, None],
}

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
)

# Fit model
random_search.fit(X, y)

# Best model and results
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

# Feature importances
feature_importances = best_model.named_steps["regressor"].feature_importances_
feature_names = (
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
    + numeric_cols
)
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
)
print(
    "Feature Importances:\n",
    importance_df.sort_values(by="Importance", ascending=False),
)

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": best_params,
            "MAE": round(best_score, 4),
        }
    ]
)
log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)

# Output summary
print("Model: Random Forest Regressor (Tuned)")
print("Approach:", best_params)
print(f"MAE: {best_score:.4f}")"""
  },
  # Random Forest Regressor (Tuned) {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}",
    "MAE": 7.8142,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]


# Custom Winsorizer
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        # Must specify axis=1 since our lower/upper bounds have columns as the index
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


# Features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]

# Use top 25 features importance list
top_features = [
    "Sem 2 Percentage",
    "Math-2 Theory",
    "Data Structures using Java Theory",
    "Fundamental of Electronics and Electrical Theory",
    "Sem 1 Percentage",
    "Physics Theory",
    "DBMS Theory",
    "Math-1 Theory",
    "Software Engineering Theory",
    "DBMS Practical",
    "Java-1 Theory",
    "Java-2 Theory",
    "Fundamental of Electronics and Electrical Practical",
    "Environmental Science Theory",
    "Data Structures using Java Practical",
    "Java-2 Attendance",
    "Data Structures using Java Attendance",
    "Software Engineering Attendance",
    "Fundamental of Electronics and Electrical Attendance",
    "Roll-1",
    "Java-1 Attendance",
    "Math-1 Attendance",
    "Math-2 Attendance",
    "Computer Workshop Practical",
    "Physics Attendance",
]

# Add relevant categorical columns (one-hot encoding will handle dummy drop)
top_features += categorical_cols
X = X[top_features]

# Identify numeric columns (excluding categoricals)
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline([("winsor", Winsorizer()), ("scaler", StandardScaler())]),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# Tuned Random Forest Model
model = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.3,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Cross-Validation MAE
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = -1 * cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}"
)
print(f"Mean MAE: {mae_scores.mean():.4f}")

# --- Log Results to CSV ---
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}",
            "MAE": round(mae_scores.mean(), 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # XGBoost Regressor
  {
    "Model": "XGBoost Regressor",
    "Approach": "Full-feature regression + OneHotEncoding + 5-Fold CV",
    "MAE": 8.7712,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define the XGBoost pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42, verbosity=0)),
    ]
)

# 5-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: XGBoost Regressor")
print("Approach: Full-feature regression + OneHotEncoding + 5-Fold CV")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # XGBoost Regressor (Tuned) (Best Params: {'regressor__colsample_bytree': 0.8, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.8})
  {
    "Model": "XGBoost Regressor",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.8, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.8})",
    "MAE": 7.8803,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define the XGBoost pipeline
xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42, verbosity=0)),
    ]
)

# Define parameter grid
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__colsample_bytree": [0.8, 0.9, 1.0],
    "regressor__subsample": [0.8, 0.9, 1.0],
}

# Set up GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=kf,
    verbose=1,
    n_jobs=-1,
)

# Fit the GridSearchCV
grid_search.fit(X, y)

# Extract best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model with cross-validation
neg_mae_scores = cross_val_score(
    best_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: XGBoost Regressor (Tuned)")
print(f"Best Params: {best_params}")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # XGBoost Regressor (Tuned) (Best Params: {'regressor__colsample_bytree': 0.9, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.9})
  {
    "Model": "XGBoost Regressor(Tuned)",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.9, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.9})",
    "MAE": 7.9463,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Tuned XGBoost Regressor with best params
xgb_best = XGBRegressor(
    colsample_bytree=0.9,
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    subsample=0.9,
    random_state=42,
    verbosity=0,
)

# Final pipeline
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", xgb_best)])

# 5-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: XGBoost Regressor(Tuned)")
print(
    "Approach: Tuned (Best Params: {'regressor__colsample_bytree': 0.9, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.9})"
)
print(f"MAE: {mean_mae:.4f}")"""
  },
  # LightGBM Regressor 
  {
    "Model": "LightGBM Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.3525,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline for encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define LightGBM model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
    ]
)

# Use 5-Fold CV with negative MAE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: LightGBM Regressor")
print("Approach: Full-feature regression with 5-Fold CV and OneHotEncoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # LightGBM Regressor (Tuned) (params: {'regressor__subsample': 0.9, 'regressor__num_leaves': 70, 'regressor__n_estimators': 100, 'regressor__max_depth': -1, 'regressor__learning_rate': 0.03, 'regressor__colsample_bytree': 1.0})
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.9, 'regressor__num_leaves': 70, 'regressor__n_estimators': 100, 'regressor__max_depth': -1, 'regressor__learning_rate': 0.03, 'regressor__colsample_bytree': 1.0})",
    "MAE": 7.9638,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define base pipeline
base_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
    ]
)

# Define parameter grid
param_distributions = {
    "regressor__num_leaves": [20, 31, 50, 70],
    "regressor__max_depth": [3, 5, 7, 9, -1],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

# Setup 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Run randomized search
random_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# Fit search
random_search.fit(X, y)
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

# Print best results
print("Model: LightGBM Regressor (Tuned)")
print("Best Params:", best_params)
print(f"Best MAE: {best_score:.4f}")
"""
  },
  # LightGBM Regressor (Tuned) (params: {'regressor__subsample': 0.8, 'regressor__num_leaves': 70, 'regressor__n_estimators': 500, 'regressor__min_child_samples': 30, 'regressor__max_depth': 3, 'regressor__learning_rate': 0.01, 'regressor__colsample_bytree': 0.7})
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.8, 'regressor__num_leaves': 70, 'regressor__n_estimators': 500, 'regressor__min_child_samples': 30, 'regressor__max_depth': 3, 'regressor__learning_rate': 0.01, 'regressor__colsample_bytree': 0.7})",
    "MAE": 7.8892,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define base pipeline
base_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
    ]
)

# Define optimized parameter grid
param_distributions = {
    "regressor__num_leaves": [20, 31, 50, 70],
    "regressor__max_depth": [3, 5, 7, 9, -1],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "regressor__min_child_samples": [10, 20, 30],
}

# Setup 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Run randomized search
random_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions=param_distributions,
    n_iter=40,  # Slightly more than before
    scoring="neg_mean_absolute_error",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# Fit search
random_search.fit(X, y)
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

# Print best results
print("Model: LightGBM Regressor (Tuned)")
print("Best Params:", best_params)
print(f"Best MAE: {best_score:.4f}")# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify column types
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define base pipeline
base_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
    ]
)

# Define optimized parameter grid
param_distributions = {
    "regressor__num_leaves": [20, 31, 50, 70],
    "regressor__max_depth": [3, 5, 7, 9, -1],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "regressor__min_child_samples": [10, 20, 30],
}

# Setup 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Run randomized search
random_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions=param_distributions,
    n_iter=40,  # Slightly more than before
    scoring="neg_mean_absolute_error",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# Fit search
random_search.fit(X, y)
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

# Print best results
print("Model: LightGBM Regressor (Tuned)")
print("Best Params:", best_params)
print(f"Best MAE: {best_score:.4f}")"""
  },
  # LightGBM Regressor (Tuned) (params: OrderedDict({'regressor__colsample_bytree': 1.0, 'regressor__learning_rate': 0.012614141235943423, 'regressor__max_depth': 3, 'regressor__min_child_samples': 44, 'regressor__n_estimators': 540, 'regressor__num_leaves': 20, 'regressor__reg_alpha': 0.0, 'regressor__reg_lambda': 0.22975045403226968, 'regressor__subsample': 1.0}))
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with BayesSearchCV (params: OrderedDict({'regressor__colsample_bytree': 1.0, 'regressor__learning_rate': 0.012614141235943423, 'regressor__max_depth': 3, 'regressor__min_child_samples': 44, 'regressor__n_estimators': 540, 'regressor__num_leaves': 20, 'regressor__reg_alpha': 0.0, 'regressor__reg_lambda': 0.22975045403226968, 'regressor__subsample': 1.0}))",
    "MAE": 7.8728,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Remove outliers using IQR
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[mask], y[mask]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define base pipeline
base_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
    ]
)

# Define Bayesian parameter space
param_distributions = {
    "regressor__num_leaves": Integer(20, 100),
    "regressor__max_depth": Integer(3, 9),
    "regressor__learning_rate": Real(0.005, 0.05, prior="log-uniform"),
    "regressor__n_estimators": Integer(100, 600),
    "regressor__subsample": Real(0.7, 1.0),
    "regressor__colsample_bytree": Real(0.7, 1.0),
    "regressor__reg_alpha": Real(0.0, 0.3),
    "regressor__reg_lambda": Real(0.0, 0.3),
    "regressor__min_child_samples": Integer(10, 50),
}

# Setup 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Run Bayesian search
search = BayesSearchCV(
    base_pipeline,
    search_spaces=param_distributions,
    n_iter=60,
    scoring="neg_mean_absolute_error",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# Fit search
search.fit(X, y)
best_model = search.best_estimator_
best_params = search.best_params_
best_score = -search.best_score_

# Print best results
print("Model: LightGBM Regressor (Tuned)")
print("Best Params:", best_params)
print(f"Best MAE: {best_score:.4f}")"""
  },
  # Ridge Regression 
  {
    "Model": "Ridge Regression",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization",
    "MAE": 7.5447,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline for categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),  # Standardize numeric features
    ]
)

# Initialize Ridge regressor
ridge_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),  # Alpha is the regularization strength
    ]
)

# Use 5-Fold CV with negative MAE for Ridge
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge_neg_mae_scores = cross_val_score(
    ridge_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ridge_mae_scores = -ridge_neg_mae_scores
ridge_mean_mae = np.mean(ridge_mae_scores)

# Print results to terminal
print("Model: Ridge Regression")
print("Approach: Full-feature regression with 5-Fold CV and Regularization")
print(f"MAE: {ridge_mean_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (alpha=79.0604 using GridSearchCV)
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Tuned alpha=79.0604 using GridSearchCV",
    "MAE": 7.4468,
    "Code": """# Define pipeline again
ridge_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", Ridge())]  # No alpha yet
)

# Alpha values to test
param_grid = {"regressor__alpha": np.logspace(-3, 3, 50)}

# Grid search with 5-fold CV on negative MAE
grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_["regressor__alpha"]
best_mae = -grid_search.best_score_

# Print best results
print("Model: Ridge Regression (Tuned)")
print("Approach: Alpha tuning with GridSearchCV and 5-Fold CV")
print(f"Best Alpha: {best_alpha}")
print(f"MAE: {best_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (Full-feature regression with 5-Fold CV and Regularization alpha: 100.0)
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization alpha: 100.0",
    "MAE": 7.448,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline for categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),  # Standardize numeric features
    ]
)

# Initialize Ridge regressor pipeline
ridge_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", Ridge())]
)

# Define parameter grid for alpha tuning
param_grid = {"regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

# Perform GridSearchCV to find the best alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    ridge_pipeline, param_grid, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
)
grid_search.fit(X, y)

# Get the best model and its MAE
best_ridge_model = grid_search.best_estimator_
ridge_neg_mae_scores = cross_val_score(
    best_ridge_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ridge_mae_scores = -ridge_neg_mae_scores
ridge_mean_mae = np.mean(ridge_mae_scores)
best_alpha = grid_search.best_params_["regressor__alpha"]

# Print results to terminal
print("Model: Ridge Regression (Tuned)")
print("Approach: Full-feature regression with 5-Fold CV and Regularization")
print(f"Best Alpha: {best_alpha}")
print(f"MAE: {ridge_mean_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12)
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Feature selection: Select top 10 features based on correlation with target
selector = SelectKBest(score_func=f_regression, k=10)

# Create polynomial features for key numeric columns
poly_cols = ["Sem 1 Percentage", "Sem 2 Percentage"]
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)

# Preprocessing pipeline for categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
        (
            "poly",
            Pipeline(
                [
                    (
                        "selector",
                        ColumnTransformer(
                            [("select", "passthrough", poly_cols)], remainder="drop"
                        ),
                    ),
                    ("poly", poly_transformer),
                ]
            ),
            poly_cols,
        ),
    ]
)

# Initialize Ridge regressor pipeline with feature selection
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("regressor", Ridge()),
    ]
)

# Define parameter grid for alpha tuning
param_grid = {
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "selector__k": [5, 8, 10, 12],  # Tune number of features
}

# Perform GridSearchCV to find the best alpha and number of features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    ridge_pipeline, param_grid, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
)
grid_search.fit(X, y)

# Get the best model and its MAE
best_ridge_model = grid_search.best_estimator_
ridge_neg_mae_scores = cross_val_score(
    best_ridge_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ridge_mae_scores = -ridge_neg_mae_scores
ridge_mean_mae = np.mean(ridge_mae_scores)
best_alpha = grid_search.best_params_["regressor__alpha"]
best_k = grid_search.best_params_["selector__k"]

# Print results to terminal
print("Model: Ridge Regression (Tuned)")
print("Approach: Feature selection and polynomial features with 5-Fold CV")
print(f"Best Alpha: {best_alpha}")
print(f"Best Number of Features: {best_k}")
print(f"MAE: {ridge_mean_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (Feature selection and polynomial features with 5-Fold CV Alpha: 10.0 Number of Features: 10)
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection and polynomial features with 5-Fold CV Alpha: 10.0 Number of Features: 10",
    "MAE": 7.7575,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]
poly_cols = ["Sem 1 Percentage", "Sem 2 Percentage"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        (
            "num",
            StandardScaler(),
            [col for col in numeric_cols if col not in poly_cols],
        ),
        ("poly", PolynomialFeatures(degree=2, include_bias=False), poly_cols),
    ]
)

# Ridge regression pipeline with feature selection
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_regression, k=10)),
        ("regressor", Ridge(alpha=10.0)),
    ]
)

# Cross-validation and scoring
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge_neg_mae_scores = cross_val_score(
    ridge_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ridge_mae_scores = -ridge_neg_mae_scores
ridge_mean_mae = np.mean(ridge_mae_scores)

# Print results to terminal
print("Model: Ridge Regression (Tuned)")
print("Approach: Feature selection and polynomial features with 5-Fold CV")
print("Best Alpha: 10.0")
print("Best Number of Features: 10")
print(f"MAE: {ridge_mean_mae:.4f}")"""
  },
  # Ridge Regression (tuned) (Tuned regression with 5-Fold CV Best Alpha: 112.8838)
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Tuned regression with 5-Fold CV Best Alpha: 112.8838",
    "MAE": 7.4505,
    "Code": """# Define the target variable (what we aim to predict) and features (what we use for prediction)
target_col = "DE Theory"
X = df.drop(columns=[target_col])  # All columns except the target are features
y = df[target_col]  # The target column itself

# Identify categorical and numeric columns for appropriate preprocessing
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Create a preprocessing pipeline using ColumnTransformer
# 'OneHotEncoder' is used for categorical features to convert them into a numerical format,
# dropping the first category to prevent multicollinearity.
# 'StandardScaler' is used for numerical features to normalize them, often beneficial for linear models.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Define the Ridge regression model pipeline
# This pipeline first applies the defined preprocessing steps and then fits the Ridge regressor.
ridge_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            Ridge(),
        ),  # Ridge model with default parameters, which GridSearchCV will tune
    ]
)

# Define the range of alpha values (regularization strength) to search over for Ridge
# np.logspace(-3, 3, 20) creates 20 evenly spaced values on a log scale from 10^-3 to 10^3.
alpha_grid = {"regressor__alpha": np.logspace(-3, 3, 20)}

# Set up 5-Fold Cross-Validation strategy for GridSearchCV
# 5 splits, shuffles the data, and sets a fixed random_state for reproducibility.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV for Ridge regression
# 'scoring='neg_mean_absolute_error'' is used for hyperparameter tuning. GridSearchCV will find
# the alpha that minimizes the negative MAE (which means it maximizes MAE, effectively minimizing MAE).
# 'refit=True' ensures that the best estimator found by GridSearchCV is refitted on the entire dataset.
ridge_search = GridSearchCV(
    ridge_pipeline, alpha_grid, cv=kf, scoring="neg_mean_absolute_error", refit=True
)
ridge_search.fit(X, y)  # Fit GridSearchCV to find the best Ridge model
ridge_best_mae = (
    -ridge_search.best_score_
)  # Convert the best negative MAE score back to positive

# ---
## Ridge Regression Results
# Print the evaluation results for the best Ridge Regression model found by GridSearchCV
print("Model: Ridge Regression")
print("Approach: Hyperparameter-tuned full-feature regression with 5-Fold CV")
print(
    f"Best Alpha: {ridge_search.best_params_['regressor__alpha']:.4f}"
)  # Print the best alpha found
print(f"Mean Absolute Error (MAE): {ridge_best_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (Tuned regression + 5-Fold CV Best Alpha: 0.01 + Number of Features: 12)
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Feature selection: Select top 10 features based on correlation with target
selector = SelectKBest(score_func=f_regression, k=10)

# Create polynomial features for key numeric columns
poly_cols = ["Sem 1 Percentage", "Sem 2 Percentage"]
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)

# Preprocessing pipeline for categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
        (
            "poly",
            Pipeline(
                [
                    (
                        "selector",
                        ColumnTransformer(
                            [("select", "passthrough", poly_cols)], remainder="drop"
                        ),
                    ),
                    ("poly", poly_transformer),
                ]
            ),
            poly_cols,
        ),
    ]
)

# Initialize Ridge regressor pipeline with feature selection
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("regressor", Ridge()),
    ]
)

# Define parameter grid for alpha tuning
param_grid = {
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "selector__k": [5, 8, 10, 12],  # Tune number of features
}

# Perform GridSearchCV to find the best alpha and number of features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    ridge_pipeline, param_grid, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
)
grid_search.fit(X, y)

# Get the best model and its MAE
best_ridge_model = grid_search.best_estimator_
ridge_neg_mae_scores = cross_val_score(
    best_ridge_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ridge_mae_scores = -ridge_neg_mae_scores
ridge_mean_mae = np.mean(ridge_mae_scores)
best_alpha = grid_search.best_params_["regressor__alpha"]
best_k = grid_search.best_params_["selector__k"]

# Print results to terminal
print("Model: Ridge Regression (Tuned)")
print("Approach: Feature selection and polynomial features with 5-Fold CV")
print(f"Best Alpha: {best_alpha}")
print(f"Best Number of Features: {best_k}")
print(f"MAE: {ridge_mean_mae:.4f}")"""
  },
  # Ridge Regression (Tuned) (Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search)
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4408,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Ridge pipeline
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Broad alpha grid search
alpha_grid_broad = {"regressor__alpha": np.logspace(-3, 3, 20)}
ridge_search_broad = GridSearchCV(
    ridge_pipeline,
    alpha_grid_broad,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
ridge_search_broad.fit(X, y)
ridge_best_alpha_broad = ridge_search_broad.best_params_["regressor__alpha"]

# Refined alpha grid search
ridge_alpha_refined_grid = {
    "regressor__alpha": np.linspace(
        max(1e-5, ridge_best_alpha_broad * 0.5), ridge_best_alpha_broad * 1.5, 30
    )
}
ridge_search_refined = GridSearchCV(
    ridge_pipeline,
    ridge_alpha_refined_grid,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
ridge_search_refined.fit(X, y)
ridge_best_mae = -ridge_search_refined.best_score_

# Print Results
print("Model: Ridge Regression (tuned)")
print(
    "Approach: Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search"
)
print(f"MAE: {ridge_best_mae:.4f}")"""
  },
  # ElasticNet Regression
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + 5-Fold CV and L1+L2 Regularization",
    "MAE": 7.4618,
    "Code": """target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Changed to ElasticNet
elastic_net_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        ),  # l1_ratio=0.5 balances L1 and L2
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

elasticnet_neg_mae_scores = cross_val_score(
    elastic_net_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
elasticnet_mae_scores = -elasticnet_neg_mae_scores
elasticnet_mean_mae = np.mean(elasticnet_mae_scores)

print("Model: ElasticNet Regression")
print("Approach: Full-feature regression + 5-Fold CV and L1+L2 Regularization")
print(f"Mean Absolute Error (MAE): {elasticnet_mean_mae:.4f}")"""
  },
  # ElasticNet Regression (tuned) (Tuned regression + 5-Fold CV Best Alpha: 0.07847599703514611)
  {
    "Model": "ElasticNet Regression (tuned)",
    "Approach": "Tuned regression + 5-Fold CV Best Alpha: 0.07847599703514611",
    "MAE": 7.4711,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Lasso pipeline
lasso_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", Lasso(max_iter=10000)),  # Ensure convergence
    ]
)

# Alpha values to try
alpha_grid = {"regressor__alpha": np.logspace(-3, 3, 20)}

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch for Lasso
lasso_search = GridSearchCV(
    lasso_pipeline, alpha_grid, cv=kf, scoring="neg_mean_absolute_error", refit=True
)
lasso_search.fit(X, y)
lasso_best_mae = -lasso_search.best_score_

# Print results
print("\nModel: Lasso Regression")
print("Approach: Hyperparameter-tuned full-feature regression with 5-Fold CV")
print(f"Best Alpha: {lasso_search.best_params_['regressor__alpha']}")
print(f"MAE: {lasso_best_mae:.4f}")"""
  },
  # Lasso Regression (tuned) (Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search)
  {
    "Model": "Lasso Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4746,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Lasso pipeline
lasso_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Broad alpha grid search
alpha_grid_broad = {"regressor__alpha": np.logspace(-3, 3, 20)}
lasso_search_broad = GridSearchCV(
    lasso_pipeline,
    alpha_grid_broad,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
lasso_search_broad.fit(X, y)
lasso_best_alpha_broad = lasso_search_broad.best_params_["regressor__alpha"]

# Refined alpha grid search
lasso_alpha_refined_grid = {
    "regressor__alpha": np.linspace(
        max(1e-5, lasso_best_alpha_broad * 0.5), lasso_best_alpha_broad * 1.5, 30
    )
}
lasso_search_refined = GridSearchCV(
    lasso_pipeline,
    lasso_alpha_refined_grid,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
lasso_search_refined.fit(X, y)
lasso_best_mae = -lasso_search_refined.best_score_

# Print Results
print("Model: Lasso Regression (tuned)")
print(
    "Approach: Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search"
)
print(f"MAE: {lasso_best_mae:.4f}")"""
  },
  # ElasticNet Regression (Full-feature regression + Repeated 5-Fold CV + GridSearch on alpha + L1 ratio)
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + Repeated 5-Fold CV + GridSearch on alpha + L1 ratio",
    "MAE": 7.4457,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# ElasticNet pipeline
elastic_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", ElasticNet(max_iter=10000))]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Hyperparameter tuning
elastic_param_grid = {
    "regressor__alpha": np.logspace(-3, 1, 15),
    "regressor__l1_ratio": np.linspace(0.1, 1.0, 10),
}
elastic_search = GridSearchCV(
    elastic_pipeline,
    elastic_param_grid,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
elastic_search.fit(X, y)
elastic_best_mae = -elastic_search.best_score_

# Print results
print("Model: ElasticNet Regression")
print(
    "Approach: Full-feature regression + Repeated 5-Fold CV + GridSearch on alpha + L1 ratio"
)
print(f"MAE: {elastic_best_mae:.4f}")"""
  },
  # ElasticNet Regression (tuned) (Full-feature regression with Repeated 5-Fold CV and Regularization, alpha and L1 ratio selected via grid search)
  {
    "Model": "ElasticNet Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha and L1 ratio selected via grid search",
    "MAE": 7.4457,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# ElasticNet pipeline
elastic_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", ElasticNet(max_iter=10000))]
)

# RepeatedKFold CV
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Hyperparameter grid
elastic_param_grid = {
    "regressor__alpha": np.logspace(-3, 1, 15),
    "regressor__l1_ratio": np.linspace(0.1, 1.0, 10),
}

# GridSearchCV
elastic_search = GridSearchCV(
    elastic_pipeline,
    elastic_param_grid,
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=-1,
)
elastic_search.fit(X, y)
elastic_best_mae = -elastic_search.best_score_

# Print results
print("\nModel: ElasticNet Regression (tuned)")
print(
    "Approach: Full-feature regression with Repeated 5-Fold CV and Regularization, alpha and L1 ratio selected via grid search"
)
print(f"MAE: {elastic_best_mae:.4f}")"""
  },
  # ExtraTrees Regressor (OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch)
  {
    "Model": "ExtraTrees Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.7777,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# ExtraTrees pipeline
et_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
    ]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Hyperparameter grid
et_param_grid = {
    "regressor__n_estimators": [200, 300, 400],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
}

# GridSearchCV
grid_search = GridSearchCV(
    et_pipeline,
    et_param_grid,
    scoring="neg_mean_absolute_error",
    cv=kf,
    refit=True,
    n_jobs=-1,
)

# Fit model
grid_search.fit(X, y)

# Evaluate
best_mae = -grid_search.best_score_
print("Model: ExtraTrees Regressor")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch")
print(f"MAE: {best_mae:.4f}")"""
  },
  # HistGradientBoosting Regressor (OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch)
  {
    "Model": "HistGradientBoosting Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.9384,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Pipeline
hgb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(random_state=42)),
    ]
)

# Cross-validation
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Hyperparameter grid
hgb_param_grid = {
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__max_iter": [200, 300, 500],
    "regressor__max_depth": [3, 5, 7],
    "regressor__l2_regularization": [0.0, 0.1, 1.0],
}

# Grid Search
grid_search = GridSearchCV(
    hgb_pipeline,
    param_grid=hgb_param_grid,
    scoring="neg_mean_absolute_error",
    cv=kf,
    n_jobs=-1,
    refit=True,
)

# Train
grid_search.fit(X, y)

# Evaluate
best_mae = -grid_search.best_score_
print("Model: HistGradientBoosting Regressor")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch")
print(f"MAE: {best_mae:.4f}")"""
  },
  # NGBoost Regressor (OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch)
  {
    "Model": "NGBoost Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 8.0045,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# NGBoost pipeline
ngb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", NGBRegressor(Dist=Normal, random_state=42, verbose=False)),
    ]
)

# Cross-validation
kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Hyperparameter grid
ngb_param_grid = {
    "regressor__n_estimators": [300, 500],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__minibatch_frac": [1.0],
    "regressor__col_sample": [0.8, 1.0],
}

# Grid Search
grid_search = GridSearchCV(
    estimator=ngb_pipeline,
    param_grid=ngb_param_grid,
    scoring="neg_mean_absolute_error",
    cv=kf,
    n_jobs=-1,
    refit=True,
)

# Fit model
grid_search.fit(X, y)

# Evaluate
best_mae = -grid_search.best_score_
print("Model: NGBoost Regressor")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch")
print(f"MAE: {best_mae:.4f}")"""
  },
  # Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Ridge meta-model",
    "MAE": 7.5258,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocessing
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Define base models (you can plug in best hyperparams if you have them)
ridge = Ridge(alpha=10.0)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.3,
    random_state=42,
)
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

# Pipeline wrappers
ridge_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", ridge)])
elastic_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", elastic)])
rf_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", rf)])
xgb_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", xgb)])

# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ("ridge", ridge_pipe),
        ("elastic", elastic_pipe),
        ("rf", rf_pipe),
        ("xgb", xgb_pipe),
    ],
    final_estimator=LinearRegression(),  
    cv=5,
    n_jobs=-1,
    passthrough=False,
)

# Evaluate
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = -cross_val_score(
    stack, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

# Print Results
print("Model: Stacked Ensemble (Ridge + ElasticNet + RF + XGBoost)")
print("Approach: Meta-learning with 5-Fold CV, Linear Regression as meta-model")
print(f"MAE: {mae_scores.mean():.4f}")"""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + XGBoost + LightGBM)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + CatBoost meta-model",
    "MAE": "",
    "Code": ""
  },
  # Stacked Regressor (Ridge + ElasticNet + XGBoost + LightGBM + CatBoost)
  {
    "Model": "Stacking Regressor (Ridge + ElasticNet + XGBoost + LightGBM + CatBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Stacking",
    "MAE": 7.5193,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocessing
categorical_cols = ["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
     ("num", RobustScaler(), numeric_cols)]
)

# Base models
ridge = Ridge(alpha=10.0, random_state=42)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
xgb = XGBRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.9,
    colsample_bytree=0.9, random_state=42
)
lgb = LGBMRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.9,
    colsample_bytree=0.9, random_state=42
)
cat = CatBoostRegressor(
    verbose=0, learning_rate=0.05, depth=4, iterations=500, random_seed=42
)

# Meta-model using StackingRegressor
stack = StackingRegressor(
    estimators=[
        ("ridge", ridge), ("elastic", elastic),
        ("xgb", xgb), ("lgb", lgb), ("cat", cat)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

# Pipeline with preprocessing
pipeline = Pipeline([("preprocessor", preprocessor), ("stacking", stack)])

# Evaluate
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = -cross_val_score(
    pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

# Results
print("Model: Stacking Regressor (Ridge + ElasticNet + XGBoost + LightGBM + CatBoost)")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + Stacking")
print(f"MAE: {mae_scores.mean():.4f}")"""
  },
  # StackNet GradientBoosting Stacking (Layer 1 + Layer 2)
  {
    "Model": "StackNet GradientBoosting Stacking (Layer 1 + Layer 2)",
    "Approach": "Layer 1: Ridge + Lasso + XGBoost; Layer 2: GradientBoostingRegressor; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 8.1737,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocessing
categorical_cols = ["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
     ("num", RobustScaler(), numeric_cols)]
)

# Base models for Layer 1
ridge = Ridge(alpha=10.0, random_state=42)
lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42)
xgb = XGBRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.9,
    colsample_bytree=0.9, random_state=42
)

# Layer 2: Meta-model
gbr = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42
)

# Pipeline wrappers for Layer 1 base models
ridge_pipe = Pipeline([("preprocessor", preprocessor), ("ridge", ridge)])
lasso_pipe = Pipeline([("preprocessor", preprocessor), ("lasso", lasso)])
xgb_pipe = Pipeline([("preprocessor", preprocessor), ("xgb", xgb)])

# Layer 1 Stacking Regressor
layer1 = StackingRegressor(
    estimators=[
        ("ridge", ridge_pipe),
        ("lasso", lasso_pipe),
        ("xgb", xgb_pipe),
    ],
    final_estimator=gbr,  # Meta-model at Layer 1
    cv=5,
    n_jobs=-1
)

# Complete Pipeline with Layer 2 GradientBoostingRegressor
stacknet = layer1

# Evaluate
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = -cross_val_score(
    stacknet, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

# Results
print("Model: StackNet GradientBoosting Stacking (Layer 1 + Layer 2)")
print("Approach: Layer 1: Ridge + Lasso + XGBoost; Layer 2: GradientBoostingRegressor; OneHot + RobustScaler + Repeated 5-Fold CV)")
print(f"MAE: {mae_scores.mean():.4f}")"""
  },
  # Voting Ensemble (Tree-Based Only)
  {
    "Model": "Voting Ensemble (Tree-Based Only)",
    "Approach": "RandomForest + ExtraTrees + LightGBM + XGBoost; Weighted Voting; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.9487,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocessing
categorical_cols = ["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
     ("num", RobustScaler(), numeric_cols)]
)

# Define tree-based models
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
et = ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42)
xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
lgb = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)

# Pipeline wrappers
rf_pipe = Pipeline([("preprocessor", preprocessor), ("rf", rf)])
et_pipe = Pipeline([("preprocessor", preprocessor), ("et", et)])
xgb_pipe = Pipeline([("preprocessor", preprocessor), ("xgb", xgb)])
lgb_pipe = Pipeline([("preprocessor", preprocessor), ("lgb", lgb)])

# Voting Regressor with weights based on inverse MAE (hypothetical, adjust based on results)
voting_ensemble = VotingRegressor(
    estimators=[
        ("rf", rf_pipe),
        ("et", et_pipe),
        ("xgb", xgb_pipe),
        ("lgb", lgb_pipe)
    ],
    weights=[1.0, 1.0, 1.2, 1.2],  # Adjust after empirical tuning
    n_jobs=-1
)

# Evaluate
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = -cross_val_score(
    voting_ensemble, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

# Results
print("Model: Voting Ensemble (Tree-Based Only)")
print("Approach: RandomForest + ExtraTrees + LightGBM + XGBoost; Weighted Voting; OneHot + RobustScaler + Repeated 5-Fold CV")
print(f"MAE: {mae_scores.mean():.4f}")"""
  },
  # Stacked Ensemble (Ridge + ElasticNet + RandomForest + XGBoost)
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "Diverse Feature Sets + OneHot + RobustScaler + Repeated 5-Fold CV + Linear meta-model",
    "MAE": 7.5496,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Define columns
categorical_cols = ["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Variant 1: Full feature set
preprocessor_full = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", RobustScaler(), numeric_cols),
])

# Variant 2: Reduced feature set (drop some correlated features)
reduced_numeric_cols = [
    col for col in numeric_cols if col not in ["Java-2 Theory", "Data Structures using Java Theory"]
]
preprocessor_reduced = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", RobustScaler(), reduced_numeric_cols),
])

# Define models
ridge = Ridge(alpha=10.0)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
rf = RandomForestRegressor(n_estimators=1000, max_depth=15, min_samples_split=5,
                           min_samples_leaf=2, max_features=0.3, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                   subsample=0.9, colsample_bytree=0.9, random_state=42)

# Pipelines with different feature sets
ridge_pipe = Pipeline([("preprocessor", preprocessor_full), ("regressor", ridge)])
elastic_pipe = Pipeline([("preprocessor", preprocessor_reduced), ("regressor", elastic)])
rf_pipe = Pipeline([("preprocessor", preprocessor_full), ("regressor", rf)])
xgb_pipe = Pipeline([("preprocessor", preprocessor_reduced), ("regressor", xgb)])

# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ("ridge", ridge_pipe),
        ("elastic", elastic_pipe),
        ("rf", rf_pipe),
        ("xgb", xgb_pipe),
    ],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1,
    passthrough=False,
)

# Evaluate
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = -cross_val_score(
    stack, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

# Print Results
print("Model: Stacked Regressor (Diverse Feature Sets)")
print("Approach: Stacking with different feature subsets per model + Linear meta-model")
print(f"MAE: {mae_scores.mean():.4f}")"""
  },
  # Bootstrap Aggregated XGBoost
  {
    "Model": "Bootstrap Aggregated XGBoost",
    "Approach": "10 Bootstrapped XGBoost Models + Averaged Predictions + OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.859,
    "Code": """# Target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Define preprocessing
categorical_cols = ["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", RobustScaler(), numeric_cols),
])

# Preprocess once
X_transformed = preprocessor.fit_transform(X)

# Bootstrap Aggregated XGBoost Regressors
n_models = 10
random_state = 42
np.random.seed(random_state)

models = []
predictions = []

# Repeated KFold CV for evaluating ensemble performance
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
mae_scores = []

for train_idx, test_idx in cv.split(X_transformed, y):
    X_train, y_train = X_transformed[train_idx], y.iloc[train_idx]
    X_test, y_test = X_transformed[test_idx], y.iloc[test_idx]
    
    fold_preds = []
    
    for i in range(n_models):
        # Bootstrap sampling
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=random_state + i)
        
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state + i,
            verbosity=0,
        )
        model.fit(X_boot, y_boot)
        pred = model.predict(X_test)
        fold_preds.append(pred)
    
    # Average predictions
    final_pred = np.mean(fold_preds, axis=0)
    fold_mae = mean_absolute_error(y_test, final_pred)
    mae_scores.append(fold_mae)

# Results
print("Model: Bootstrap Aggregated XGBoost (Bagged XGBoost)")
print("Approach: 10 XGBoost models trained on bootstrapped data, averaged predictions across CV")
print(f"MAE: {np.mean(mae_scores):.4f}")"""
  },
  # Voting Regressor (Ridge + Lasso + ElasticNet)
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Define pipelines for individual models
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])
lasso_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]
)
rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Bayesian Optimization for Ridge and Lasso
param_space = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
}

# Ridge Optimization
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space["ridge"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
ridge_search.fit(X, y)

# Lasso Optimization
lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space["lasso"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
lasso_search.fit(X, y)

# Ensemble Model with Ridge, Lasso, and Random Forest
ensemble = VotingRegressor(
    [
        ("ridge", ridge_search.best_estimator_),
        ("lasso", lasso_search.best_estimator_),
        ("rf", rf_pipeline),
    ],
    weights=[0.3, 0.3, 0.4],
)  # Weights based on model performance

# Fit ensemble
ensemble.fit(X, y)

# Evaluate Performance
ridge_mae = -ridge_search.best_score_
lasso_mae = -lasso_search.best_score_
rf_scores = cross_val_score(rf_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
rf_mae = -np.mean(rf_scores)
ensemble_scores = cross_val_score(
    ensemble, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ensemble_mae = -np.mean(ensemble_scores)

# Results
print("Model: Weighted Voting Ensemble (Ridge + Lasso + Random Forest)")
print(
    "Approach: OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights"
)
print(f"MAE: {ensemble_mae:.4f}")"""
  },
  # Voting Regressor (Ridge + Lasso + ElasticNet) (OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Weighted Voting)
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Weighted Voting",
    "MAE": 7.4714,
    "Code": """# Define target
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Pipelines
ridge_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])
lasso_pipe = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]
)
elastic_pipe = Pipeline(
    [("preprocessor", preprocessor), ("regressor", ElasticNet(max_iter=10000))]
)

# BayesSearchCV Parameter Grids
param_space = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "elastic": {
        "regressor__alpha": (1e-3, 1e3, "log-uniform"),
        "regressor__l1_ratio": (0.05, 1.0, "uniform"),
    },
}


def tune_and_score(pipe, space, name):
    search = BayesSearchCV(
        pipe,
        space,
        n_iter=50,
        cv=kf,
        scoring="neg_mean_absolute_error",
        random_state=42,
    )
    search.fit(X, y)
    best_est = search.best_estimator_
    mae = -np.mean(
        cross_val_score(best_est, X, y, scoring="neg_mean_absolute_error", cv=kf)
    )
    print(f"{name} MAE: {mae:.4f}")
    return best_est, mae


ridge_best, ridge_mae = tune_and_score(ridge_pipe, param_space["ridge"], "Ridge")
lasso_best, lasso_mae = tune_and_score(lasso_pipe, param_space["lasso"], "Lasso")
elastic_best, elastic_mae = tune_and_score(
    elastic_pipe, param_space["elastic"], "ElasticNet"
)

# Inverse MAE weighting
total = sum(1 / m for m in [ridge_mae, lasso_mae, elastic_mae])
ridge_wt = (1 / ridge_mae) / total
lasso_wt = (1 / lasso_mae) / total
elastic_wt = (1 / elastic_mae) / total

# Final Voting Regressor
ensemble = VotingRegressor(
    estimators=[
        ("ridge", ridge_best),
        ("lasso", lasso_best),
        ("elastic", elastic_best),
    ],
    weights=[ridge_wt, lasso_wt, elastic_wt],
)
ensemble_mae = -np.mean(
    cross_val_score(ensemble, X, y, scoring="neg_mean_absolute_error", cv=kf)
)

# Output
print("\nModel: Weighted Voting Ensemble (Ridge + Lasso + ElasticNet)")
print(
    "Approach: OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Weighted Voting"
)
print(f"MAE: {ensemble_mae:.4f}")"""
  },
  # Voting Regressor (Ridge + Lasso + ElasticNet) (OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights)
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Define pipelines for individual models
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])

lasso_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]
)

rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Bayesian Optimization search spaces
param_space = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
}

# Ridge Optimization
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space["ridge"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
ridge_search.fit(X, y)

# Lasso Optimization
lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space["lasso"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
lasso_search.fit(X, y)

# Ensemble Model with Ridge, Lasso, and Random Forest
ensemble = VotingRegressor(
    [
        ("ridge", ridge_search.best_estimator_),
        ("lasso", lasso_search.best_estimator_),
        ("rf", rf_pipeline),
    ],
    weights=[0.3, 0.3, 0.4],
)  # Weights based on model performance

# Fit ensemble
ensemble.fit(X, y)

# Evaluate Performance
ridge_mae = -ridge_search.best_score_
lasso_mae = -lasso_search.best_score_
rf_scores = cross_val_score(rf_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
rf_mae = -np.mean(rf_scores)
ensemble_scores = cross_val_score(
    ensemble, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
ensemble_mae = -np.mean(ensemble_scores)

# Results
print("Model: Weighted Voting Ensemble (Ridge + Lasso + Random Forest)")
print(
    "Approach: OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights"
)
print(f"MAE: {ensemble_mae:.4f}")"""
  },
  # Voting Regressor (Ridge + Lasso + ElasticNet) (OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights)
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.4702,
    "Code": """# Define target and features
target_col = 'DE Theory'
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = ['Gender', 'Religion', 'Branch', 'Section-1', 'Section-2', 'Section-3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', RobustScaler(), numeric_cols)
])

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Define pipelines
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=10000))
])

elastic_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=10000))
])

# Search space definitions
param_space = {
    'ridge': {'regressor__alpha': (1e-3, 1e3, 'log-uniform')},
    'lasso': {'regressor__alpha': (1e-3, 1e3, 'log-uniform')},
    'elastic': {
        'regressor__alpha': (1e-3, 1e3, 'log-uniform'),
        'regressor__l1_ratio': (0.0, 1.0, 'uniform')
    }
}

# BayesSearchCV for Ridge
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space['ridge'],
    n_iter=50,
    cv=kf,
    scoring='neg_mean_absolute_error',
    random_state=42
)
ridge_search.fit(X, y)

# BayesSearchCV for Lasso
lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space['lasso'],
    n_iter=50,
    cv=kf,
    scoring='neg_mean_absolute_error',
    random_state=42
)
lasso_search.fit(X, y)

# BayesSearchCV for ElasticNet
elastic_search = BayesSearchCV(
    estimator=elastic_pipeline,
    search_spaces=param_space['elastic'],
    n_iter=50,
    cv=kf,
    scoring='neg_mean_absolute_error',
    random_state=42
)
elastic_search.fit(X, y)

# Ensemble Model: Voting Regressor
ensemble = VotingRegressor([
    ('ridge', ridge_search.best_estimator_),
    ('lasso', lasso_search.best_estimator_),
    ('elastic', elastic_search.best_estimator_)
], weights=[0.3, 0.3, 0.4])

# Fit ensemble
ensemble.fit(X, y)

# Evaluate MAEs
ridge_mae = -ridge_search.best_score_
lasso_mae = -lasso_search.best_score_
elastic_mae = -elastic_search.best_score_
ensemble_scores = cross_val_score(ensemble, X, y, cv=kf, scoring='neg_mean_absolute_error')
ensemble_mae = -np.mean(ensemble_scores)

# Display results
print("Model: Weighted Voting Ensemble (Ridge + Lasso + ElasticNet)")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights")
print(f"MAE: {ensemble_mae:.4f}")"""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV",
    "MAE": 7.1206,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Models and pipelines
ridge_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Ridge())])
lasso_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]
)
elastic_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", ElasticNet(max_iter=10000))]
)

# Cross-validation strategy
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Bayesian Optimization
param_space = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "elastic": {
        "regressor__alpha": (1e-3, 1e3, "log-uniform"),
        "regressor__l1_ratio": (0.05, 1.0, "uniform"),
    },
}

# Ridge Optimization
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space["ridge"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
ridge_search.fit(X, y)

# Lasso Optimization
lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space["lasso"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
lasso_search.fit(X, y)

# ElasticNet Optimization
elastic_search = BayesSearchCV(
    estimator=elastic_pipeline,
    search_spaces=param_space["elastic"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
elastic_search.fit(X, y)

# Ensemble Model
ensemble = VotingRegressor(
    [
        ("ridge", ridge_search.best_estimator_),
        ("lasso", lasso_search.best_estimator_),
        ("elastic", elastic_search.best_estimator_),
    ]
)
ensemble.fit(X, y)

# Results
ensemble_mae = np.mean(np.abs(y - ensemble.predict(X)))
print("Model: Weighted Voting Ensemble (Ridge + Lasso + ElasticNet)")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + Bayesian Optimization")
print(f"MAE: {ensemble_mae:.4f}")"""
  },
  {
    "Model": "Stacking Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Stacking",
    "MAE": 7.4405,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols),
    ]
)

# Cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Bayesian Optimization for base models (Ridge, Lasso, ElasticNet)...")

# --- Ridge Pipeline and Optimization ---
ridge_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", Ridge(random_state=42))]
)
param_space_ridge = {"regressor__alpha": (1e-3, 1e3, "log-uniform")}
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space_ridge,
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
)
ridge_search.fit(X, y)
print(f"Best Ridge MAE: {-ridge_search.best_score_:.4f}")
print(f"Best Ridge params: {ridge_search.best_params_}")

# --- Lasso Pipeline and Optimization ---
lasso_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", Lasso(max_iter=10000, random_state=42)),
    ]
)
param_space_lasso = {"regressor__alpha": (1e-3, 1e3, "log-uniform")}
lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space_lasso,
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
)
lasso_search.fit(X, y)
print(f"Best Lasso MAE: {-lasso_search.best_score_:.4f}")
print(f"Best Lasso params: {lasso_search.best_params_}")

# --- ElasticNet Pipeline and Optimization ---
elastic_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", ElasticNet(max_iter=10000, random_state=42)),
    ]
)
param_space_elastic = {
    "regressor__alpha": (1e-3, 1e3, "log-uniform"),
    "regressor__l1_ratio": (0.05, 1.0, "uniform"),
}
elastic_search = BayesSearchCV(
    estimator=elastic_pipeline,
    search_spaces=param_space_elastic,
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
)
elastic_search.fit(X, y)
print(f"Best ElasticNet MAE: {-elastic_search.best_score_:.4f}")
print(f"Best ElasticNet params: {elastic_search.best_params_}")


# --- Stacking Ensemble Model (only Ridge, Lasso, ElasticNet) ---
print("\nTraining Stacking Regressor with Ridge, Lasso, and ElasticNet...")

# Define the base estimators using their optimized versions
estimators = [
    ("ridge", ridge_search.best_estimator_),
    ("lasso", lasso_search.best_estimator_),
    ("elastic", elastic_search.best_estimator_),
]

# The final_estimator (meta-learner) learns to combine predictions.
# A simple Ridge regressor is a common and effective choice for the meta-learner.
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(random_state=42),
    cv=kf, # Use the same CV strategy for stacking's fitting process
    # n_jobs=-1 is still excluded to prevent the 'cross_val_predict only works for partitions' error
)

# Fit the stacking ensemble
stacking_regressor.fit(X, y)

# --- Evaluate Performance ---
print("\nEvaluating Stacking Regressor performance using cross-validation...")
ensemble_scores = cross_val_score(
    stacking_regressor, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
)
ensemble_mae = -np.mean(ensemble_scores)

# --- Results ---
print("\n--- Model Performance Summary ---")
print("Model: Stacking Regressor (Ridge + Lasso + ElasticNet)")
print("Approach: OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Stacking")
print("Individual Model Best MAEs (from BayesSearchCV):")
print(f"  Ridge MAE: {-ridge_search.best_score_:.4f}")
print(f"  Lasso MAE: {-lasso_search.best_score_:.4f}")
print(f"  ElasticNet MAE: {-elastic_search.best_score_:.4f}")
print(f"Overall Stacking Ensemble MAE (Cross-Validated): {ensemble_mae:.4f}")"""
  },
  {
    "Model": "Full Ensemble (Linear + Tree Models)",
    "Approach": "OneHot + RobustScaler + KFold CV + Stacking",
    "MAE": 7.5555,
    "Code": """# Define target and input
target_col = "DE Theory"
X_raw = df.drop(columns=[target_col])
y = df[target_col]

# Define CV
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# Define base pipelines (CustomPreprocessor will handle preprocessing now)
ridge_pipeline = Pipeline([
    ("preprocessor", CustomPreprocessor()),
    ("regressor", Ridge())
])

lasso_pipeline = Pipeline([
    ("preprocessor", CustomPreprocessor()),
    ("regressor", Lasso(max_iter=10000))
])

elastic_pipeline = Pipeline([
    ("preprocessor", CustomPreprocessor()),
    ("regressor", ElasticNet(max_iter=10000))
])

# Bayesian search space
param_space = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "elastic": {
        "regressor__alpha": (1e-3, 1e3, "log-uniform"),
        "regressor__l1_ratio": (0.05, 1.0, "uniform"),
    },
}

# Run BayesSearchCV
ridge_search = BayesSearchCV(
    estimator=ridge_pipeline,
    search_spaces=param_space["ridge"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
ridge_search.fit(X_raw, y)

lasso_search = BayesSearchCV(
    estimator=lasso_pipeline,
    search_spaces=param_space["lasso"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
lasso_search.fit(X_raw, y)

elastic_search = BayesSearchCV(
    estimator=elastic_pipeline,
    search_spaces=param_space["elastic"],
    n_iter=50,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
)
elastic_search.fit(X_raw, y)

# Create VotingRegressor with tuned base regressors
voting_regressor = VotingRegressor([
    ("ridge", ridge_search.best_estimator_.named_steps["regressor"]),
    ("lasso", lasso_search.best_estimator_.named_steps["regressor"]),
    ("elastic", elastic_search.best_estimator_.named_steps["regressor"]),
])

# Final pipeline with preprocessing and ensemble
final_pipeline = Pipeline([
    ("preprocessor", CustomPreprocessor()),
    ("model", voting_regressor)
])

# Fit on raw data
final_pipeline.fit(X_raw, y)"""
  }
]