import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. LOAD DATA & DATA CLEANING
df = pd.read_csv('Salary_Data.csv')
df_clean = df.dropna().copy()

# Standardize Education Levels
education_map = {
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
}
df_clean['Education Level'] = df_clean['Education Level'].replace(education_map)

# 2. FEATURE ENGINEERING
df_clean['Is_Senior'] = df_clean['Job Title'].str.contains(
    'Senior|Manager|Director|VP|Lead', case=False
).astype(int)

# 3. PEOPLE ANALYTICS SUMMARY
print("=== AVERAGE SALARY BY EDUCATION & GENDER ===")
pay_equity = df_clean.groupby(['Education Level', 'Gender'])['Salary'].mean().unstack()
print(pay_equity.applymap(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"))
print("\n" + "="*50 + "\n")

# 4. PREPARE DATA
X = df_clean[['Age', 'Gender', 'Education Level', 'Job Title', 'Experience', 'Is_Senior']]
y = df_clean['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ['Age', 'Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title', 'Is_Senior']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 5. MODEL PIPELINE & TRAINING
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

# 6. EVALUATION & RESIDUAL ANALYSIS
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== MODEL PERFORMANCE METRICS ===")
print(f"R² Score: {r2:.4f}")
print(f"MAE:      ${mae:,.2f}")
print(f"RMSE:     ${rmse:,.2f}")
print("\n" + "="*50 + "\n")

# Residual Analysis
test_results = X_test.copy()
test_results['Actual_Salary'] = y_test
test_results['Predicted_Salary'] = y_pred
test_results['Residual'] = test_results['Actual_Salary'] - test_results['Predicted_Salary']

std_dev = test_results['Residual'].std()
underpaid = test_results[test_results['Residual'] < -2 * std_dev]
overpaid = test_results[test_results['Residual'] > 2 * std_dev]

print(f"Anomalies Detected:")
print(f"  - Underpaid / Attrition Risk Cases: {len(underpaid)}")
print(f"  - Overpaid / Specialized Cases:    {len(overpaid)}")

# 7. EXPORT MODEL
joblib.dump(model_pipeline, 'salary_predictor_model.pkl')
print("\nSaved model pipeline to 'salary_predictor_model.pkl' successfully!")