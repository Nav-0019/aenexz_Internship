import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('visualizations', exist_ok=True)

# 1. Load Dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()

print("--- 1. DATASET OVERVIEW ---")
print(f"Dataset shape: {df.shape}")
print("Column names:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe().T[['mean', 'std', 'min', '50%', 'max']])

# 2. EDA Visualizations
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

feature_names = diabetes.feature_names
for i, col in enumerate(feature_names):
    sns.histplot(df[col], kde=True, ax=axes[i], color='#1f77b4', bins=20)
    axes[i].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')

# Target distribution
sns.histplot(df['target'], kde=True, ax=axes[10], color='#d62728', bins=20)
axes[10].set_title('Target: Disease Progression', fontsize=12, fontweight='bold')
axes[10].set_xlabel('Quantitative Measure')

axes[11].axis('off')
plt.suptitle('Exploratory Data Analysis: Feature & Target Distributions', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('visualizations/eda_distributions.png', dpi=300)
plt.close()

# 3. Correlation Analysis
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix (Features & Target)', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
plt.close()

target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
print("\n--- 3. CORRELATION WITH TARGET ---")
print(target_corr)

# Barplot of correlation with target
plt.figure(figsize=(10, 6))
colors = ['#2ca02c' if c > 0 else '#d62728' for c in target_corr.values]
sns.barplot(x=target_corr.values, y=target_corr.index, palette=colors)
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title('Feature Correlation with Disease Progression Target', fontsize=14, fontweight='bold')
plt.xlabel('Pearson Correlation Coefficient')
plt.tight_layout()
plt.savefig('visualizations/target_correlation_bar.png', dpi=300)
plt.close()

# 4. Train Linear Regression Model
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 5. Evaluate
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n--- 5. MODEL EVALUATION ---")
print(f"Training R²:   {r2_train:.4f} | Testing R²:   {r2_test:.4f}")
print(f"Training MAE:  {mae_train:.4f} | Testing MAE:  {mae_test:.4f}")
print(f"Training RMSE: {rmse_train:.4f} | Testing RMSE: {rmse_test:.4f}")

# 6. Feature Coefficients / Influential Features
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values(by='Abs_Coefficient', ascending=False)

print("\n--- 6. FEATURE COEFFICIENTS ---")
print(coef_df.to_string(index=False))

plt.figure(figsize=(10, 6))
coef_sorted = coef_df.sort_values(by='Coefficient', ascending=True)
bar_colors = ['#d62728' if c < 0 else '#1f77b4' for c in coef_sorted['Coefficient']]
plt.barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=bar_colors)
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title('Linear Regression Feature Coefficients (Direction & Impact)', fontsize=14, fontweight='bold')
plt.xlabel('Regression Coefficient')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300)
plt.close()

# 7. Compare Actual vs Predicted
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Error (Residual)': y_test.values - y_test_pred,
    'Abs_Error': np.abs(y_test.values - y_test_pred)
})

print("\n--- 7. ACTUAL VS PREDICTED (Sample First 10) ---")
print(comparison_df.head(10).round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Actual vs Predicted
axes[0].scatter(y_test, y_test_pred, color='#1f77b4', edgecolors='k', alpha=0.7, s=60, label='Test Samples')
# 45 degree perfect line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
# Best fit regression line
m, b = np.polyfit(y_test, y_test_pred, 1)
axes[0].plot(y_test, m*y_test + b, color='darkblue', linestyle=':', lw=2, label=f'Best Fit Line: y={m:.2f}x+{b:.2f}')
axes[0].set_title('Actual vs Predicted Disease Progression (Test Set)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Actual Disease Progression', fontsize=12)
axes[0].set_ylabel('Predicted Disease Progression', fontsize=12)
axes[0].legend()

# Subplot 2: Residuals vs Predicted
residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals, color='#2ca02c', edgecolors='k', alpha=0.7, s=60)
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Disease Progression', fontsize=12)
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/actual_vs_predicted.png', dpi=300)
plt.close()

print("\nAll visualizations saved successfully in visualizations/ folder.")

