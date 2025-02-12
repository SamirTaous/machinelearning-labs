import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
insurance = pd.read_csv('datasets/insurance.csv')

# --- Part 1: EDA ---
# Basic Information
# print("Dataset Info:")
# print(insurance.info())
# print("\nStatistical Summary:")
# print(insurance.describe())

# # Check for missing values
# print("\nMissing Values:")
# print(insurance.isnull().sum())

# # Visualize distributions
# print("\nVisualizing Distributions:")
# insurance.hist(bins=20, figsize=(10, 8))
# plt.tight_layout()
# plt.show()

# Use one-hot encoding to handle categorical variables
insurance_encoded = pd.get_dummies(insurance, drop_first=True)

# Display the first few rows to confirm encoding
print("\nEncoded Dataset:\n", insurance_encoded.head())

# --- Step 2: Calculate Correlation and Select Features ---
# Calculate correlation with the target variable 'charges'
correlations = insurance_encoded.corr()['charges'].sort_values(ascending=False)
print("\nCorrelation with Target Variable 'charges':")
print(correlations)


# Select the top 3 features with the highest correlation (excluding the target itself)
selected_features = correlations.index[1:4].tolist()
print("\nSelected Features for Regression:", selected_features)

# --- Step 3: Prepare Data for Regression ---
# Extract the selected features (X) and the target variable (y)
X = insurance_encoded[selected_features]
y = insurance_encoded['charges']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Standardize the Features ---
# Standardize the features to ensure they are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 5: Train the Linear Regression Model ---
# Initialize the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Display the model coefficients
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# --- Step 6: Make Predictions on the Test Data ---
# Predict the target variable for the test set
y_pred = model.predict(X_test_scaled)

# Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Charges:")
print(results.head())

# --- Step 7: Visualize the Regression Results ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.title("Actual vs Predicted Charges")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Step 8: Evaluate the Model ---
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")