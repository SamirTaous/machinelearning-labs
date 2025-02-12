import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# --- Step 1: Load the Dataset ---
# Load the China GDP dataset
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv"
china_gdp = pd.read_csv(url)

# Display the first few rows of the dataset
print("\nChina GDP Dataset:\n", china_gdp.head())

# Separate features (X) and target variable (y)
X = china_gdp[['Year']].values
y = china_gdp['Value'].values

# --- Step 2: Train-Test Split ---
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train Linear Regression Model ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# --- Step 4: Train Polynomial Regression Model ---
# Use PolynomialFeatures to create higher-order terms (degree=3 is an example)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# --- Step 5: Predictions ---
# Linear model predictions
y_pred_linear = linear_model.predict(X_test)

# Polynomial model predictions
y_pred_poly = poly_model.predict(X_poly_test)

# --- Step 6: Visualization ---
plt.figure(figsize=(10, 6))

# Plot linear regression predictions
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, linear_model.predict(X), color='green', label='Linear Regression')

# Plot polynomial regression predictions
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_poly_plot = poly.transform(X_plot)
plt.plot(X_plot, poly_model.predict(X_poly_plot), color='red', label='Polynomial Regression')

# Configure the plot
plt.title('Linear vs Polynomial Regression (China GDP)')
plt.xlabel('Year')
plt.ylabel('GDP Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Step 7: Evaluation ---
# Linear regression evaluation
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# Polynomial regression evaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print("\nLinear Regression:")
print(f"Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear:.2f}")
print(f"Mean Absolute Error (MAE): {mae_linear:.2f}")

print("\nPolynomial Regression:")
print(f"Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_poly:.2f}")
print(f"Mean Absolute Error (MAE): {mae_poly:.2f}")
