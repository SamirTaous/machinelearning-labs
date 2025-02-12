import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Load datasets
experience_salary = pd.read_csv('datasets/Salary_Data.csv')
insurance = pd.read_csv('datasets/insurance.csv')

# Explore datasets
print("Experience-Salary Dataset Info:")
print(experience_salary.info())
print("\nInsurance Dataset Info:")
print(insurance.info())

# Display statistical summaries
print("Experience-Salary Dataset Summary:\n", experience_salary.describe())
print("\nInsurance Dataset Summary:\n", insurance.describe())

# Scatter plot for Experience vs Salary
plt.scatter(experience_salary['YearsExperience'], experience_salary['Salary'], color='blue', alpha=0.7)
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Scatter matrix for Insurance dataset
scatter_matrix(insurance, figsize=(10, 10), diagonal='kde', alpha=0.7)
plt.suptitle("Scatter Matrix for Insurance Dataset")
plt.show()
