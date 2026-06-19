# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
data = pd.read_csv("Advertising.csv")

# Clean column names (remove spaces if any)
data.columns = data.columns.str.strip()

print("First 5 rows of dataset:")
print(data.head())

# Step 3: Basic Info
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
sns.scatterplot(x="TV", y="Sales", data=data)
plt.subplot(1,3,2)
sns.scatterplot(x="Radio", y="Sales", data=data)
plt.subplot(1,3,3)
sns.scatterplot(x="Newspaper", y="Sales", data=data)
plt.suptitle("Advertising Channels vs Sales")
plt.show()

# Step 5: Prepare Data for Model
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Model Evaluation
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Visualization of Predictions
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
