# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv("unemployment.csv")

# Step 3: Clean Column Names (remove extra spaces)
data.columns = data.columns.str.strip()

print("First 5 rows of dataset:")
print(data.head())

# Step 4: Basic Info
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Step 5: Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop missing values if any
data = data.dropna()

# Step 6: Overall Unemployment Trend
plt.figure(figsize=(12,6))
sns.lineplot(x="Date", y="Estimated Unemployment Rate (%)", data=data, marker="o")
plt.title("Unemployment Trend Over Time")
plt.xticks(rotation=90)
plt.show()

# Step 7: Region-wise Unemployment
plt.figure(figsize=(14,7))
sns.barplot(x="Region", y="Estimated Unemployment Rate (%)", data=data, ci=None)
plt.xticks(rotation=90)
plt.title("Region-wise Unemployment Rate")
plt.show()

# Step 8: Rural vs Urban Comparison
plt.figure(figsize=(10,6))
sns.boxplot(x="Area", y="Estimated Unemployment Rate (%)", data=data)
plt.title("Rural vs Urban Unemployment Comparison")
plt.show()

# Step 9: Covid-19 Impact (March 2020 to Dec 2020)
covid_data = data[data["Date"].between("31-03-2020", "31-12-2020")]
plt.figure(figsize=(12,6))
sns.lineplot(x="Date", y="Estimated Unemployment Rate (%)", data=covid_data, marker="o", color="red")
plt.title("Covid-19 Impact on Unemployment")
plt.xticks(rotation=90)
plt.show()

# Step 10: Insights
highest_region = data.groupby("Region")["Estimated Unemployment Rate (%)"].mean().idxmax()
lowest_region = data.groupby("Region")["Estimated Unemployment Rate (%)"].mean().idxmin()

print(f"\nRegion with highest average unemployment: {highest_region}")
print(f"Region with lowest average unemployment: {lowest_region}")
