
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load original dataset
df = pd.read_csv("data/claims_dataset.csv")

print("Dataset loaded successfully")
print(df.head())
print("Shape:", df.shape)
print(df.isnull().sum())


# Check duplicate rows
print("Total duplicates:", df.duplicated().sum())

# Check duplicate IDs
print("Duplicate IDs:", df["id"].duplicated().sum())

print("Data Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()



# Function to detect outliers
def check_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print("\nColumn:", column)
    print("Lower limit:", lower)
    print("Upper limit:", upper)
    print("Outlier count:", len(outliers))

    return outliers

# Check numerical columns
amount_outliers = check_outliers(df, "billed_amount")
age_outliers = check_outliers(df, "patient_age")

plt.figure(figsize=(8, 5))

plt.boxplot(df["billed_amount"])

plt.title("Boxplot of Billed Amount")
plt.ylabel("Billed Amount")
plt.xlabel("Insurance Claims")

plt.show()


plt.figure(figsize=(8, 5))

plt.boxplot(df["patient_age"])

plt.title("Boxplot of Patient Age")
plt.ylabel("Patient Age")
plt.xlabel("Insurance Claims")

plt.show()



# Count each class
print("Class Distribution:")
print(df["status"].value_counts())

# Calculate percentage
print("\nClass Percentage:")
print(df["status"].value_counts(normalize=True) * 100)

# Count status values
status_counts = df["status"].value_counts()

# Create bar chart
plt.figure(figsize=(8, 5))

plt.bar(status_counts.index, status_counts.values)

plt.title("Class Distribution of Claim Status")
plt.xlabel("Claim Status")
plt.ylabel("Number of Records")

plt.show()


df.to_csv("data/cleaned_claims.csv", index=False)

print("\nData cleaning completed successfully!")
