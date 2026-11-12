
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_claims.csv")


# Univariate Analysis - Numerical Features

# Distribution of billed amount
plt.figure(figsize=(8, 5))
plt.hist(df["billed_amount"], bins=20, edgecolor="black")
plt.title("Distribution of Billed Amount")
plt.xlabel("Billed Amount")
plt.ylabel("Frequency")
plt.show()

# Distribution of patient age
plt.figure(figsize=(8, 5))
plt.hist(df["patient_age"], bins=15, edgecolor="black")
plt.title("Distribution of Patient Age")
plt.xlabel("Patient Age")
plt.ylabel("Frequency")
plt.show()

# Bivariate Analysis

# Billed Amount vs Status
plt.figure(figsize=(8, 5))
df.boxplot(column="billed_amount", by="status")
plt.title("Billed Amount vs Claim Status")
plt.xlabel("Claim Status")
plt.ylabel("Billed Amount")
plt.show()


# Patient Age vs Status
plt.figure(figsize=(8, 5))
df.boxplot(column="patient_age", by="status")
plt.title("Patient Age vs Claim Status")
plt.xlabel("Claim Status")
plt.ylabel("Patient Age")
plt.show()


# Payer vs Status
payer_status = pd.crosstab(df["payer"], df["status"])

print("\nPayer vs Claim Status:")
print(payer_status)

payer_status.plot(kind="bar", figsize=(9, 5))
plt.title("Payer vs Claim Status")
plt.xlabel("Payer")
plt.ylabel("Number of Claims")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Diagnosis Code vs Status
diagnosis_status = pd.crosstab(
    df["diagnosis_code"], df["status"]
)

print("\nDiagnosis Code vs Claim Status:")
print(diagnosis_status)

diagnosis_status.plot(kind="bar", figsize=(10, 5))
plt.title("Diagnosis Code vs Claim Status")
plt.xlabel("Diagnosis Code")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()




# Multivariate Analysis

import seaborn as sns

# Correlation between numerical columns
numerical_df = df[["billed_amount", "patient_age"]]

correlation = numerical_df.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()