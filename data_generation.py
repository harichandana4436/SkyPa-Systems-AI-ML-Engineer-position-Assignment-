
import pandas as pd
import numpy as np
import os


# 1. SET RANDOM SEED


np.random.seed(42)

# Number of records
n = 500



# 2. GENERATE CLAIM DETAILS


data = {
    "id": range(1, n + 1),

    "payer": np.random.choice(
        ["Aetna", "Cigna", "UnitedHC", "BlueCross"], n
    ),

    "provider": np.random.choice(
        ["Dr.Rao", "Dr.Shah", "Dr.Lee", "Dr.Kumar"], n
    ),

    "cpt_code": np.random.choice(
        ["99213", "99214", "99215", "99203"], n
    ),

    "diagnosis_code": np.random.choice(
        ["M54.5", "J06.9", "E11.9", "I10"], n
    ),

    "billed_amount": np.random.randint(100, 1000, n),

    "patient_age": np.random.randint(18, 85, n)
}

# Convert into DataFrame
df = pd.DataFrame(data)



# 3. CREATE SYNTHETIC CLAIM RISK SCORE




risk_score = np.zeros(n)

# Higher billed amount increases simulated risk
risk_score += (
    df["billed_amount"] - df["billed_amount"].mean()
) / df["billed_amount"].std() * 0.8

# Payer risk factors
payer_risk = {
    "Aetna": 0.2,
    "Cigna": 0.4,
    "UnitedHC": 0.6,
    "BlueCross": 0.3
}

risk_score += df["payer"].map(payer_risk)

# CPT code risk factors
cpt_risk = {
    "99213": 0.1,
    "99214": 0.3,
    "99215": 0.7,
    "99203": 0.2
}

risk_score += df["cpt_code"].map(cpt_risk)

# Diagnosis risk factors
diagnosis_risk = {
    "M54.5": 0.2,
    "J06.9": 0.1,
    "E11.9": 0.6,
    "I10": 0.4
}

risk_score += df["diagnosis_code"].map(diagnosis_risk)

# Add small random variation
risk_score += np.random.normal(0, 0.3, n)




# Sort claims from lowest to highest risk
sorted_indices = np.argsort(risk_score)

# Create status column
df["status"] = "Paid"

# Highest-risk 10% become Denied
denied_count = int(n * 0.10)

# Next 20% become Partially_Paid
partial_count = int(n * 0.20)

df.loc[
    sorted_indices[-denied_count:],
    "status"
] = "Denied"

df.loc[
    sorted_indices[-(denied_count + partial_count):-denied_count],
    "status"
] = "Partially_Paid"



# 5. SHUFFLE DATASET


df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)



# 4. SAVE DATASET

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/claims_dataset.csv",
    index=False
)

print("Dataset created successfully!")



# 5. DISPLAY DATASET INFORMATION


print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nClaim Status Distribution:")
print(df["status"].value_counts())

print("\nClaim Status Percentage:")
print(
    df["status"].value_counts(normalize=True) * 100
)
