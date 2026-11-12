# Claim Outcome Prediction

## Project Overview

This project is developed as part of the SkyPa Systems technical assessment. The main objective is to predict the outcome of insurance claims based on details such as payer, provider, CPT code, diagnosis code, billed amount, and patient age.

The project uses Python and Machine Learning to classify insurance claims into three categories: Paid, Partially Paid, and Denied. It also includes a FastAPI application that allows users to submit claim details and receive predictions through an API.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- Random Forest Classifier
- FastAPI
- Uvicorn
- Pickle

## Dataset Description

A synthetic dataset containing 500 insurance claim records was created for this project.

The dataset contains the following columns:

| Column | Description |
|---|---|
| id | Unique claim ID |
| payer | Insurance company |
| provider | Healthcare provider |
| cpt_code | Medical procedure code |
| diagnosis_code | Diagnosis code |
| billed_amount | Amount charged for the claim |
| patient_age | Age of the patient |
| status | Claim outcome |

The target variable is `status`, which contains three classes:

- Paid – 70%
- Partially_Paid – 20%
- Denied – 10%

The dataset is synthetic and intended for learning and assessment purposes.

## Project Workflow

### 1. Data Generation

Created 500 synthetic insurance claim records with the required columns and claim status distribution. Illustrative risk-based rules were used to assign claim outcomes so that the model could learn patterns from the generated data.

### 2. Data Cleaning

Loaded the generated dataset and checked for missing values, duplicate records, invalid ages, invalid billed amounts, and inconsistent claim statuses.

The cleaned dataset was saved for further analysis.

### 3. Exploratory Data Analysis (EDA)

Performed exploratory data analysis to understand the dataset.

The analysis included:

- Distribution of claim statuses.
- Distribution of billed amounts.
- Patient age distribution.
- Claim status comparisons.
- Relationships between numerical and categorical features.

Used Matplotlib and Seaborn to create visualizations.

### 4. Feature Engineering

Created additional features to improve the model input:

- **Age Group:** Categorized patients into different age groups.
- **High Billed Amount:** Identified claims with billed amounts above the training-data median.
- **Payer Denial Rate:** Calculated the denial rate for each payer using training data only.

### 5. Handling Class Imbalance

The original dataset contains an unequal distribution of claim statuses.

Used SMOTENC to balance the training data because it contains both numerical and categorical features.

SMOTENC was applied only to the training data. The test data was kept unchanged to evaluate the model on the original distribution.

### 6. Model Training

Used the Random Forest Classifier to predict insurance claim outcomes.

A stratified train-test split was performed to maintain the original class distribution in both sets.

A preprocessing pipeline was created using:

- SimpleImputer
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- SMOTENC
- Random Forest Classifier

The trained pipeline and required training information were saved using Python's built-in Pickle module.

### 7. Model Evaluation

Evaluated the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help understand how well the model predicts each claim status.

### 8. Prediction Testing

Created a separate Python file to test the saved model with new claim details.

The program accepts claim information, predicts the claim status, displays class probabilities, and optionally compares the prediction with a known actual status.

### 9. FastAPI Deployment

Created a FastAPI application to make predictions through an API.

The application provides:

- `GET /` – Checks whether the API is running.
- `POST /predict` – Accepts claim details and returns the predicted status and class probabilities.

Swagger documentation is available at:

`http://127.0.0.1:8000/docs`

## Project Structure

```text
SkyPa Systems assesment/
│
├── data/
│   ├── claims_dataset.csv
│   └── cleaned_claims.csv
│
├── models/
│   └── claim_pipeline.pkl
│
├── data_generation.py
├── data_cleaning.py
├── eda.py
├── train_model.py
├── test_prediction.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Execution

### Step 1: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 2: Generate the dataset

```bash
python data_generation.py
```

### Step 3: Clean the dataset

```bash
python data_cleaning.py
```

### Step 4: Perform EDA

```bash
python eda.py
```

### Step 5: Train the model

```bash
python train_model.py
```

### Step 6: Test the prediction

```bash
python test_prediction.py
```

### Step 7: Start the FastAPI application

```bash
uvicorn app:app --reload
```

### Step 8: Open API documentation

Open the following URL in your browser:

`http://127.0.0.1:8000/docs`

Use the POST `/predict` endpoint to submit claim details and receive a prediction.

## Conclusion

Through this project, I worked on the complete machine learning workflow, starting from synthetic data generation and data cleaning to exploratory analysis, feature engineering, model training, evaluation, and API deployment.

This project helped me understand how machine learning models can be developed and exposed through APIs for making predictions.

**Note:** This project uses synthetic data and illustrative risk rules. Its predictions are for assessment and demonstration purposes and should not be treated as actual insurance claim decisions.