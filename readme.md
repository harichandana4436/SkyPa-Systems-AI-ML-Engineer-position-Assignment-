# Claim Outcome Prediction

## Project Overview

This project focuses on predicting insurance claim outcomes using Machine Learning. The model predicts whether an insurance claim will be **Paid, Partially_Paid, or Denied** based on details such as payer, provider, CPT code, diagnosis code, billed amount, and patient age.

The project uses a synthetic insurance claims dataset and a Random Forest classification model. A FastAPI application is also developed to accept claim details and return predictions with class probabilities.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Random Forest Classifier
- FastAPI
- Uvicorn
- Pickle

## Dataset Description

The dataset contains synthetic insurance claim records with the following columns:

- `id` – Unique claim identifier
- `payer` – Insurance company
- `provider` – Healthcare provider
- `cpt_code` – Medical procedure code
- `diagnosis_code` – Patient diagnosis code
- `billed_amount` – Amount charged for the claim
- `patient_age` – Age of the patient
- `status` – Final claim outcome

The dataset contains three target classes:

- Paid – 70%
- Partially_Paid – 20%
- Denied – 10%

## Project Structure

```text
SkyPa Systems assesment/
│
├── data/
│   ├── claims_dataset.csv
│   ├── cleaned_claims.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── X_train_balanced.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── y_train_balanced.csv
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

## Data Preparation

The dataset is generated using Python with synthetic insurance claim details.

The following data preparation steps are performed:

- Checking missing values
- Checking duplicate records
- Validating patient age and billed amount
- Checking unique claim statuses
- Cleaning and saving the processed dataset

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset and identify patterns.

The analysis includes:

- Distribution of claim statuses
- Analysis of insurance payers
- Analysis of healthcare providers
- Billed amount distribution
- Patient age distribution
- Visualizations using Matplotlib and Seaborn

## Feature Engineering

The following additional features are created:

- `age_group` – Groups patients based on their age.
- `high_billed_amount` – Identifies claims with higher billed amounts using the training median.
- `payer_denial_rate` – Calculates the denial rate for each payer using training data.

## Model Development

A Random Forest Classifier is used to predict insurance claim outcomes.

The model development process includes:

1. Separating input features and target variable.
2. Splitting the data into training and testing sets using stratified sampling.
3. Handling numerical and categorical features.
4. Applying SMOTENC to balance the training data.
5. Training the Random Forest classification model.
6. Evaluating the model using accuracy, precision, recall, F1-score, and confusion matrix.

The testing data is kept separate from the balancing process.

## Model Evaluation

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help understand how well the model predicts each claim outcome.

## Model Saving

The trained machine learning pipeline and required metadata are saved using Python's built-in `pickle` module.

Saved model location:

```text
models/claim_pipeline.pkl
```

## Prediction Testing

The `test_prediction.py` file allows users to enter claim details manually.

The program displays:

- Predicted claim status
- Probability of Paid
- Probability of Partially_Paid
- Probability of Denied

It can also compare the predicted status with an actual status entered by the user.

## FastAPI Application

FastAPI is used to deploy the trained model through an API.

**Available endpoints:**

- `GET /` – Displays the application information.
- `POST /predict` – Accepts claim details and returns the predicted claim status with class probabilities.

The API documentation is available through Swagger UI.

## How to Run the Project

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

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /predict` endpoint to test the model with new claim details.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for insurance claim outcome prediction, starting from synthetic data generation and preprocessing to model training, evaluation, prediction, and API deployment.

The project was developed as part of the SkyPa Systems assessment. Since the dataset is synthetic, the predictions are intended for demonstration and learning purposes only.
