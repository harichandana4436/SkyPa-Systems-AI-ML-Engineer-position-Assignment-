
import pandas as pd
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



# 1. CREATE FASTAPI APPLICATION


app = FastAPI(
    title="Claim Outcome Prediction API",
    description="API for predicting insurance claim outcomes",
    version="1.0"
)



# 2. LOAD TRAINED MODEL


try:
    with open("models/claim_pipeline.pkl", "rb") as file:
        model_bundle = pickle.load(file)

    pipeline = model_bundle["pipeline"]

    payer_denial_rate = model_bundle["payer_denial_rate"]
    global_denial_rate = model_bundle["global_denial_rate"]
    amount_median = model_bundle["amount_median"]

    feature_columns = model_bundle["feature_columns"]

    age_bins = model_bundle["age_bins"]
    age_labels = model_bundle["age_labels"]

    print("Model Loaded Successfully!")

except Exception as error:
    raise RuntimeError(f"Unable to load model: {error}")



# 3. DEFINE INPUT VALIDATION


class ClaimInput(BaseModel):

    payer: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    cpt_code: str = Field(min_length=1)
    diagnosis_code: str = Field(min_length=1)

    billed_amount: float = Field(gt=0)

    patient_age: int = Field(ge=1, le=120)




@app.get("/")
def home():

    return {
        "message": "Claim Outcome Prediction API is running",
        "docs": "/docs"
    }




@app.post("/predict")
def predict_claim(claim_data: ClaimInput):

    try:

        # Convert input into DataFrame

        claim = pd.DataFrame([{
            "payer": claim_data.payer,
            "provider": claim_data.provider,
            "cpt_code": claim_data.cpt_code,
            "diagnosis_code": claim_data.diagnosis_code,
            "billed_amount": claim_data.billed_amount,
            "patient_age": claim_data.patient_age
        }])


        # FEATURE ENGINEERING
        

        claim["age_group"] = pd.cut(
            claim["patient_age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )

        claim["high_billed_amount"] = (
            claim["billed_amount"] > amount_median
        ).astype(int)

        claim["payer_denial_rate"] = (
            claim["payer"]
            .map(payer_denial_rate)
            .fillna(global_denial_rate)
        )

        

        claim = claim[feature_columns]

         # MODEL PREDICTION


        prediction = pipeline.predict(claim)[0]

        probabilities = pipeline.predict_proba(claim)[0]

        classes = pipeline.named_steps["classifier"].classes_

# RETURN API RESPONSE
        

        class_probabilities = {
            str(class_name): round(float(probability), 4)
            for class_name, probability in zip(classes, probabilities)
        }

        return {
            "success": True,
            "predicted_status": str(prediction),
            "class_probabilities": class_probabilities
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
