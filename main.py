import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware

# ✅ Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add Rate Limiter (10 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.get("/")
@limiter.limit("10/minute")  # Limit root endpoint to 10 requests per minute
async def read_root(request: Request):
    return {"message": "Welcome to Breast Cancer Detection API"}

# ✅ Load the dataset (excluding 'id' column)
csv_path = "breast-cancer.csv"
df = pd.read_csv(csv_path)
df.drop(columns=["id"], inplace=True)  # Drop ID column

# ✅ Train Logistic Regression Model
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)  # Convert 'M' -> 1, 'B' -> 0
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# ✅ Define Input Model with Constraints
class CancerFeatures(BaseModel):
    radius_mean: float = Field(..., ge=5, le=30)
    texture_mean: float = Field(..., ge=5, le=40)
    perimeter_mean: float = Field(..., ge=40, le=200)
    area_mean: float = Field(..., ge=100, le=2500)
    smoothness_mean: float = Field(..., ge=0.05, le=0.2)
    compactness_mean: float = Field(..., ge=0.01, le=1.5)
    concavity_mean: float = Field(..., ge=0.01, le=1.5)
    concave_points_mean: float = Field(..., ge=0.01, le=1.5)
    symmetry_mean: float = Field(..., ge=0.1, le=0.5)
    fractal_dimension_mean: float = Field(..., ge=0.04, le=0.2)
    radius_se: float = Field(..., ge=0.1, le=3)
    texture_se: float = Field(..., ge=0.1, le=5)
    perimeter_se: float = Field(..., ge=0.1, le=10)
    area_se: float = Field(..., ge=1, le=200)
    smoothness_se: float = Field(..., ge=0.001, le=0.05)
    compactness_se: float = Field(..., ge=0.001, le=0.5)
    concavity_se: float = Field(..., ge=0.001, le=0.5)
    concave_points_se: float = Field(..., ge=0.001, le=0.2)
    symmetry_se: float = Field(..., ge=0.01, le=0.5)
    fractal_dimension_se: float = Field(..., ge=0.001, le=0.1)
    radius_worst: float = Field(..., ge=5, le=40)
    texture_worst: float = Field(..., ge=5, le=50)
    perimeter_worst: float = Field(..., ge=40, le=250)
    area_worst: float = Field(..., ge=100, le=3000)
    smoothness_worst: float = Field(..., ge=0.05, le=0.3)
    compactness_worst: float = Field(..., ge=0.01, le=1.5)
    concavity_worst: float = Field(..., ge=0.01, le=1.5)
    concave_points_worst: float = Field(..., ge=0.01, le=1.5)
    symmetry_worst: float = Field(..., ge=0.1, le=0.6)
    fractal_dimension_worst: float = Field(..., ge=0.04, le=0.3)

# ✅ POST Route for Predictions
@app.post("/predict/")
@limiter.limit("10/minute")
async def predict_cancer(request: Request, features: CancerFeatures):
    try:
        # Convert input to numpy array
        input_data = np.array([list(features.model_dump().values())]).reshape(1, -1)

        # Get prediction
        prediction = model.predict(input_data)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        # Get prediction probability in percentage
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(input_data)[0][prediction] * 100

        return {
            "prediction": result,
            "confidence": f"{confidence:.2f}%" if confidence is not None else "N/A"
        }

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")