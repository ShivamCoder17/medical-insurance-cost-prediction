
from pathlib import Path
import math
from numbers import Real

import joblib
import pandas as pd


# Resolve model path relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "insurance_cost_pipeline.joblib"
)

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGIONS = {
    "northeast",
    "northwest",
    "southeast",
    "southwest"
}


def load_model():
    """Load the serialized insurance prediction pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_insurance_cost(
    age,
    sex,
    bmi,
    children,
    smoker,
    region
):
    """Predict insurance charges from customer information."""

    if (
        isinstance(age, bool)
        or not isinstance(age, Real)
        or not math.isfinite(age)
        or not 0 < age <= 120
    ):
        raise ValueError("Age must be between 1 and 120.")

    if (
        isinstance(bmi, bool)
        or not isinstance(bmi, Real)
        or not math.isfinite(bmi)
        or not 0 < bmi < 100
    ):
        raise ValueError("BMI must be greater than 0 and below 100.")

    if (
        isinstance(children, bool)
        or not isinstance(children, int)
        or children < 0
    ):
        raise ValueError("Children must be a non-negative integer.")

    if not isinstance(sex, str) or sex.lower() not in VALID_SEX:
        raise ValueError("Sex must be male or female.")

    if not isinstance(smoker, str) or smoker.lower() not in VALID_SMOKER:
        raise ValueError("Smoker must be yes or no.")

    if not isinstance(region, str) or region.lower() not in VALID_REGIONS:
        raise ValueError("Invalid region.")

    customer = pd.DataFrame([{
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower()
    }])

    model = load_model()
    prediction = model.predict(customer)[0]

    return round(float(prediction), 2)


if __name__ == "__main__":

    prediction = predict_insurance_cost(
        age=35,
        sex="male",
        bmi=28.5,
        children=1,
        smoker="no",
        region="northwest"
    )

    print(f"Predicted insurance charge: {prediction:,.2f}")
