from pathlib import Path

import pandas as pd
from pypmml import Model


_MODEL = None


def _load_pmml_model():
    global _MODEL

    if _MODEL is None:
        pmml_path = str(
            Path(__file__).parent / "loan_approval_xgb_model_1.pmml"
        )

        _MODEL = Model.load(pmml_path)

    return _MODEL


def predict_risk(applicant: dict) -> dict:
    """
    Run PMML loan approval model.

    Parameters
    ----------
    applicant : dict

    Returns
    -------
    dict
    """

    model = _load_pmml_model()

    df = pd.DataFrame([applicant])

    result = model.predict(df)

    prob_low = float(result["probability(0)"].iloc[0])
    prob_high = float(result["probability(1)"].iloc[0])

    risk = (
        "Low Risk"
        if prob_low >= prob_high
        else "High Risk"
    )

    return {
        "risk": risk,
        "probability_low": prob_low,
        "probability_high": prob_high,
    }