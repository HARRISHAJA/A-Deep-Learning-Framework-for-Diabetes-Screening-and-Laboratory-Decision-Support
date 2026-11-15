"""
Kidney Disease Risk Flagging Module
Rule-based layer applied AFTER diabetes prediction.
Uses established clinical risk factors associated with progression
toward diabetic kidney disease: HbA1c, hypertension, age, BMI.

This is the project's original contribution beyond the base paper
(Iftikhar et al., 2025), which does not link its diabetes prediction
to any downstream kidney-risk output.
"""

def assess_kidney_risk(hba1c_level, hypertension, age, bmi, diabetes_predicted):
    """
    Returns a kidney risk flag and reasoning based on established
    clinical risk indicators for diabetic kidney disease progression.

    Thresholds are based on commonly cited clinical risk markers:
      - HbA1c >= 7.0%  -> poor long-term glycemic control
      - Hypertension present -> major CKD progression risk factor
      - Age >= 55       -> age-related kidney function decline risk
      - BMI >= 30        -> obesity-linked renal stress
    """
    risk_points = 0
    reasons = []

    if not diabetes_predicted:
        return {
            "risk_level": "Not Applicable",
            "risk_points": 0,
            "reasons": ["Patient not flagged as diabetic; kidney risk flagging applies to diabetic patients."]
        }

    if hba1c_level >= 8.0:
        risk_points += 2
        reasons.append(f"Very high HbA1c ({hba1c_level}%) indicates poor long-term glucose control")
    elif hba1c_level >= 7.0:
        risk_points += 1
        reasons.append(f"Elevated HbA1c ({hba1c_level}%) above target range")

    if hypertension == 1:
        risk_points += 2
        reasons.append("Hypertension present — major known risk factor for CKD progression")

    if age >= 55:
        risk_points += 1
        reasons.append(f"Age ({age}) associated with natural decline in kidney function")

    if bmi >= 30:
        risk_points += 1
        reasons.append(f"BMI ({bmi}) in obese range — linked to increased renal stress")

    if risk_points >= 4:
        risk_level = "High"
    elif risk_points >= 2:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append("No major additional risk factors identified beyond diabetes status")

    return {
        "risk_level": risk_level,
        "risk_points": risk_points,
        "reasons": reasons
    }


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        {"hba1c_level": 8.5, "hypertension": 1, "age": 60, "bmi": 32, "diabetes_predicted": 1},
        {"hba1c_level": 6.2, "hypertension": 0, "age": 30, "bmi": 22, "diabetes_predicted": 1},
        {"hba1c_level": 5.0, "hypertension": 0, "age": 25, "bmi": 21, "diabetes_predicted": 0},
    ]
    for tc in test_cases:
        result = assess_kidney_risk(**tc)
        print(tc, "->", result)
