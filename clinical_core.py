"""Pure, testable input and laboratory support logic. No model is a diagnosis."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

ADA_SOURCE = "https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes"
FEATURES = ["gender", "hypertension", "heart_disease", "smoking_history", "age", "bmi", "HbA1c_level", "blood_glucose_level"]
SMOKING = ["No Info", "current", "ever", "former", "never", "not current"]


def binary(value):
    if isinstance(value, str):
        value = value.strip().lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    if value not in mapping:
        raise ValueError("Use Yes/No or 1/0 for hypertension and heart disease; unknown is not No.")
    return mapping[value]


def validate_benchmark(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("No records supplied.")
    out = frame[FEATURES].copy()
    for col in ["hypertension", "heart_disease"]:
        out[col] = out[col].map(binary).astype(int)
    for col, options in [("gender", ["Female", "Male", "Other"]), ("smoking_history", SMOKING)]:
        if not out[col].isin(options).all():
            raise ValueError(f"Invalid or missing {col}; allowed: {', '.join(options)}.")
    # Broad plausibility guards, not claims of model validity throughout these ranges.
    for col, bounds in {"age": (0, 100), "bmi": (10, 100), "HbA1c_level": (3, 20), "blood_glucose_level": (20, 700)}.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = ~np.isfinite(out[col]) | ~out[col].between(*bounds)
        if bad.any():
            rows = ", ".join(str(i + 1) for i in np.flatnonzero(bad)[:5])
            raise ValueError(f"{col}: missing, non-numeric or outside {bounds}; data rows {rows}.")
    return out


def probability_text(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Model returned an invalid probability.")
    if value < .0001:
        return "<0.01%"
    if value > .9999:
        return ">99.99%"
    return f"{value * 100:.2f}%"


def screening_frame(frame):
    out = frame[["gender", "smoking_history", "age", "bmi", "hypertension"]].copy()
    out["hypertension"] = out.hypertension.map(binary)
    # 'not current' may include never smokers: do not silently call it 'ever'.
    mapping = {"current": "ever", "former": "ever", "ever": "ever", "never": "never", "No Info": "unknown", "not current": "unknown"}
    if not out.smoking_history.isin(mapping).all():
        raise ValueError("Unrecognized smoking category.")
    out["smoking_history"] = out.smoking_history.map(mapping)
    if not out.gender.isin(["Female", "Male", "Other"]).all():
        raise ValueError("Unrecognized dataset sex/gender category.")
    for col, bounds in {"age": (18, 80), "bmi": (10, 100)}.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if not (np.isfinite(out[col]) & out[col].between(*bounds)).all():
            raise ValueError(f"Screening {col} must be within {bounds}; this is a research input guard, not validation.")
    return out


def interpret_labs(*, age=45, hba1c=None, glucose=None, timing="Unknown", fasting_hours=None,
                   symptoms="Unknown", pregnancy="No / not applicable", a1c_interference=False,
                   repeat_hba1c=None):
    """Nonpregnant adult laboratory ranges; no autonomous diagnosis or treatment.

    Repeat HbA1c must be a separately collected laboratory measurement.
    ADA 2026: https://doi.org/10.2337/dc26-S002
    """
    reasons = []
    def result(code, status, kind, next_step):
        return dict(code=code, status=status, kind=kind, reasons=reasons.copy(), next_step=next_step, source=ADA_SOURCE)
    if not math.isfinite(float(age)) or age < 18 or pregnancy != "No / not applicable":
        return result("outside_scope", "Outside this adult laboratory pathway", "warning", "Use the appropriate pediatric or pregnancy pathway with a clinician.")
    for name, value, low, high in [("HbA1c", hba1c, 3, 20), ("repeat HbA1c", repeat_hba1c, 3, 20), ("glucose", glucose, 20, 700)]:
        if value is not None and (not math.isfinite(float(value)) or not low <= value <= high):
            raise ValueError(f"Verify {name} and units; supported input range is {low}–{high}.")
    if timing not in ["Unknown", "Fasting", "Random"] or symptoms not in ["Unknown", "Yes", "No"]:
        raise ValueError("Invalid timing or symptom context.")
    if fasting_hours is not None and (not math.isfinite(float(fasting_hours)) or not 0 <= fasting_hours <= 24):
        raise ValueError("Verify fasting duration (0–24 hours).")
    if glucose is not None and glucose < 70:
        reasons.append("Glucose is below 70 mg/dL. A low-glucose result must not be presented as reassuring.")
        return result("low_glucose", "Low glucose — clinical review needed", "warning", "Verify the measurement and seek prompt clinical guidance; severe symptoms need urgent care.")
    positive, intermediate, usable = [], [], 0
    a1c_values = [("HbA1c", hba1c), ("separate repeat HbA1c", repeat_hba1c)]
    if a1c_interference:
        reasons.append("HbA1c interpretation withheld because an interference/altered red-cell condition was reported; use clinician-selected plasma-glucose testing.")
    else:
        for name, value in a1c_values:
            if value is None:
                continue
            usable += 1
            if value >= 6.5:
                positive.append(name)
                reasons.append(f"{name} {value:g}% meets the diabetes-range reference (≥6.5%).")
            elif value >= 5.7:
                intermediate.append(name)
                reasons.append(f"{name} {value:g}% is in the increased-risk range (5.7–6.4%).")
            else:
                reasons.append(f"{name} {value:g}% is below the increased-risk reference.")
    valid_fasting = timing == "Fasting" and fasting_hours is not None and fasting_hours >= 8
    if glucose is not None:
        if valid_fasting:
            usable += 1
            if glucose >= 126:
                positive.append("fasting plasma glucose")
                reasons.append(f"Fasting plasma glucose {glucose:g} mg/dL meets the diabetes-range reference (≥126).")
            elif glucose >= 100:
                intermediate.append("fasting plasma glucose")
                reasons.append(f"Fasting plasma glucose {glucose:g} mg/dL is in the increased-risk range (100–125).")
            else:
                reasons.append("Entered fasting plasma glucose is below the increased-risk reference.")
        elif timing == "Random" and glucose >= 200 and symptoms == "Yes":
            reasons.append("Random plasma glucose ≥200 mg/dL with reported classic symptoms meets a guideline criterion; a clinician must verify both.")
            return result("symptomatic_high", "Symptomatic high glucose — prompt clinician assessment", "warning", "Do not delay assessment to obtain a model score. Urgent symptoms require urgent care.")
        else:
            reasons.append("Glucose cannot be classified with fasting criteria: fasting ≥8 hours is unverified, or this is a random measurement.")
    if positive:
        discordant = (usable > len(positive))
        if len(positive) >= 2:
            if discordant:
                reasons.append("Another result is below its cutoff; two qualifying abnormal results still require clinician verification and review of the discrepancy.")
            return result("two_abnormal", "Two diabetes-range laboratory results", "warning", "A clinician must verify specimen context, test quality and diagnostic criteria; these are rule matches, not a model diagnosis.")
        if discordant:
            return result("discordant", "Discordant laboratory evidence", "warning", "Clinician review: repeat the test above its diagnostic reference and consider factors affecting the tests.")
        return result("confirmation", "Diabetes-range result — confirmation needed", "warning", "Arrange clinician review and appropriate confirmatory laboratory testing.")
    if glucose is not None and glucose >= 200:
        return result("high_unclassified", "High glucose — incomplete diagnostic context", "warning", "Prompt clinician review; missing timing or symptoms must not be interpreted as low risk.")
    if intermediate:
        return result("increased_range", "Increased-risk laboratory range", "info", "Discuss follow-up testing and prevention with a clinician. This pathway cannot determine diabetes type.")
    if not usable:
        return result("insufficient", "Not enough interpretable laboratory evidence", "info", "Enter an available laboratory HbA1c or fasting plasma glucose with verified timing. Do not guess missing values.")
    return result("below_range", "Entered interpretable results are below these cutoffs", "info", "This does not rule out diabetes. Symptoms, discordance, medicines and clinical context can require further assessment.")
