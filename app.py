"""Patient-friendly Streamlit dashboard for the six-model diabetes benchmark.

Run from this directory:
    streamlit run app.py

The dashboard is a research/education prototype. It is not a diagnostic device
and must not replace a clinician or emergency care.
"""

from __future__ import annotations

import io
import hashlib
import json
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from clinical_core import validate_benchmark, probability_text
import clinical_workspace


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "benchmark_outputs"
MODEL_NAMES = ["XGBoost", "DNN", "MLP", "TabNet", "Wide and Deep", "TabPFN"]
RAW_INPUT_COLUMNS = [
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

st.set_page_config(
    page_title="DiaLens: A Deep Learning Framework for Diabetes Screening and Laboratory Decision Support",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 3rem; }
        .hero { padding: 1.35rem 1.55rem; border-radius: 20px; background: linear-gradient(120deg,#123C55 0%,#167D8D 58%,#4C9FA1 100%); color: #fff; margin-bottom: .85rem; box-shadow: 0 12px 30px rgba(18,60,85,.16); animation: rise-in .5s ease-out both; }
        .hero h1 { margin: 0 0 .3rem 0; font-size: clamp(1.65rem, 3vw, 2.25rem); letter-spacing: -.025em; }
        .hero p { margin: 0; color: rgba(255,255,255,.88); font-size: .98rem; }
        .hero-meta { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .85rem; }
        .hero-chip { display: inline-flex; align-items: center; padding: .28rem .62rem; border: 1px solid rgba(255,255,255,.28); border-radius: 999px; background: rgba(255,255,255,.11); font-size: .78rem; }
        .notice { padding: .72rem .9rem; border-radius: 12px; background: #FFF4E5; border: 1px solid #F0D3A2; color: #76430C; animation: rise-in .45s .08s ease-out both; }
        .safe { padding: .72rem .9rem; border-radius: 12px; background: #E8F6EF; border: 1px solid #B8DEC9; color: #0F5B40; }
        .section-kicker { color: #167D8D; font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: -.45rem; }
        .helper-line { color: #5B7180; font-size: .87rem; margin-top: -.45rem; }
        .result-card { padding: .85rem 1rem; border-radius: 14px; background: #FFFFFF; border: 1px solid #D6E4E8; box-shadow: 0 5px 18px rgba(23,48,66,.06); animation: rise-in .35s ease-out both; }
        .compact-note { color: #5B7180; font-size: .82rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stMetric"] { border-radius: 14px; transition: transform .2s ease, box-shadow .2s ease; }
        [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 7px 18px rgba(23,48,66,.08); }
        .stButton > button, .stDownloadButton > button { transition: transform .18s ease, box-shadow .18s ease, background-color .18s ease; }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(22,125,141,.16); }
        div[data-baseweb="tab-list"] { gap: .25rem; padding-bottom: .18rem; }
        button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; transition: background-color .2s ease, color .2s ease; }
        button[data-baseweb="tab"]:hover { background: #E4F0F3; }
        @keyframes rise-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_results() -> tuple[pd.DataFrame, dict, dict]:
    results_path = OUTPUT_DIR / "benchmark_results.csv"
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    matrices_path = OUTPUT_DIR / "confusion_matrices.json"
    if not results_path.exists():
        return pd.DataFrame(), {}, {}
    results = pd.read_csv(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    matrices = json.loads(matrices_path.read_text(encoding="utf-8")) if matrices_path.exists() else {}
    return results, summary, matrices


@st.cache_resource
def load_benchmark_preprocessing():
    path = OUTPUT_DIR / "benchmark_preprocessing.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    path = OUTPUT_DIR / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_test_population() -> pd.DataFrame:
    """Reconstruct the deterministic test rows for subgroup analysis."""
    dataset_path = ROOT / "diabetes_prediction_dataset.csv"
    if not dataset_path.exists():
        return pd.DataFrame()
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(dataset_path)
    _, test = train_test_split(frame, test_size=0.20, random_state=42, stratify=frame["diabetes"])
    return test.reset_index(drop=True)


def transform_benchmark_frame(frame: pd.DataFrame) -> np.ndarray:
    prep = load_benchmark_preprocessing()
    if prep is None:
        raise FileNotFoundError("benchmark_preprocessing.pkl was not found")
    frame = validate_benchmark(frame)
    cat_cols = ["gender", "hypertension", "heart_disease", "smoking_history"]
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    cat = np.rint(prep["ordinal_encoder"].transform(frame[cat_cols])).astype(np.int64)
    numeric = prep["scaler"].transform(frame[num_cols].to_numpy(dtype=np.float32))
    return np.column_stack([prep["onehot_encoder"].transform(cat), numeric]).astype(np.float32)


@st.cache_resource
def load_model(model_name: str):
    """Load one model only when the user requests it."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(OUTPUT_DIR / "xgboost.json")
        return model
    if model_name == "MLP":
        with (OUTPUT_DIR / "mlp.pkl").open("rb") as file:
            return pickle.load(file)
    if model_name == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        model = TabNetClassifier()
        model.load_model(str(OUTPUT_DIR / "tabnet.zip"))
        return model
    if model_name == "TabPFN":
        with (OUTPUT_DIR / "tabpfn.pkl").open("rb") as file:
            return pickle.load(file)
    if model_name in {"DNN", "Wide and Deep"}:
        from tensorflow import keras

        filename = "dnn.keras" if model_name == "DNN" else "wide_and_deep.keras"
        return keras.models.load_model(OUTPUT_DIR / filename)
    raise ValueError(f"Unknown model: {model_name}")


def model_available(model_name: str) -> bool:
    paths = {
        "XGBoost": OUTPUT_DIR / "xgboost.json",
        "DNN": OUTPUT_DIR / "dnn.keras",
        "MLP": OUTPUT_DIR / "mlp.pkl",
        "TabNet": OUTPUT_DIR / "tabnet.zip",
        "Wide and Deep": OUTPUT_DIR / "wide_and_deep.keras",
        "TabPFN": OUTPUT_DIR / "tabpfn.pkl",
    }
    return paths[model_name].exists()


def predict_probabilities(model_name: str, frame: pd.DataFrame) -> np.ndarray:
    X = transform_benchmark_frame(frame)
    model = load_model(model_name)
    if model_name in {"XGBoost", "MLP", "TabNet", "TabPFN"}:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X, verbose=0).ravel(), dtype=float)


def patient_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{column: values[column] for column in RAW_INPUT_COLUMNS}])


def probability_label(probability: float, threshold: float) -> tuple[str, str]:
    if probability >= threshold:
        return "Above research cutoff", "warning"
    return "Below research cutoff", "info"


def format_probability(probability: float) -> str:
    return probability_text(probability)


def model_agreement(probabilities: dict, threshold: float) -> dict:
    values = np.asarray(list(probabilities.values()), dtype=float)
    votes = values >= threshold
    spread = float(values.max() - values.min()) if len(values) else 0.0
    if spread < 0.10:
        level = "Strong agreement"
    elif spread < 0.25:
        level = "Moderate agreement"
    else:
        level = "Model disagreement"
    return {
        "level": level,
        "spread": spread,
        "positive_votes": int(votes.sum()),
        "total_models": int(len(values)),
        "mean_probability": float(values.mean()) if len(values) else 0.0,
    }


def data_quality_checks(values: dict) -> list[str]:
    issues = []
    if values["age"] < 18:
        issues.append("Age is below 18; this benchmark was not validated as a pediatric model.")
    if values["bmi"] < 15 or values["bmi"] > 50:
        issues.append("BMI is outside the central range represented in many adult screening settings.")
    if values["blood_glucose_level"] >= 200:
        issues.append("Blood glucose is high; verify units, timing, symptoms, and clinical context.")
    if values["HbA1c_level"] >= 6.5:
        issues.append("HbA1c is in a range that warrants professional interpretation and confirmation.")
    if not issues:
        issues.append("No basic input-range warnings were triggered. This is not a clinical clearance.")
    return issues


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight range"
    if bmi < 25:
        return "Healthy-weight range"
    if bmi < 30:
        return "Overweight range"
    return "Obesity range"


def glucose_note(glucose: float) -> str:
    if glucose >= 200:
        return "High reading: discuss promptly with a qualified clinician, especially if symptoms are present."
    if glucose >= 140:
        return "Above the usual fasting reference range; interpretation depends on when the test was taken."
    return "Interpret together with the test timing and clinical context."


def patient_guidance(values: dict, diabetes_prediction: int) -> list[str]:
    guidance = []
    if values["smoking_history"] in {"current", "former", "ever"}:
        guidance.append("If relevant, ask a healthcare professional about evidence-based smoking cessation support.")
    if values["bmi"] >= 30:
        guidance.append("A clinician or dietitian can help create a gradual, sustainable nutrition and activity plan.")
    if values["hypertension"] == "Yes":
        guidance.append("Keep blood-pressure follow-up and prescribed treatment consistent; do not change medication here.")
    if values["hba1c"] >= 7:
        guidance.append("Bring this HbA1c value to a clinician for a personalized glucose-management discussion.")
    if not guidance:
        guidance.append("Continue routine preventive care and periodic screening recommended by your healthcare professional.")
    if diabetes_prediction:
        guidance.append("A model flag is not a diagnosis; confirmatory clinical testing is required.")
    return guidance


def render_prediction_card(model_name: str, probability: float, threshold: float) -> None:
    label, kind = probability_label(probability, threshold)
    getattr(st, kind)(f"{label} · positive-class score {format_probability(probability)}")
    st.caption(f"{model_name} · cutoff {threshold:.2f} · uncalibrated benchmark score, not certainty or diagnosis")


def score_patient(values: dict, selected_model: str, threshold: float, compare_all: bool):
    frame = patient_frame(values)
    probabilities = {selected_model: float(predict_probabilities(selected_model, frame)[0])}
    if compare_all:
        for model_name in MODEL_NAMES:
            if model_name not in probabilities and model_available(model_name):
                probabilities[model_name] = float(predict_probabilities(model_name, frame)[0])
    return frame, probabilities


def render_patient_tab(selected_model: str, threshold: float, compare_all: bool) -> None:
    st.markdown('<div class="section-kicker">Step 1 · Patient inputs</div>', unsafe_allow_html=True)
    st.subheader("Enter patient information")
    st.markdown('<div class="helper-line">Pre-filled values are fictional examples. Replace them with measured values to explore the legacy benchmark.</div>', unsafe_allow_html=True)
    with st.expander("Need help with a field?", icon=":material/help:"):
        st.caption("Use the units shown. HbA1c is a percentage; glucose is mg/dL. If a value is unknown, do not invent one.")
    with st.form("patient_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="patient_gender", persist_state="session", help="Recorded gender category in the benchmark dataset.")
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=45.0, step=1.0, key="patient_age", persist_state="session", help="Age in years.")
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="patient_hypertension", persist_state="session", help="Whether hypertension is recorded.")
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"], key="patient_heart", persist_state="session", help="Whether heart disease is recorded.")
        with right:
            smoking_history = st.selectbox("Smoking history", ["never", "No Info", "current", "former", "ever", "not current"], key="patient_smoking", persist_state="session", help="Smoking category used by the original dataset.")
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="patient_bmi", persist_state="session", help="Body-mass index in kg/m².")
            hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.7, step=0.1, key="patient_hba1c", persist_state="session", help="Longer-term blood-glucose marker.")
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=400.0, value=120.0, step=1.0, key="patient_glucose", persist_state="session", help="Blood glucose value; timing matters clinically.")
        st.caption(f"BMI interpretation: {bmi_category(bmi)} · {glucose_note(glucose)}")
        submitted = st.form_submit_button("Run prediction", type="primary", width="stretch", icon=":material/play_arrow:")

    if not submitted and "benchmark_snapshot" not in st.session_state:
        st.info("Enter the values and select **Run prediction** to score this patient in real time.")
        return

    values = {
        "gender": gender,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hba1c": hba1c,
    }
    if submitted:
        st.session_state.pop("benchmark_snapshot", None)
        try:
            with st.spinner("Running selected benchmark models…"):
                frame, probabilities = score_patient(values, selected_model, threshold, compare_all)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
        st.session_state["benchmark_snapshot"] = dict(values=values, probabilities=probabilities, model=selected_model, threshold=threshold, compare_all=compare_all, generated_at=datetime.now().isoformat(timespec="seconds"))
    snapshot = st.session_state["benchmark_snapshot"]
    if (selected_model, threshold, compare_all) != (snapshot["model"], snapshot["threshold"], snapshot["compare_all"]):
        st.info("Controls changed. Run prediction to apply them; the saved result below keeps its original settings.")
    values, probabilities = snapshot["values"], snapshot["probabilities"]
    selected_model, threshold, compare_all = snapshot["model"], snapshot["threshold"], snapshot["compare_all"]
    frame = patient_frame(values)
    bmi, hba1c, glucose = values["bmi"], values["hba1c"], values["blood_glucose_level"]
    st.caption(f"Saved result · {snapshot['generated_at']}. Form edits require a new submission.")

    selected_probability = probabilities[selected_model]
    prediction = int(selected_probability >= threshold)
    st.session_state["last_patient"] = values
    st.session_state["last_probabilities"] = probabilities
    st.session_state["last_frame"] = frame

    st.space("small")
    st.markdown('<div class="section-kicker">Step 2 · Results</div>', unsafe_allow_html=True)
    st.subheader("Prediction result")
    with st.container(border=True):
        result_col, risk_col, context_col = st.columns([1.2, 1, 1])
        with result_col:
            render_prediction_card(selected_model, selected_probability, threshold)
        with risk_col:
            st.metric("Risk estimate", format_probability(selected_probability), border=True)
            st.metric("BMI", f"{bmi:.1f}", bmi_category(bmi), border=True)
        with context_col:
            st.metric("HbA1c", f"{hba1c:.1f}%", border=True)
            st.metric("Glucose", f"{glucose:.0f} mg/dL", border=True)

    with st.expander("How should I read this?", icon=":material/visibility:"):
        st.write(f"**{selected_model}** estimated a {format_probability(selected_probability)} probability for the positive class. The displayed classification uses a cutoff of **{threshold:.2f}**; it is not a diagnosis or certainty.")

    if compare_all and len(probabilities) > 1:
        st.subheader("Model comparison")
        st.caption("Same inputs, different model architectures trained from the same source. Agreement is not independent clinical evidence or an uncertainty interval.")
        comparison = pd.DataFrame({"Model": list(probabilities), "Estimated diabetes probability": [probabilities[name] for name in probabilities]}).sort_values("Estimated diabetes probability", ascending=False)
        st.bar_chart(comparison.set_index("Model"), y="Estimated diabetes probability", y_label="Probability", height=280)
        display_comparison = comparison.copy()
        display_comparison["Estimated diabetes probability"] = display_comparison["Estimated diabetes probability"].map(format_probability)
        st.dataframe(display_comparison, hide_index=True, width="stretch")
        agreement = model_agreement(probabilities, threshold)
        st.info(
            f"{agreement['level']}: {agreement['positive_votes']}/{agreement['total_models']} models cross the threshold; "
            f"the probability spread is {format_probability(agreement['spread'])}. Disagreement is a reason to seek professional review, not to average away uncertainty."
        )

    with st.expander("Input quality and model limitations"):
        for issue in data_quality_checks(values):
            st.write(f"- {issue}")
        st.caption("The model sees only eight structured inputs. It does not know symptoms, medications, family history, pregnancy status, or laboratory timing unless represented by these fields.")

    risk = {"status": "Not assessed: no kidney outcome labels or validated kidney model."}
    with st.expander("Why is there no kidney-disease prediction?"):
        st.write("The dataset has no kidney outcome label. The former hand-written risk points were not validated and are no longer displayed as a patient result.")

    st.subheader("Practical next steps")
    for item in patient_guidance(values, prediction):
        st.write(f"- {item}")

    summary = {
        "generated_at": snapshot["generated_at"],
        "selected_model": selected_model,
        "decision_threshold": threshold,
        "patient_inputs": {key: value for key, value in values.items() if key != "hba1c"},
        "model_probabilities": probabilities,
        "selected_prediction": prediction,
        "kidney_risk_flag": risk,
        "disclaimer": "Research/education prototype; not a diagnosis or emergency-care tool.",
    }
    st.download_button("Download patient summary (JSON)", data=json.dumps(summary, indent=2, default=str), file_name="patient_risk_summary.json", mime="application/json", on_click="ignore")


def render_what_if_tab(selected_model: str) -> None:
    st.markdown('<div class="section-kicker">Explore sensitivity</div>', unsafe_allow_html=True)
    st.subheader("What-if sensitivity simulator")
    st.markdown('<div class="helper-line">Change one factor at a time to see how the estimate moves.</div>', unsafe_allow_html=True)
    values = st.session_state.get("last_patient")
    if not values:
        st.info("Run one patient prediction first, then return here to explore scenarios.")
        return
    available_models = [name for name in MODEL_NAMES if model_available(name)]
    default_index = available_models.index(selected_model) if selected_model in available_models else 0
    scenario_model = st.selectbox("Model for simulation", available_models, index=default_index, key="scenario_model")
    deltas = {"HbA1c_level": st.slider("HbA1c change", -2.0, 2.0, 0.0, 0.1), "blood_glucose_level": st.slider("Blood glucose change", -50.0, 50.0, 0.0, 1.0), "bmi": st.slider("BMI change", -5.0, 5.0, 0.0, 0.1)}
    rows = []
    base = {key: values[key] for key in RAW_INPUT_COLUMNS}
    scenarios = [("Current inputs", base)]
    for feature, delta in deltas.items():
        if delta == 0:
            continue
        changed = base.copy()
        changed[feature] = float(changed[feature]) + float(delta)
        scenarios.append((f"Change {feature} by {delta:+g}", changed))
    for label, scenario in scenarios:
        try:
            probability = float(predict_probabilities(scenario_model, pd.DataFrame([scenario]))[0])
        except ValueError as exc:
            st.warning(f"Scenario skipped: {exc}")
            continue
        rows.append({"Scenario": label, "Estimated probability": probability, "Display": format_probability(probability)})
    if not rows:
        return
    scenario_frame = pd.DataFrame(rows)
    st.bar_chart(scenario_frame.set_index("Scenario"), y="Estimated probability", y_label="Probability", height=300)
    st.dataframe(scenario_frame[["Scenario", "Display"]].rename(columns={"Display": "Estimated diabetes probability"}), hide_index=True, width="stretch")
    st.warning("Changing a single variable in a model simulation does not prove that changing that variable will cause the same change in a real patient.")


def render_history_tab() -> None:
    st.markdown('<div class="section-kicker">Monitor over time</div>', unsafe_allow_html=True)
    st.subheader("Personal trend tracker")
    st.markdown('<div class="helper-line">Session-only trend view for HbA1c, glucose, and BMI.</div>', unsafe_allow_html=True)
    with st.form("history_form", clear_on_submit=True):
        reading_date = st.date_input("Reading date", value=date.today())
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("Blood glucose (mg/dL)", 50.0, 400.0, 120.0, 1.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
        add_reading = st.form_submit_button("Add reading", type="primary")
    if add_reading:
        history = st.session_state.setdefault("health_history", [])
        history.append({"date": str(reading_date), "HbA1c": hba1c, "Blood glucose": glucose, "BMI": bmi})
        st.session_state["health_history"] = history
        st.success("Reading added to this session.")
    history = pd.DataFrame(st.session_state.get("health_history", []))
    if history.empty:
        st.info("Add at least one reading to build a trend chart.")
        return
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")
    for column, unit in [("HbA1c", "%"), ("Blood glucose", "mg/dL"), ("BMI", "kg/m²")]:
        st.write(f"{column} ({unit})")
        st.line_chart(history.set_index("date")[[column]], height=180, y_label=unit)
    st.dataframe(history, hide_index=True, width="stretch")
    st.download_button("Download session trend CSV", history.to_csv(index=False), "health_trend.csv", "text/csv", on_click="ignore")


def render_research_lab(results: pd.DataFrame, summary: dict) -> None:
    st.warning("Legacy benchmark audit: exact duplicate rows were not grouped before splitting; internal validation followed SMOTE. Do not cite these as leakage-controlled clinical results. Use Evidence for the separate screening v2 experiment.")
    st.markdown('<div class="section-kicker">For evaluation and discussion</div>', unsafe_allow_html=True)
    st.subheader("Validation, calibration, and subgroup analysis")
    st.markdown('<div class="helper-line">Audit saved test predictions without running the models again.</div>', unsafe_allow_html=True)
    predictions = load_test_predictions()
    population = load_test_population()
    if predictions.empty:
        st.warning("Saved test predictions are unavailable. Run the benchmark first.")
        return
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score

    model = st.selectbox("Model to audit", [name for name in MODEL_NAMES if f"{name}_probability" in predictions.columns], key="audit_model")
    probability_column = f"{model}_probability"
    threshold = st.slider("Audit threshold", 0.05, 0.95, 0.50, 0.01, key="audit_threshold")
    y_true = predictions["y_true"].to_numpy()
    probabilities = predictions[probability_column].to_numpy()
    labels = (probabilities >= threshold).astype(int)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{accuracy_score(y_true, labels) * 100:.2f}%")
    metric_cols[1].metric("Precision", f"{precision_score(y_true, labels, zero_division=0) * 100:.2f}%")
    metric_cols[2].metric("Recall", f"{recall_score(y_true, labels, zero_division=0) * 100:.2f}%")
    metric_cols[3].metric("F1", f"{f1_score(y_true, labels, zero_division=0) * 100:.2f}%")
    metric_cols[4].metric("Brier score", f"{brier_score_loss(y_true, probabilities):.4f}")

    thresholds = np.linspace(0.05, 0.95, 19)
    threshold_rows = []
    for cutoff in thresholds:
        cutoff_labels = (probabilities >= cutoff).astype(int)
        threshold_rows.append({"Threshold": cutoff, "Precision": precision_score(y_true, cutoff_labels, zero_division=0), "Recall": recall_score(y_true, cutoff_labels, zero_division=0), "F1": f1_score(y_true, cutoff_labels, zero_division=0)})
    threshold_frame = pd.DataFrame(threshold_rows).set_index("Threshold")
    st.markdown("### Threshold trade-off")
    st.line_chart(threshold_frame, height=300)
    st.caption("Lower thresholds generally catch more positives but create more false positives. The best cutoff depends on the intended screening objective and must be chosen on validation data.")

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    calibration_frame = pd.DataFrame({"Predicted probability": predicted, "Observed frequency": observed}).set_index("Predicted probability")
    st.markdown("### Reliability check")
    st.line_chart(calibration_frame, height=300)
    st.caption("A calibrated model would track the diagonal relationship between predicted probability and observed frequency. The benchmark has not undergone formal post-hoc calibration.")

    if not population.empty and len(population) == len(predictions):
        audit = population.copy()
        audit["probability"] = probabilities
        audit["prediction"] = labels
        audit["age_group"] = pd.cut(audit["age"], bins=[-1, 30, 45, 60, 200], labels=["≤30", ">30–45", ">45–60", ">60"])
        st.markdown("### Subgroup performance")
        subgroup = st.selectbox("Subgroup variable", ["gender", "age_group", "hypertension", "smoking_history"], key="subgroup_variable")
        rows = []
        for group, part in audit.groupby(subgroup, observed=False):
            if len(part) == 0:
                continue
            rows.append({"Group": str(group), "Rows": len(part), "Observed positive rate": part["diabetes"].mean(), "Recall": recall_score(part["diabetes"], part["prediction"], zero_division=0), "Precision": precision_score(part["diabetes"], part["prediction"], zero_division=0)})
        subgroup_frame = pd.DataFrame(rows)
        for column in ["Observed positive rate", "Recall", "Precision"]:
            subgroup_frame[column] = subgroup_frame[column].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(subgroup_frame, hide_index=True, width="stretch")
        st.caption("Subgroup comparisons are descriptive checks on this dataset, not proof of fairness or clinical equivalence.")

    importance_path = OUTPUT_DIR / "feature_importance.json"
    if importance_path.exists():
        all_importance = json.loads(importance_path.read_text(encoding="utf-8"))
        if model in all_importance:
            st.markdown("### Saved permutation importance")
            importance_frame = pd.Series(all_importance[model], name="Accuracy drop").sort_values(ascending=False).to_frame()
            st.bar_chart(importance_frame, height=320)
def render_batch_tab(selected_model: str, threshold: float) -> None:
    st.markdown('<div class="section-kicker">Research workflow</div>', unsafe_allow_html=True)
    st.subheader("Batch scoring")
    st.markdown('<div class="helper-line">Score many rows at once using the selected model and cutoff.</div>', unsafe_allow_html=True)
    with st.expander("Required CSV columns", icon=":material/table_view:"):
        st.code(",".join(RAW_INPUT_COLUMNS), language="text")
        st.caption("Yes/No and 1/0 are accepted for hypertension and heart disease. Missing/invalid values are rejected; they are not silently converted to low risk.")
    st.download_button("Download fictional CSV example", pd.DataFrame([dict(gender="Female",hypertension=0,heart_disease=0,smoking_history="never",age=45,bmi=25,HbA1c_level=5.7,blood_glucose_level=120)])[RAW_INPUT_COLUMNS].to_csv(index=False), "fictional_example.csv", "text/csv", on_click="ignore")
    uploaded = st.file_uploader("Upload patient CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        st.info("No file is uploaded. The CSV must contain the exact columns shown above.")
        return
    try:
        fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest()
        identity = (fingerprint, selected_model, threshold)
        frame = pd.read_csv(uploaded)
        if len(frame) > 5000:
            st.error("Limit each interactive batch to 5,000 rows. Use an offline job for larger files.")
            return
        validate_benchmark(frame)
        st.write(f"Loaded {len(frame):,} rows.")
        st.dataframe(frame.head(10), width="stretch")
        missing = [column for column in RAW_INPUT_COLUMNS if column not in frame.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return
        if st.button(f"Score CSV with {selected_model}", type="primary", key="score_batch"):
            with st.spinner("Scoring records..."):
                probabilities = predict_probabilities(selected_model, frame)
            scored = frame.copy()
            scored["model"] = selected_model
            scored["diabetes_probability"] = probabilities
            scored["predicted_diabetes"] = (probabilities >= threshold).astype(int)
            scored["decision_threshold"] = threshold
            st.session_state["batch_snapshot"] = (identity, scored)
        cached = st.session_state.get("batch_snapshot")
        if cached and cached[0] == identity:
            scored = cached[1]
            st.success(f"Scored {len(scored):,} records.")
            st.dataframe(scored.head(50), width="stretch")
            buffer = io.StringIO()
            scored.to_csv(buffer, index=False)
            st.download_button("Download scored CSV", data=buffer.getvalue(), file_name="scored_diabetes_predictions.csv", mime="text/csv", key="download_scored_csv", on_click="ignore")
    except Exception as exc:
        st.error(f"Could not read or score this file: {exc}")


def render_model_lab(results: pd.DataFrame, summary: dict, matrices: dict) -> None:
    st.warning("Original six-model benchmark, preserved for comparison. Duplicate/synthetic-validation leakage risks remain in this old training run. These metrics are not directly comparable with the new five-input screening model.")
    st.markdown('<div class="section-kicker">Benchmark overview</div>', unsafe_allow_html=True)
    st.subheader("Model performance and transparency")
    st.markdown('<div class="helper-line">Compare saved metrics and understand where the models differ.</div>', unsafe_allow_html=True)
    if results.empty:
        st.warning("Benchmark results are not available. Run run_benchmark.py first.")
        return
    display_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "training_rows"]
    table = results[display_columns].sort_values("roc_auc", ascending=False).copy()
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        table[column] = table[column].map(lambda value: f"{value * 100:.2f}%")
    table = table.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "training_rows": "Training rows"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Metrics are from the saved benchmark. TabPFN used a deliberately small training context for CPU feasibility; all models were evaluated on the saved test split.")
    chart = results.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].sort_values("roc_auc")
    st.bar_chart(chart, height=360)
    image_col, roc_col = st.columns(2)
    with image_col:
        comparison_image = OUTPUT_DIR / "model_comparison.png"
        if comparison_image.exists():
            st.image(str(comparison_image), caption="Saved benchmark comparison")
    with roc_col:
        roc_image = OUTPUT_DIR / "roc_curves.png"
        if roc_image.exists():
            st.image(str(roc_image), caption="Saved ROC curves")
    st.subheader("Confusion matrix counts")
    matrix_frame = pd.DataFrame(matrices).T.rename(columns={"tn": "True negative", "fp": "False positive", "fn": "False negative", "tp": "True positive"})
    st.dataframe(matrix_frame, width="stretch")
    st.info("Feature importance is an association measure from permutation testing. It should not be interpreted as causation.")


def render_guidance(summary: dict) -> None:
    st.markdown('<div class="section-kicker">Use responsibly</div>', unsafe_allow_html=True)
    st.subheader("Patient safety and responsible use")
    st.markdown("<div class=\"notice\"><strong>Important:</strong> This dashboard is a final-year research prototype. It estimates risk from a small set of self-reported or measured inputs; it does not diagnose diabetes, kidney disease, or any other condition.</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"safe\"><strong>Use it safely:</strong> Verify values, consider the timing and conditions of laboratory tests, discuss unexpected results with a qualified clinician, and seek urgent local medical help for severe symptoms or an emergency.</div>", unsafe_allow_html=True)
    st.markdown("### What each output means")
    st.write("- **Probability:** the model's estimated probability for the positive class, not a personal certainty.")
    st.write("- **Decision threshold:** a configurable cutoff for a demonstration classification; changing it changes precision and recall.")
    st.write("- **Kidney disease:** not assessed. The source has no kidney outcome label; the old heuristic flag was withdrawn from patient results.")
    st.write("- **Model comparison:** useful for understanding disagreement and uncertainty, not for choosing a medical treatment.")
    st.markdown("### Dataset context")
    if summary:
        st.write(f"The benchmark dataset contains {summary.get('rows', 0):,} records and an observed positive rate of {summary.get('positive_rate_percent', 0):.2f}%. The test set was kept separate from SMOTENC balancing.")
    st.markdown("### Helpful follow-up questions for a clinician")
    st.write("- Should this result be confirmed with a repeat or laboratory test?")
    st.write("- What do my blood-glucose and HbA1c values mean given when they were taken?")
    st.write("- Do my blood pressure, medications, family history, or symptoms change the interpretation?")


def main() -> None:
    inject_css()
    results, summary, matrices = load_results()
    available = [name for name in MODEL_NAMES if model_available(name)]
    with st.sidebar:
        st.markdown("### DiaLens")
        page = st.radio("Navigate", ["Screening", "Lab review", "Evidence", "Learn & present", "Model comparison", "What-if", "Trends", "Batch", "Benchmark results", "Benchmark audit", "Safety"], key="workspace_page")
        selected_model, threshold, compare_all = "DNN", .5, False
        if page in ["Model comparison", "What-if", "Batch"] and available:
            st.divider()
            default_model = "DNN" if "DNN" in available else available[0]
            selected_model = st.selectbox("Benchmark model", available, index=available.index(default_model), key="selected_benchmark", persist_state="session")
            threshold = st.slider("Research cutoff", .10, .90, .50, .01, key="benchmark_threshold", persist_state="session", help="This control changes only the legacy benchmark demonstration, not the frozen screening model.")
            compare_all = st.checkbox("Compare all six", value=False, key="compare_benchmark", persist_state="session", help="Loads additional models; first use can take longer.")
        st.divider()
        st.caption("Local research workspace · no patient names needed · only the selected page runs.")
        if st.button("Clear session", icon=":material/restart_alt:", help="Clears this tab's inputs, results and trends. Downloaded files are not deleted."):
            for key in list(st.session_state): del st.session_state[key]
            st.rerun()
    st.markdown("<div class=\"hero\"><h1>DiaLens</h1><p>A Deep Learning Framework for Diabetes Screening and Laboratory Decision Support</p><div class=\"hero-meta\"><span class=\"hero-chip\">Screening DNN</span><span class=\"hero-chip\">Laboratory review</span><span class=\"hero-chip\">External evaluation</span></div></div>", unsafe_allow_html=True)
    st.caption("Research decision support · not clinically validated · not for autonomous diagnosis or emergency care")
    if page == "Screening":
        clinical_workspace.screening()
    elif page == "Lab review":
        clinical_workspace.lab_review()
    elif page == "Evidence":
        clinical_workspace.evidence_page()
    elif page == "Learn & present":
        clinical_workspace.learn()
    elif page == "Model comparison" and available:
        st.info("Legacy eight-input experiment. For the new non-laboratory DNN use Screening; these scores have not been calibrated.")
        render_patient_tab(selected_model, threshold, compare_all)
    elif page == "What-if" and available:
        render_what_if_tab(selected_model)
    elif page == "Trends":
        render_history_tab()
    elif page == "Batch" and available:
        render_batch_tab(selected_model, threshold)
    elif page == "Benchmark results":
        render_model_lab(results, summary, matrices)
    elif page == "Benchmark audit":
        render_research_lab(results, summary)
    elif page == "Safety":
        render_guidance(summary)
    else:
        st.info("Benchmark model artifacts are unavailable; the screening and laboratory pages remain independent.")


if __name__ == "__main__":
    main()
