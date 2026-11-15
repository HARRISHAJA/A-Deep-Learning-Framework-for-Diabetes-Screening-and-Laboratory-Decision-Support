"""Independent screening, lab review, evidence and viva pages."""
from __future__ import annotations
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from clinical_core import ADA_SOURCE, interpret_labs, probability_text, screening_frame

ARTIFACTS = Path(__file__).resolve().parent / "clinical_outputs" / "v2"


def summary():
    path = ARTIFACTS / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource(max_entries=2)
def bundle(version_stamp):
    from tensorflow import keras
    with (ARTIFACTS / "preprocessing.pkl").open("rb") as handle:
        prep = pickle.load(handle)
    return keras.models.load_model(ARTIFACTS / "screening_dnn.keras"), prep


def score(values):
    evidence = summary()
    if not evidence:
        raise FileNotFoundError("Run python train_screening_v2.py first.")
    frame = screening_frame(pd.DataFrame([values]))
    model, prep = bundle(evidence["created_at"])
    features = np.column_stack([prep["encoder"].transform(frame[prep["categorical"]]), prep["scaler"].transform(frame[prep["numeric"]])]).astype(np.float32)
    raw = np.asarray(model(features, training=False)).ravel()
    clipped = np.clip(raw, 1e-7, 1-1e-7)
    probability = float(prep["calibrator"].predict_proba(np.log(clipped/(1-clipped)).reshape(-1,1))[0,1])
    return probability, evidence


def export_snapshot(snapshot, name):
    st.caption(f"Saved result · {snapshot['generated_at']} · changes require a new submission.")
    with st.expander("Submitted inputs and audit trail"):
        st.json(snapshot)
    text = [f"# {name}", "", f"Generated: {snapshot['generated_at']}", "", "Research decision-support output — not a diagnosis.", ""]
    for key, value in snapshot.items():
        if key != "generated_at":
            text += [f"## {key.replace('_', ' ').title()}", "", "```json", json.dumps(value, indent=2), "```", ""]
    left, right = st.columns(2)
    left.download_button("Download readable summary", "\n".join(text), f"{name.lower().replace(' ', '_')}.md", "text/markdown", key=f"{name}_md", on_click="ignore")
    right.download_button("Download audit JSON", json.dumps(snapshot, indent=2), f"{name.lower().replace(' ', '_')}.json", "application/json", key=f"{name}_json", on_click="ignore")


def screening():
    st.subheader("Screen without laboratory inputs")
    st.caption("Five inputs · deep neural network · no HbA1c or glucose required")
    evidence = summary()
    if not evidence:
        st.info("The audited screening experiment is not ready. Run `python train_screening_v2.py`.")
        return
    with st.expander("What does this estimate?", icon=":material/help:"):
        st.write("Association with the dataset's diabetes label, using age, BMI, recorded sex/gender, hypertension and smoking history. This is not a forecast of developing diabetes in a specified number of years. A low score does not rule out diabetes.")
        st.caption("For adult research demonstrations. Not for pregnancy, children, acute symptoms or managing already diagnosed diabetes. No patient names or identifiers are needed.")
    if st.button("Load fictional demo", icon=":material/science:", key="screen_demo"):
        st.session_state.update(s_age=45.0, s_bmi=25.0, s_gender="Female", s_hypertension="No", s_smoking="never", s_exclusion="None reported")
        st.session_state.pop("screen_snapshot", None)
        st.info("Fictional values loaded. Select Estimate to run the model.")
    with st.form("screening_v2"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age (years)", 18.0, 80.0, value=None, step=1.0, key="s_age", persist_state="session")
            gender = st.selectbox("Recorded sex/gender", ["Female", "Male", "Other"], index=None, key="s_gender", persist_state="session", help="Source dataset categories; these are not interchangeable with a complete clinical sex/gender history.")
            hypertension = st.selectbox("Known hypertension", ["No", "Yes", "Unknown"], index=None, key="s_hypertension", persist_state="session")
        with right:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 100.0, value=None, step=.1, key="s_bmi", persist_state="session")
            smoking = st.selectbox("Smoking history", ["never", "current", "former", "ever", "not current", "No Info"], index=None, key="s_smoking", persist_state="session")
            exclusion = st.selectbox("Context check", ["None reported", "Pregnant / possibly pregnant", "Already diagnosed with diabetes", "Acute symptoms / unwell"], key="s_exclusion", persist_state="session")
        submitted = st.form_submit_button("Estimate screening score", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("screen_snapshot", None)
        if exclusion != "None reported":
            st.warning("This screening pathway is not appropriate for the selected context. Use clinician assessment; acute or severe symptoms may require urgent care.")
        elif any(v is None for v in [age, gender, hypertension, bmi, smoking]) or hypertension == "Unknown":
            st.error("Complete all five inputs. Unknown hypertension cannot be treated as No.")
        else:
            values = dict(age=age, gender=gender, hypertension=hypertension, bmi=bmi, smoking_history=smoking)
            try:
                with st.spinner("Running the screening DNN…"):
                    p, evidence = score(values)
                st.session_state["screen_snapshot"] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model": evidence["version"], "model_created_at": evidence["created_at"], "inputs": values,
                    "probability": p, "threshold": evidence["threshold"], "flagged": p >= evidence["threshold"],
                    "interpretation": "Research screening score; not a diagnosis or future-risk forecast."}
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
    snapshot = st.session_state.get("screen_snapshot")
    if snapshot:
        with st.container(border=True):
            left, right = st.columns([1,2])
            left.metric("Screening score", probability_text(snapshot["probability"]))
            left.caption(f"Frozen research cutoff: {probability_text(snapshot['threshold'])}")
            if snapshot["flagged"]:
                right.warning("Above the screening cutoff. Discuss whether confirmatory testing is appropriate; this is not a diabetes diagnosis.")
            else:
                right.info("Below the screening cutoff. This does not rule out diabetes or override symptoms and laboratory results.")
        if snapshot["inputs"]["gender"] == "Other":
            st.warning("This category has very few source records and no corresponding NHANES category; evidence is particularly limited.")
        st.caption("Calibration was fitted on a separate local partition. That does not establish calibrated probabilities for every population.")
        export_snapshot(snapshot, "Screening summary")


def lab_review():
    st.subheader("Review laboratory evidence")
    st.caption("Transparent rules · optional measurements · explicit missing-data handling")
    st.info("Use laboratory plasma glucose in mg/dL and a standardized laboratory HbA1c in %. Home-meter readings are not diagnostic laboratory confirmation.")
    with st.form("lab_review"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age (years)", 0.0, 120.0, value=45.0, key="l_age", persist_state="session")
            a1c = st.number_input("HbA1c (%) — optional", 3.0, 20.0, value=None, step=.1, key="l_a1c", persist_state="session")
            glucose = st.number_input("Plasma glucose (mg/dL) — optional", 20.0, 700.0, value=None, step=1.0, key="l_glucose", persist_state="session")
            repeat = st.number_input("Separate repeat HbA1c (%) — optional", 3.0, 20.0, value=None, step=.1, key="l_repeat", persist_state="session", help="Enter the actual result from a separate specimen, not a duplicate entry of the first test.")
        with right:
            timing = st.selectbox("Glucose sample timing", ["Unknown", "Fasting", "Random"], key="l_timing", persist_state="session")
            hours = st.number_input("Hours without caloric intake — if fasting", 0.0, 24.0, value=None, step=.5, key="l_hours", persist_state="session")
            symptoms = st.selectbox("Classic high-glucose symptoms", ["Unknown", "Yes", "No"], key="l_symptoms", persist_state="session", help="Such as frequent urination, excessive thirst or unexplained weight loss. A clinician must verify.")
            pregnancy = st.selectbox("Pregnancy context", ["No / not applicable", "Yes / possible", "Unknown"], key="l_pregnancy", persist_state="session")
        interference = st.checkbox("HbA1c may be unreliable (for example, altered red-cell turnover or an interfering condition)", key="l_interference", persist_state="session")
        submitted = st.form_submit_button("Review evidence", type="primary", width="stretch")
    if submitted:
        values = dict(age=age,hba1c=a1c,glucose=glucose,repeat_hba1c=repeat,timing=timing,fasting_hours=hours,symptoms=symptoms,pregnancy=pregnancy,a1c_interference=interference)
        st.session_state.pop("lab_snapshot", None)
        try:
            result = interpret_labs(**values)
            st.session_state["lab_snapshot"] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rules_version": "ADA-2026-support-v2", "inputs": values, "result": result}
        except ValueError as exc:
            st.error(str(exc))
    snapshot = st.session_state.get("lab_snapshot")
    if snapshot:
        result = snapshot["result"]
        getattr(st, result["kind"])(result["status"])
        for reason in result["reasons"]: st.write(f"- {reason}")
        st.write(result["next_step"])
        export_snapshot(snapshot, "Laboratory review")
    st.caption(f"[ADA Standards of Care 2026 — diagnostic criteria]({ADA_SOURCE}). This limited implementation does not cover OGTT, gestational diabetes, diabetes type or every interference.")


def evidence_page():
    st.subheader("Evidence, not just accuracy")
    evidence = summary()
    if not evidence:
        st.info("Train the screening v2 experiment to populate this page.")
        return
    st.caption("Screening DNN v2 · the model and cutoff were frozen before either test evaluation")
    choice = st.radio("Evaluation population", ["Local held-out test", "NHANES external cohort"], horizontal=True, key="evidence_cohort")
    m = evidence["local_test" if choice.startswith("Local") else "nhanes_external"]
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Sensitivity", "Specificity", "Precision (PPV)", "ROC-AUC"], ["sensitivity", "specificity", "precision", "roc_auc"]):
        col.metric(label, f"{m[key]:.1%}")
    st.caption(f"{m['rows']:,} records · observed positive rate {m['positive_rate']:.1%} · cutoff {m['threshold']:.4f}. Primary metrics here are unweighted sample results, not population estimates.")
    with st.container(border=True):
        st.write("What happened at this cutoff?")
        st.dataframe(pd.DataFrame([[int(m['tn']),int(m['fp'])],[int(m['fn']),int(m['tp'])]], index=["Reference negative","Reference positive"], columns=["Model below cutoff","Model flagged"]), width="stretch")
        st.caption(f"{int(m['fn']):,} reference-positive records were missed; {int(m['fp']):,} reference-negative records were flagged. Both matter when discussing usefulness.")
    with st.expander("Calibration: do scores match observed frequencies?", expanded=True):
        import altair as alt
        curves = pd.read_csv(ARTIFACTS / "reliability.csv")
        curves = curves[curves.cohort.eq("Local test" if choice.startswith("Local") else "NHANES external")]
        chart = alt.Chart(curves).mark_line(point=True).encode(x=alt.X("mean_score:Q", title="Mean model score", scale=alt.Scale(domain=[0,1])), y=alt.Y("observed_rate:Q", title="Observed positive fraction", scale=alt.Scale(domain=[0,1])), tooltip=["mean_score","observed_rate"])
        diagonal = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(strokeDash=[5,5],color="#7B8790").encode(x="x:Q",y="y:Q")
        st.altair_chart((chart+diagonal).properties(height=270), width="stretch")
        st.caption(f"Brier score: {m['brier']:.4f} (lower is better). The dashed diagonal is ideal calibration. Calibration can fail when the population changes.")
    with st.expander("Dataset flow and leakage controls"):
        st.write(f"Source: {evidence['source_rows']:,} rows; {evidence['duplicate_rows_removed']:,} exact duplicates removed; {evidence['adult_rows']:,} adult records retained.")
        st.dataframe(pd.DataFrame(evidence["splits"]).T.rename_axis("Partition"), width="stretch")
        st.write("Identical five-feature profiles stay in one partition. Preprocessing uses training data only. Separate partitions handle early stopping, probability calibration and cutoff selection. No SMOTE in v2.")
        st.dataframe(pd.DataFrame(evidence["external_flow"]), hide_index=True, width="stretch")
        st.caption("NHANES is NOT merged into training. Negative reference labels require both available lab markers below the diabetes cutoffs and no self-reported diabetes. Known diabetes remains included.")
    with st.expander("Threshold selection and model card"):
        thresholds = pd.read_csv(ARTIFACTS / "threshold_selection.csv")
        st.line_chart(thresholds.set_index("threshold")[["sensitivity","specificity"]], height=240)
        st.caption("This curve uses only the cutoff-selection partition. The frozen cutoff is the highest value attaining at least 90% sensitivity there; test sensitivity is not guaranteed to be 90%.")
        st.json({k:evidence[k] for k in ["version","created_at","seed","trainable_parameters","epochs_completed","threshold","versions","source_sha256","local_brier_before_calibration","local_prevalence_baseline_brier","nhanes_weighted_point_estimates"]})
    with st.expander("Limitations you should disclose", expanded=True):
        for item in evidence["limitations"]: st.write(f"- {item}")
    st.download_button("Download evidence report", (ARTIFACTS / "evidence_report.md").read_text(encoding="utf-8"), "screening_evidence_report.md", "text/markdown", on_click="ignore")


def learn():
    st.subheader("Understand it. Demonstrate it. Defend it.")
    st.caption("A practical guide for your project review — accurate answers, not inflated claims")
    for question, answer in [
        ("What exactly did we build?", "A deep-learning screening experiment plus a transparent laboratory-rule support layer, with a separate six-model research benchmark. The screening DNN uses five non-laboratory inputs; the benchmark uses eight including HbA1c and glucose. These are different tasks, so their metrics are not directly comparable."),
        ("Which two layers should I explain?", "1. Screening DNN: learns statistical associations from examples. 2. Laboratory evidence engine: applies explicit reference thresholds with timing, symptoms and missing-data handling. The second layer is intentionally not an opaque trained model."),
        ("Is XGBoost a neural network?", "No. XGBoost is a gradient-boosted decision-tree baseline. DNN, MLP and Wide & Deep are neural networks; TabNet is an attentive tabular network; TabPFN uses a pretrained transformer. A deep-learning project can compare against a tree baseline to test whether deep learning adds value. XGBoost is optional in the dashboard, not the main screening model."),
        ("What changed in the evaluation?", "v2 removes exact duplicates and separates identical input profiles across train, early-stop, calibration, cutoff-selection and test partitions. The previous screening cutoff was chosen on test labels, so its threshold-dependent results were optimistic for model selection and should not be cited as an untouched evaluation."),
        ("Why not combine NHANES with the local data?", "Keeping it separate gives a cross-dataset stress test. Merging it and then calling those same people external validation would be misleading. The outcome and smoking definitions still differ, so this is external research evaluation, not proof of hospital readiness."),
        ("What do sensitivity, specificity and precision mean?", "Sensitivity: fraction of reference positives flagged. Specificity: fraction of reference negatives not flagged. Precision: fraction of flags that are reference positives. A model can catch nearly all positives while generating many false alarms. Accuracy alone hides this trade-off."),
        ("Why can a score be close to 0% or 100%?", "A sigmoid or classifier can return extreme scores. The old one-decimal display also rounded small values to 0.0%. The new display uses bounds such as <0.01%, and never presents the complement as certainty of no diabetes. Calibration checks scores against observed frequencies; it does not eliminate individual uncertainty."),
        ("What is a decision threshold?", "A cutoff applied to the score, not a diagnostic laboratory threshold. Score ≥ cutoff becomes a screening flag. v2 selects its cutoff on a separate partition for a predeclared 90% sensitivity target, then freezes it for evaluation. Lower cutoffs generally flag more records and create more false positives."),
        ("Does this predict future diabetes?", "No. These data are cross-sectional and do not provide time-to-onset outcomes. We estimate association with a current reference label. A future-risk model needs longitudinal follow-up and a defined prediction horizon."),
        ("Can I claim clinical validation?", "No. We have software tests and a retrospective external research evaluation. Prospective clinical validation requires a defined intended-use population, reliable reference standard, ethics/data governance, independent site testing and assessment of clinical benefit and harm. Neither many features nor high accuracy supplies those."),
        ("How do I demonstrate the project?", "Start with Screening → Load fictional demo → Estimate. Open Lab review with no results to show missing-data handling, then try HbA1c 6.8% and fasting glucose 110 mg/dL with 8 hours fasting to show discordance. Open Evidence to explain missed cases and false alarms. Use Model comparison for the six-model research experiment. These are fictional cases, not patient advice."),
        ("Where do data go?", "Interactive inputs and trends live in the current Streamlit session, not a patient database. Downloaded reports stay wherever you save them. Clear session removes this tab's stored inputs/results, not exported files. The local dashboard has no authentication or production clinical security layer; do not expose it as a public patient service.")]:
        with st.expander(question): st.write(answer)
