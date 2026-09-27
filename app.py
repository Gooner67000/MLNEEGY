"""Step 9: Streamlit web app.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import RAW_FEATURES, TYPE_CODES, add_features, load_model, risk_level

ROOT = Path(__file__).parent

st.set_page_config(page_title="Predictive Maintenance System", page_icon="🔧", layout="wide")


@st.cache_resource  # load the model once, not on every interaction
def get_model():
    return load_model()


model, feature_names, threshold = get_model()

st.title("🔧 Predictive Maintenance Alert System")
st.markdown("Failure-risk prediction for industrial equipment from sensor readings.")

single, batch_tab, about = st.tabs(["Single machine", "Batch scoring", "Model performance"])

# ------------------------------------------------------------- single machine
with single:
    st.sidebar.header("Equipment Sensor Data")
    type_label = st.sidebar.selectbox("Product quality type", ["L (low)", "M (medium)", "H (high)"])
    raw = {
        "Type": TYPE_CODES[type_label[0]],
        "Air temperature K": st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1),
        "Process temperature K": st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1),
        "Rotational speed rpm": st.sidebar.slider("Rotational Speed (rpm)", 1150, 2900, 1500),
        "Torque Nm": st.sidebar.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.1),
        "Tool wear min": st.sidebar.slider("Tool Wear (min)", 0, 260, 100),
    }
    input_df = add_features(pd.DataFrame([raw]))[feature_names]
    prob = float(model.predict_proba(input_df)[0][1])
    level = risk_level(prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure Probability", f"{prob * 100:.1f}%")
    c2.metric("Alert threshold", f"{threshold * 100:.1f}%",
              help="Chosen during training so the model catches at least 85% of failures.")
    with c3:
        if level == "Low":
            st.success("✅ Low Risk - Normal operation")
        elif level == "Medium":
            st.warning("⚠️ Medium Risk - Schedule maintenance soon")
        else:
            st.error("🚨 High Risk - Schedule urgent maintenance")
    if prob >= threshold:
        st.info("🔔 Above the alert threshold: this machine would be flagged for inspection.")

    st.subheader("Current Equipment Status")
    status = input_df.T.reset_index()
    status.columns = ["Feature", "Value"]
    status["Value"] = status["Value"].astype(object)
    status.loc[status["Feature"] == "Type", "Value"] = type_label
    st.table(status)

    st.subheader("Maintenance Recommendations")
    if level == "Low":
        st.write("✅ Continue regular monitoring. No action needed.")
    elif level == "Medium":
        st.write("⚠️ Schedule preventive maintenance within 2 weeks.")
    else:
        st.write("🚨 URGENT: Schedule maintenance immediately. Risk of critical failure.")

# -------------------------------------------------------------- batch scoring
with batch_tab:
    st.write("Upload a CSV with these columns (Type as L/M/H or 0/1/2):")
    st.code(", ".join(RAW_FEATURES))
    template = pd.DataFrame([{**raw, "Type": type_label[0]}])[RAW_FEATURES]
    st.download_button("Download a template CSV", template.to_csv(index=False),
                       "machines_template.csv", "text/csv")
    uploaded = st.file_uploader("Machines CSV", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in RAW_FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch["Type"] = batch["Type"].replace(TYPE_CODES)
            probs = model.predict_proba(add_features(batch[RAW_FEATURES])[feature_names])[:, 1]
            batch["failure_probability"] = probs.round(4)
            batch["risk_level"] = [risk_level(p) for p in probs]
            batch["alert"] = probs >= threshold
            st.metric("Machines flagged", f"{int(batch['alert'].sum())} of {len(batch)}")
            st.dataframe(batch.sort_values("failure_probability", ascending=False))
            st.download_button("Download scored CSV", batch.to_csv(index=False),
                               "scored_machines.csv", "text/csv")

# --------------------------------------------------------- model performance
with about:
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        final = json.loads(metrics_path.read_text())["final"]
        cm = final["confusion_matrix"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC AUC", f"{final['roc_auc']:.3f}")
        m2.metric("Recall", f"{final['recall']:.1%}")
        m3.metric("Precision", f"{final['precision']:.1%}")
        m4.metric("PR AUC", f"{final['pr_auc']:.3f}")
        st.caption(
            f"Held-out test set of {sum(cm.values())} machines: caught {cm['tp']} of "
            f"{cm['tp'] + cm['fn']} failures with {cm['fp']} false alarms."
        )
    for img in ("pr_curve.png", "permutation_importance.png"):
        if (ROOT / "reports" / img).exists():
            st.image(str(ROOT / "reports" / img))
