"""Step 9: Streamlit web app.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from predict import load_model, risk_level

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")


@st.cache_resource  # load the model once, not on every interaction
def get_model():
    return load_model()


model, feature_names = get_model()

st.title("🔧 Predictive Maintenance Alert System")
st.markdown("Failure-risk prediction for industrial equipment from sensor readings")

st.sidebar.header("Equipment Sensor Data")
type_label = st.sidebar.selectbox(
    "Product quality type", ["L (low)", "M (medium)", "H (high)"]
)
input_data = {
    "Type": {"L": 0, "M": 1, "H": 2}[type_label[0]],
    "Air temperature K": st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1),
    "Process temperature K": st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1),
    "Rotational speed rpm": st.sidebar.slider("Rotational Speed (rpm)", 1150, 2900, 1500),
    "Torque Nm": st.sidebar.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.1),
    "Tool wear min": st.sidebar.slider("Tool Wear (min)", 0, 260, 100),
}
input_df = pd.DataFrame([input_data])[feature_names]

failure_probability = float(model.predict_proba(input_df)[0][1])
level = risk_level(failure_probability)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Failure Probability", value=f"{failure_probability * 100:.1f}%")
with col2:
    if level == "Low":
        st.success("✅ Low Risk - Normal operation")
    elif level == "Medium":
        st.warning("⚠️ Medium Risk - Schedule maintenance soon")
    else:
        st.error("🚨 High Risk - Schedule urgent maintenance")

st.subheader("Current Equipment Status")
status = input_df.T.reset_index()
status.columns = ["Feature", "Value"]
status.loc[status["Feature"] == "Type", "Value"] = type_label
st.table(status)

st.subheader("Maintenance Recommendations")
if level == "Low":
    st.write("✅ Continue regular monitoring. No action needed.")
elif level == "Medium":
    st.write("⚠️ Schedule preventive maintenance within 2 weeks.")
else:
    st.write("🚨 URGENT: Schedule maintenance immediately. Risk of critical failure.")

st.subheader("Batch scoring")
uploaded = st.file_uploader(
    "Upload a CSV with columns: " + ", ".join(feature_names), type="csv"
)
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    missing = [c for c in feature_names if c not in batch.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        probs = model.predict_proba(batch[feature_names])[:, 1]
        batch["failure_probability"] = probs.round(4)
        batch["risk_level"] = [risk_level(p) for p in probs]
        st.dataframe(batch.sort_values("failure_probability", ascending=False))
