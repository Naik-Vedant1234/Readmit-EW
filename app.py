# ======================================================
# READMIT-EW
# Hospital Readmission Early Warning System
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ======================================================
# CONFIGURATION
# ======================================================
RISK_THRESHOLD = 0.40
MEDIUM_THRESHOLD = 0.30
FN_COST = 5
FP_COST = 1

st.set_page_config(
    page_title="READMIT-EW",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("processed_data.csv")

df = load_data()
X = df.drop(columns=["readmit_30"])
y = df["readmit_30"]

# ======================================================
# TRAIN MODELS
# ======================================================
@st.cache_resource
def train_models(X, y):
    log_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    log_model.fit(X, y)
    rf_model.fit(X, y)

    return log_model, rf_model

log_model, rf_model = train_models(X, y)

# ======================================================
# RF PROBABILITIES (GLOBAL)
# ======================================================
@st.cache_data
def compute_rf_probs(_model, X):
    return _model.predict_proba(X)[:, 1]

rf_probs_all = compute_rf_probs(rf_model, X)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# ======================================================
# SIDEBAR — PATIENT SELECTION
# ======================================================
st.sidebar.title("Patient Selection")

mode = st.sidebar.radio(
    "Select Patient By",
    ["High Risk Examples", "Low Risk Examples", "Manual Index"]
)

if mode == "High Risk Examples":
    ids = np.where(rf_probs_all >= RISK_THRESHOLD)[0]
    patient_id = st.sidebar.selectbox("High-Risk Patients", ids)

elif mode == "Low Risk Examples":
    ids = np.where(rf_probs_all < MEDIUM_THRESHOLD)[0]
    patient_id = st.sidebar.selectbox("Low-Risk Patients", ids)

else:
    patient_id = st.sidebar.number_input(
        "Patient ID",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

patient = X.iloc[[patient_id]]
true_label = y.iloc[patient_id]

# ======================================================
# PREDICTIONS
# ======================================================
log_prob = log_model.predict_proba(patient)[0, 1]
rf_prob = rf_model.predict_proba(patient)[0, 1]

def risk_decision(p):
    if p >= RISK_THRESHOLD:
        return "HIGH RISK", "INTERVENTION RECOMMENDED", "#d9534f"
    elif p >= MEDIUM_THRESHOLD:
        return "MEDIUM RISK", "ENHANCED MONITORING ADVISED", "#f0ad4e"
    else:
        return "LOW RISK", "STANDARD DISCHARGE PROTOCOL", "#5cb85c"

log_level, log_action, log_color = risk_decision(log_prob)
rf_level, rf_action, rf_color = risk_decision(rf_prob)

# ======================================================
# RISK PERCENTILE
# ======================================================
risk_percentile = (rf_probs_all < rf_prob).mean() * 100

# ======================================================
# POPULATION RISK DISTRIBUTION
# ======================================================
high_pct = (rf_probs_all >= RISK_THRESHOLD).mean() * 100
med_pct = ((rf_probs_all >= MEDIUM_THRESHOLD) &
           (rf_probs_all < RISK_THRESHOLD)).mean() * 100
low_pct = (rf_probs_all < MEDIUM_THRESHOLD).mean() * 100

# ======================================================
# MAIN UI
# ======================================================
st.title("🏥 READMIT-EW")
st.caption(
    "A cost-sensitive decision-support system for identifying high-risk patients "
    "at discharge and prioritizing post-discharge care."
)

st.markdown("---")

# ======================================================
# POPULATION OVERVIEW
# ======================================================
st.subheader("Population Risk Overview")
st.markdown(
    f"""
    - 🔴 **High Risk:** {high_pct:.1f}%  
    - 🟠 **Medium Risk:** {med_pct:.1f}%  
    - 🟢 **Low Risk:** {low_pct:.1f}%
    """
)

st.markdown("---")

# ======================================================
# PATIENT OVERVIEW
# ======================================================
st.subheader("Patient Overview")
st.markdown(
    f"""
    **Patient ID:** `{patient_id}`  
    **Actual 30-Day Readmission:** `{'YES' if true_label == 1 else 'NO'}`
    """
)

# ======================================================
# MODEL COMPARISON
# ======================================================
st.markdown("---")
st.subheader("Model Predictions")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Logistic Regression")
    st.metric("Predicted Risk", f"{log_prob:.2f}")
    st.markdown(
        f"<div style='background:{log_color};padding:12px;border-radius:6px;"
        f"color:white;font-weight:bold;text-align:center;'>{log_action}</div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown("### Random Forest (Final Model)")
    st.metric("Predicted Risk", f"{rf_prob:.2f}")
    st.markdown(
        f"<div style='background:{rf_color};padding:12px;border-radius:6px;"
        f"color:white;font-weight:bold;text-align:center;'>{rf_action}</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Higher risk than approximately **{risk_percentile:.1f}%** of patients"
    )

# ======================================================
# MODEL AGREEMENT
# ======================================================
if log_level != rf_level:
    st.warning(
        "⚠️ Model Disagreement Detected — manual review recommended."
    )
else:
    st.success(
        "✅ Model Agreement — consistent risk assessment."
    )

# ======================================================
# FOLLOW-UP ACTIONS
# ======================================================
st.markdown("---")
st.subheader("Recommended Follow-up Actions")

if rf_level == "HIGH RISK":
    st.markdown("""
    **High Risk – Targeted Intervention**
    - Early follow-up (3–5 days)
    - Assign care coordinator
    - Post-discharge call within 48 hours
    - Consider remote monitoring
    """)
elif rf_level == "MEDIUM RISK":
    st.markdown("""
    **Medium Risk – Enhanced Monitoring**
    - Early outpatient follow-up
    - Medication reconciliation
    - Post-discharge call within 72 hours
    """)
else:
    st.markdown("""
    **Low Risk – Standard Discharge**
    - Routine discharge protocol
    """)

st.caption(
    "Decisions are optimized using a cost-sensitive framework where missing a "
    "high-risk patient is more costly than additional monitoring."
)

# ======================================================
# LOCAL PATIENT CONTEXT
# ======================================================
st.markdown("---")
st.subheader("Key Factors for This Patient")

top_feats = feature_importance.head(6).index
local_df = pd.DataFrame({
    "Feature": top_feats,
    "Patient Value": patient[top_feats].iloc[0].values
})

st.table(local_df)

# ======================================================
# GLOBAL FEATURE IMPORTANCE
# ======================================================
st.markdown("---")
st.subheader("Global Drivers of Readmission Risk")
st.bar_chart(feature_importance.head(10))

# ======================================================
# COST vs THRESHOLD CURVE (MEDIUM-VALUE ADDITION)
# ======================================================
st.markdown("---")
st.subheader("Cost vs Decision Threshold")

thresholds = np.arange(0.1, 0.6, 0.05)
costs = []

for t in thresholds:
    preds = (rf_probs_all >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    cost = FN_COST * fn + FP_COST * fp
    costs.append(cost)

cost_df = pd.DataFrame({
    "Threshold": thresholds,
    "Cost": costs
})

st.line_chart(cost_df.set_index("Threshold"))
st.caption(
    f"Selected threshold = {RISK_THRESHOLD} "
    f"(FN cost={FN_COST}, FP cost={FP_COST})"
)

# ======================================================
# EXPORT PATIENT RISK REPORT
# ======================================================
st.markdown("---")
st.subheader("Export Patient Risk Summary")

export_df = pd.DataFrame({
    "Patient_ID": [patient_id],
    "RF_Risk_Probability": [rf_prob],
    "Risk_Percentile": [risk_percentile],
    "Risk_Category": [rf_level],
    "Recommended_Action": [rf_action]
})

st.download_button(
    label="Download Patient Risk Report (CSV)",
    data=export_df.to_csv(index=False),
    file_name=f"patient_{patient_id}_risk_report.csv",
    mime="text/csv"
)

# ======================================================
# DISCLAIMER
# ======================================================
st.markdown("---")
st.caption(
    "⚠️ This system provides decision-support insights and does not replace "
    "clinical judgment."
)
