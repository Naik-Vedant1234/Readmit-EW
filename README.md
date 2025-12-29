🏥 READMIT-EW

Hospital Readmission Early Warning System

READMIT-EW is a cost-sensitive predictive analytics and decision-support system designed to identify patients at high risk of 30-day hospital readmission at the time of discharge and support post-discharge care planning.

The system focuses on risk stratification and decision support, not deterministic medical prediction.

📌 Problem Statement

Hospital readmissions within 30 days are:

Costly for healthcare systems

Indicators of suboptimal discharge planning

Difficult to prevent with limited follow-up resources

Providing intensive post-discharge care to all patients is infeasible.
The challenge is to prioritize limited resources toward patients most likely to be readmitted.

🎯 Project Objective

The objective of READMIT-EW is to:

Stratify discharged patients into risk tiers (Low / Medium / High) using cost-sensitive predictive analytics so that hospitals can proactively plan follow-up care.

This system does not replace clinicians and does not provide medical advice.

📊 Dataset

Source: Diabetes 130-US Hospitals Dataset

Size: ~101,000 patient encounters

Target: Readmission within 30 days (readmit_30)

Target Engineering

Original 3-class target (NO, >30, <30) converted to binary:

1 → Readmitted within 30 days

0 → Otherwise

This aligns with real hospital penalty windows and intervention timelines.

🧹 Data Preprocessing & Leakage Prevention

To ensure validity and real-world applicability:

Removed:

Patient identifiers

Post-discharge information (leakage)

High-sparsity diagnosis and medication codes

Columns with excessive missing values

Retained:

Utilization history (inpatient, emergency, outpatient counts)

Care complexity indicators

Length of stay

Demographics available at discharge

Admission context

All features used are available at or before discharge.

🧠 Modeling Approach

Two models were implemented for comparison:

1️⃣ Logistic Regression (Baseline)

Interpretable baseline

Handles class imbalance via class_weight="balanced"

2️⃣ Random Forest (Final Model)

Captures non-linear interactions

Robust to noise

Provides feature importance for interpretability

Model complexity was intentionally limited to avoid overfitting and maintain explainability.

⚖️ Cost-Sensitive Decision Framework

Healthcare decisions involve asymmetric costs:

False Negative (missed high-risk patient) → High cost

False Positive (extra monitoring) → Lower cost

Cost Function
Cost = 5 × False Negatives + 1 × False Positives

Decision Threshold

Optimized using cost analysis

Final threshold: 0.40 (not the default 0.50)

This prioritizes recall over precision, which is appropriate in healthcare settings.

📈 Evaluation Metrics

Because of class imbalance (~11% positive class), accuracy is not meaningful.

Primary metrics:

Recall (missed high-risk patients)

PR-AUC (ranking ability for rare events)

Cost-based evaluation

The final system achieves meaningful improvement over random baseline and demonstrates effective risk stratification.

🖥️ Dashboard Features (Streamlit)

The interactive dashboard provides:

🔹 Patient-Level View

Risk probability and percentile

Risk category (Low / Medium / High)

Model agreement / disagreement indicator

Recommended follow-up actions

Local patient context (key risk drivers)

🔹 Population-Level View

Risk distribution across all patients

High / Medium / Low risk percentages

🔹 Decision Transparency

Cost vs threshold visualization

Explicit threshold justification

Global feature importance

🔹 Export

Downloadable patient risk summary (CSV)

🚦 Interpretation of Predictions

A high-risk prediction does NOT mean:

The patient will definitely be readmitted

The patient should be kept hospitalized

It means:

The patient should receive enhanced post-discharge follow-up and monitoring.

READMIT-EW supports discharge planning, not diagnosis or treatment decisions.

🛡️ Ethical & Practical Safeguards

No medical advice is provided

No treatment recommendations

Final decisions remain with clinicians

Clear disclaimers included in the UI

🧩 Project Structure
├── setup.py              # Data preprocessing & model evaluation
├── app.py                # Streamlit dashboard
├── processed_data.csv    # Final processed dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Prepare data
python setup.py

3. Launch dashboard
python -m streamlit run app.py

📌 Limitations & Future Work

Diagnosis and medication interactions are not modeled

Temporal sequencing of visits is not included

External validation on other hospital systems is required

Possible extensions:

ICD code grouping (Charlson / Elixhauser)

Temporal modeling

Integration with hospital EHR systems

🏁 Conclusion

READMIT-EW demonstrates how predictive analytics can be translated into actionable decision support by combining:

Leakage-free preprocessing

Cost-sensitive modeling

Transparent evaluation

Human-centered system design

The project emphasizes practical impact over algorithmic complexity.

⚠️ Disclaimer

This system is intended for educational and decision-support purposes only and does not replace professional clinical judgment.