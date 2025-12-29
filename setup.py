import pandas as pd
import numpy as np

# ============================================================
# HOSPITAL READMISSION PREDICTION – DATA PREPARATION & MODELING
# ============================================================


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
df = pd.read_csv("diabetic_data.csv")


# ============================================================
# 2. TARGET VARIABLE CREATION
#    Binary target: 1 = Readmitted within 30 days
# ============================================================
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
df.drop(columns=["readmitted"], inplace=True)


# ============================================================
# 3. DROP LEAKAGE & NON-INFORMATIVE COLUMNS
# ============================================================
df.drop(columns=[
    "encounter_id",
    "patient_nbr",
    "discharge_disposition_id",
    "weight",
    "payer_code",
    "medical_specialty"
], inplace=True)

# Drop lab result columns with excessive missing values
df.drop(columns=["max_glu_serum", "A1Cresult"], inplace=True)

# Drop diagnosis codes to avoid sparsity and leakage risk
df.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)

# Drop individual medication indicators (high dimensional & sparse)
med_cols = [
    'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
    'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
    'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
    'examide','citoglipton','insulin','glyburide-metformin',
    'glipizide-metformin','glimepiride-pioglitazone',
    'metformin-rosiglitazone','metformin-pioglitazone'
]
df.drop(columns=med_cols, inplace=True)


# ============================================================
# 4. FEATURE ENCODING
# ============================================================
def apply_encoding(df):
    """
    Encodes categorical variables using:
    - Binary encoding
    - Ordinal encoding for age
    """

    # Gender encoding (explicit handling of unknowns)
    gender_map = {
        "Male": 0,
        "Female": 1,
        "Unknown/Invalid": 2
    }
    df["gender_encoded"] = df["gender"].map(gender_map)

    # Binary categorical features
    df["change_encoded"] = df["change"].map({"No": 0, "Ch": 1})
    df["diabetesMed_encoded"] = df["diabetesMed"].map({"No": 0, "Yes": 1})

    # Ordinal age encoding
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)": 4, "[50-60)": 5, "[60-70)": 6,
        "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
    }
    df["age_encoded"] = df["age"].map(age_map)

    # Drop original categorical columns
    df.drop(columns=["gender", "change", "diabetesMed", "age"], inplace=True)

    return df


df = apply_encoding(df)


# ============================================================
# 5. NaN VALIDATION (CRITICAL SAFETY CHECK)
# ============================================================
print("\nNaN check after encoding:")
print(df.isna().sum().sort_values(ascending=False).head())

assert df.isna().sum().sum() == 0, "NaNs still present after preprocessing!"


# ============================================================
# 6. ONE-HOT ENCODING (NOMINAL VARIABLES)
# ============================================================
categorical_cols = ["race", "admission_type_id", "admission_source_id"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nShape after one-hot encoding:", df.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
from sklearn.model_selection import train_test_split

X = df.drop(columns=["readmit_30"])
y = df["readmit_30"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 8. BASELINE MODEL – LOGISTIC REGRESSION
# ============================================================
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_reg.fit(X_train, y_train)


# ============================================================
# 9. BASELINE MODEL EVALUATION
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("\nLogistic Regression – Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nLogistic Regression – Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nLogistic Regression – PR-AUC:")
print(average_precision_score(y_test, y_prob))


# ============================================================
# 10. COST-SENSITIVE THRESHOLD ANALYSIS (LOGISTIC)
# ============================================================
thresholds = np.arange(0.1, 0.6, 0.05)

print("\nThreshold tuning (FN cost = 5, FP cost = 1)\n")

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()

    cost = 5 * fn + 1 * fp
    recall = tp / (tp + fn)

    print(f"Threshold={t:.2f} | Recall={recall:.3f} | Cost={cost}")


# ============================================================
# 11. ADVANCED MODEL – RANDOM FOREST
# ============================================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# ============================================================
# 12. RANDOM FOREST EVALUATION
# ============================================================
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST RESULTS ===")

print("\nConfusion Matrix (RF):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (RF):")
print(classification_report(y_test, y_pred_rf, digits=3))

print("\nPR-AUC (RF):")
print(average_precision_score(y_test, y_prob_rf))


# ============================================================
# 13. COST-SENSITIVE THRESHOLD ANALYSIS (RF)
# ============================================================
print("\nRF Threshold tuning (FN cost = 5, FP cost = 1)\n")

for t in thresholds:
    y_pred_t = (y_prob_rf >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()

    cost = 5 * fn + 1 * fp
    recall = tp / (tp + fn)

    print(f"Threshold={t:.2f} | Recall={recall:.3f} | Cost={cost}")


# ============================================================
# 14. FEATURE IMPORTANCE (RANDOM FOREST)
# ============================================================
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 15 Important Features (Random Forest):")
print(feature_importance.head(15))


# ============================================================
# 15. SAVE FINAL PROCESSED DATA
# ============================================================
df.to_csv("processed_data.csv", index=False)

print("\nProcessed dataset saved as 'processed_data.csv'")
