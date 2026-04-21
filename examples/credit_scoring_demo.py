"""End-to-end demo: bias detection + audit trail + model card on a synthetic credit dataset.

Run from the project root:
    python examples/credit_scoring_demo.py
"""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from rai_toolkit.audit.trail import AuditTrail
from rai_toolkit.bias.detector import BiasDetector
from rai_toolkit.model_card.generator import ModelCardGenerator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUTS = Path("outputs/demo")
OUTPUTS.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = "sqlite:///outputs/demo/mlflow.db"
MODEL_NAME = "credit_model_demo"
SENSITIVE_COL = "gender"
TARGET_COL = "approved"

# ---------------------------------------------------------------------------
# Step 1 — Generate synthetic biased dataset
# ---------------------------------------------------------------------------

print("\n[1/6] Generating synthetic credit dataset...")

np.random.seed(42)
n = 1000
n_male, n_female = 650, 350
gender = np.array(["male"] * n_male + ["female"] * n_female)
age = np.concatenate([
    np.random.randint(25, 65, n_male),
    np.random.randint(22, 60, n_female),
])
credit_score = np.concatenate([
    np.random.randint(620, 800, n_male),
    np.random.randint(580, 760, n_female),
])
income = np.concatenate([
    np.random.randint(45000, 90000, n_male),
    np.random.randint(30000, 75000, n_female),
])
y_true = (credit_score > 680).astype(int)

# Introduce bias: flip 80% of female approvals to refusals
female_approved = np.where((gender == "female") & (y_true == 1))[0]
flipped = np.random.choice(female_approved, size=int(len(female_approved) * 0.80), replace=False)
bias_mask = np.zeros(n, dtype=int)
bias_mask[flipped] = 1
y_biased = np.clip(y_true - bias_mask, 0, 1).astype(int)

df = pd.DataFrame({
    "age": age,
    "credit_score": credit_score,
    "income": income,
    SENSITIVE_COL: gender,
    TARGET_COL: y_biased,
})

csv_path = OUTPUTS / "credit_data.csv"
df.to_csv(csv_path, index=False)
print(f"   Dataset saved -> {csv_path}  (n={n})")
print(f"   Male approval rate  : {y_biased[gender=='male'].mean():.1%}")
print(f"   Female approval rate: {y_biased[gender=='female'].mean():.1%}")

# ---------------------------------------------------------------------------
# Step 2 — Train model and log to MLflow
# ---------------------------------------------------------------------------

print("\n[2/6] Training LogisticRegression and logging to MLflow...")

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment("credit_scoring_demo")

X = df.drop(columns=[SENSITIVE_COL, TARGET_COL])  # age, credit_score, income
y = df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="logistic_C1") as run:
    model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", round(acc, 4))
    mlflow.log_metric("f1_score", round(f1, 4))
    mlflow.set_tag("author", "credit_scoring_demo")
    mlflow.set_tag("dataset", "synthetic_credit_biased")
    mlflow.set_tag("training_date", "2026-04-20")
    mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)

    run_id = run.info.run_id

print(f"   accuracy={acc:.4f}  f1={f1:.4f}")
print(f"   MLflow run_id: {run_id}")

# Save model as pkl for the CLI demo
pkl_path = OUTPUTS / "credit_model.pkl"
with pkl_path.open("wb") as f:
    pickle.dump(model, f)
print(f"   Model saved -> {pkl_path}")

# ---------------------------------------------------------------------------
# Step 3 — Bias analysis
# ---------------------------------------------------------------------------

print("\n[3/6] Running bias analysis...")

X_full = df.drop(columns=[TARGET_COL])
bias_report = BiasDetector().analyze(model, X_full, y_true, sensitive_col=SENSITIVE_COL)

print(f"   Demographic Parity Diff : {bias_report.demographic_parity_diff:.4f}")
print(f"   Equal Opportunity Diff  : {bias_report.equal_opportunity_diff:.4f}")
print(f"   Disparate Impact Ratio  : {bias_report.disparate_impact_ratio:.4f}")
print(f"   Risk level              : {bias_report.risk_level}")

bias_json_path = OUTPUTS / "bias_report.json"
bias_json_path.write_text(
    json.dumps(bias_report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"   Bias report saved -> {bias_json_path}")

# ---------------------------------------------------------------------------
# Step 4 — Audit trail: log 20 decisions
# ---------------------------------------------------------------------------

print("\n[4/6] Logging 20 decisions to audit trail...")

trail = AuditTrail(log_dir=str(OUTPUTS / "audit_logs"))
logged_ids = []

sample = df.sample(20, random_state=0)
for _, row in sample.iterrows():
    features = {
        "age": int(row["age"]),
        "credit_score": int(row["credit_score"]),
        "income": int(row["income"]),
    }
    pred_input = pd.DataFrame([features])
    pred = int(model.predict(pred_input)[0])
    proba = float(model.predict_proba(pred_input)[0][1])

    record_id = trail.log(
        model_name=MODEL_NAME,
        model_version="1",
        input_features=features,
        prediction=pred,
        prediction_proba=round(proba, 4),
        user_id="demo_analyst",
    )
    logged_ids.append(record_id)

# Verify all records
valid = all(trail.verify(rid) for rid in logged_ids)
print(f"   Logged {len(logged_ids)} records — all signatures valid: {valid}")

# Export to CSV
audit_csv = OUTPUTS / "audit_export.csv"
count = trail.export_csv(audit_csv)
print(f"   Audit export saved -> {audit_csv}  ({count} rows)")

# ---------------------------------------------------------------------------
# Step 5 — Model card generation
# ---------------------------------------------------------------------------

print("\n[5/6] Generating model card...")

card_path = OUTPUTS / "model_card.md"
ModelCardGenerator(tracking_uri=MLFLOW_DB).generate(
    model_uri=f"models:/{MODEL_NAME}/1",
    output_path=card_path,
    bias_report=bias_report,
)
print(f"   Model card saved -> {card_path}")

# ---------------------------------------------------------------------------
# Step 6 — Summary
# ---------------------------------------------------------------------------

print("\n[6/6] Demo complete. Outputs:")
for p in sorted(OUTPUTS.rglob("*")):
    if p.is_file():
        size_kb = p.stat().st_size / 1024
        print(f"   {p.relative_to(OUTPUTS)}  ({size_kb:.1f} KB)")

print(f"""
Summary
-------
  Model accuracy    : {acc:.1%}
  Bias risk level   : {bias_report.risk_level}
  Audit records     : {len(logged_ids)} (all verified)
  Outputs folder    : {OUTPUTS.resolve()}
""")
