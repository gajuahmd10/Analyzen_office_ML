"""
Creates test_predictions.csv and prints the train-vs-test report.

Put this file in the SAME folder as train_base_model.py (it imports from it), then run:
    python make_test_predictions.py

What it does
  1. Sorts loans by admission date and splits: first 80% = train, last 20% = test
  2. Trains the base model (4 features) on the train part only
  3. Predicts the unseen test part and writes one row per test loan
  4. Prints train-vs-test metrics, sample rows and error by loan size
"""
import os
import sys

# make `train_base_model` importable when run from any folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (DEFAULT_SET, FEATURE_SETS, LADDER, build_features,
                              clean_purpose, fit_meta, fit_with_early_stopping,
                              metrics, np, pd, rung)
# ---- EDIT THESE PATHS -------------------------------------------------------
CSV_PATH = "SingleCycleLoaneeMember.csv"
OUT_CSV = "test_predictions.csv"
# -----------------------------------------------------------------------------

# 1. Load, order by time, split 80/20 (same split as training script)
df = pd.read_csv(CSV_PATH)
df["_adm"] = pd.to_datetime(df["admission_date"])
df = df.sort_values(["_adm", "member_id"]).reset_index(drop=True)
y = df["loan_amount"].to_numpy(float)
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]

# 2. Build features (constants learned from TRAIN only) and train
meta = fit_meta(tr)
X_tr, X_te = build_features(tr, meta), build_features(te, meta)
feats = FEATURE_SETS[DEFAULT_SET]
model = fit_with_early_stopping(X_tr, np.log1p(y[:cut]), feats)

# 3. Predict (model outputs log1p(loan), so expm1 converts back to taka)
pred_tr = np.expm1(model.predict(X_tr[feats]))
pred_te = np.expm1(model.predict(X_te[feats]))

# 4. Train vs test metrics
report = pd.DataFrame({
    f"TRAIN ({len(tr):,} rows, seen by model)": metrics(y[:cut], pred_tr),
    f"TEST ({len(te):,} later rows, unseen)": metrics(y[cut:], pred_te),
}).T
print(report[["MAE", "R2_log", "RMSLE", "within_1_ladder_step",
              "exact_ladder_step", "AUC_loan_ge_300k"]].round(3).to_string())

# 5. One row per test loan -> CSV
out = pd.DataFrame({
    "member_id": te["member_id"].values,
    "admission_date": te["_adm"].dt.date.values,
    "income": te["total_monthly_income"].values,
    "purpose": clean_purpose(te["loan_use_sources"]).values,
    "actual_loan": y[cut:],
    "predicted_loan": np.clip(pred_te, 4000, 5e6).round(0),
})
out["nearest_standard_amount"] = LADDER[rung(out["predicted_loan"])]   # snap to ladder
out["abs_error"] = (out["predicted_loan"] - out["actual_loan"]).abs()
out["error_%"] = ((out["predicted_loan"] - out["actual_loan"]) / out["actual_loan"] * 100).round(1)
out["exact_standard_match"] = out["nearest_standard_amount"] == out["actual_loan"]
out.to_csv(OUT_CSV, index=False)
print(f"\nwrote {len(out):,} rows -> {OUT_CSV}")

# 6. Quick diagnostics
print("\nSAMPLE of test predictions")
print(out.sample(12, random_state=7)[["income", "purpose", "actual_loan", "predicted_loan",
                                      "nearest_standard_amount", "error_%"]].to_string(index=False))

out["bucket"] = pd.cut(out["actual_loan"], [0, 30000, 50000, 100000, 200000, 299999, 1e9],
                       labels=["≤30k", "40–50k", "60–100k", "120–200k", "250–290k", "≥300k"])
by_size = out.groupby("bucket", observed=True).agg(
    n=("actual_loan", "size"), MAE=("abs_error", "mean"),
    median_pred=("predicted_loan", "median"), median_actual=("actual_loan", "median"),
    median_pct_error=("error_%", "median"))
print("\nERROR BY ACTUAL LOAN SIZE")
print(by_size.round(0).to_string())

print("\n|error| <= 20%:", round((out["error_%"].abs() <= 20).mean(), 3),
      "| <= 50%:", round((out["error_%"].abs() <= 50).mean(), 3),
      "| exact standard match:", round(out["exact_standard_match"].mean(), 3))
big_share = out.loc[out["actual_loan"] >= 300000, "abs_error"].sum() / out["abs_error"].sum()
print("share of total absolute error from loans >= 300k:", round(big_share, 3))