import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def calc_metrics(actual, pred):
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred),
        "Bias": np.mean(pred - actual)
    }

print("Loading predictions...")
preds_df = pd.read_csv("results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv")
print("Predictions columns:", preds_df.columns.tolist())
print("Total rows:", len(preds_df))

print("\nLoading split...")
test_split = pd.read_csv("data/processed/splits/test.csv")
print("Split total rows:", len(test_split))

# Ensure alignment
if (preds_df['date'] == test_split['date']).all() and (preds_df['reservoir'] == test_split['reservoir']).all():
    print("Alignment: PERFECT MATCH")
else:
    print("Alignment: MISMATCH")

# Recalculate Overall Metrics
print("\n--- OVERALL METRICS ---")
for t in ['target_1d', 'target_3d', 'target_7d']:
    a = preds_df[f"{t}_actual"]
    p = preds_df[f"{t}_prediction"]
    m = calc_metrics(a, p)
    neg = (p < 0).sum()
    print(f"{t}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, Bias={m['Bias']:.3f}, Neg={neg}")

# Check original vs prediction actuals
max_err_1d = np.abs(preds_df['target_1d_actual'] - test_split['target_1d']).max()
max_err_3d = np.abs(preds_df['target_3d_actual'] - test_split['target_3d']).max()
max_err_7d = np.abs(preds_df['target_7d_actual'] - test_split['target_7d']).max()
print(f"\nMax diff between prediction actuals and split raw actuals: 1d={max_err_1d:.6f}, 3d={max_err_3d:.6f}, 7d={max_err_7d:.6f}")

# Period-wise Metrics
preds_df['date'] = pd.to_datetime(preds_df['date'])
jan_apr = preds_df[(preds_df['date'] >= "2025-01-01") & (preds_df['date'] < "2025-05-01")]
jun_aug = preds_df[(preds_df['date'] >= "2025-06-01") & (preds_df['date'] <= "2025-08-31")]

print(f"\n--- JAN-APR 2025 ({len(jan_apr)} rows) ---")
for t in ['target_1d', 'target_3d', 'target_7d']:
    a = jan_apr[f"{t}_actual"]
    p = jan_apr[f"{t}_prediction"]
    m = calc_metrics(a, p)
    print(f"{t}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, Bias={m['Bias']:.3f}")

print(f"\n--- JUN-AUG 2025 ({len(jun_aug)} rows) ---")
for t in ['target_1d', 'target_3d', 'target_7d']:
    a = jun_aug[f"{t}_actual"]
    p = jun_aug[f"{t}_prediction"]
    m = calc_metrics(a, p)
    print(f"{t}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, Bias={m['Bias']:.3f}")

# Scaler verification
fs = joblib.load("data/processed/scaled/feature_scaler.pkl")
ts = joblib.load("models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl")
print(f"\nFeature scaler samples seen: {fs['feature_scaler'].n_samples_seen_}")
print(f"Log target scaler samples seen: {ts.n_samples_seen_}")

# Negative prediction / Plausibility check
print("\n--- PHYSICAL VALIDITY ---")
for t in ['target_1d', 'target_3d', 'target_7d']:
    p = preds_df[f"{t}_prediction"]
    a = preds_df[f"{t}_actual"]
    print(f"{t}: pred min={p.min():.3f}, pred max={p.max():.3f} | actual min={a.min():.3f}, actual max={a.max():.3f}")
    
# V3 Error Analysis by Reservoir
print("\n--- ERROR ANALYSIS BY RESERVOIR ---")
for r, grp in preds_df.groupby('reservoir'):
    a = grp['target_1d_actual']
    p = grp['target_1d_prediction']
    m = calc_metrics(a, p)
    print(f"{r}: 1d R2={m['R2']:.3f}, 1d MAE={m['MAE']:.3f}, 1d Bias={m['Bias']:.3f}")
