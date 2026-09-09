import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
V3_PREDS = "results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv"
V3_HIST = "results/lstm_pytorch_v3_logtarget/training_history.csv"
COMP_CSV = "results/final_model_comparison/final_model_comparison_original_units.csv"
PERIOD_CSV = "results/lstm_pytorch_v3_logtarget/v3_period_comparison.csv"
OUT_FIGS = "results/thesis_visualizations/figures"
OUT_TABLES = "results/thesis_visualizations/tables"

# Setup Seaborn theme for thesis quality
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Load Data
preds_df = pd.read_csv(V3_PREDS)
preds_df['date'] = pd.to_datetime(preds_df['date'])
hist_df = pd.read_csv(V3_HIST)
comp_df = pd.read_csv(COMP_CSV)
period_df = pd.read_csv(PERIOD_CSV)

# Figure 1: Actual vs Predicted Time-Series (Per Horizon)
for h in ['1d', '3d', '7d']:
    plt.figure(figsize=(12, 5))
    agg_df = preds_df.groupby('date')[[f'target_{h}_actual', f'target_{h}_prediction']].mean()
    agg_df = agg_df.resample('D').asfreq().reset_index()
    plt.plot(agg_df['date'], agg_df[f'target_{h}_actual'], label='Actual Inflow', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(agg_df['date'], agg_df[f'target_{h}_prediction'], label='Predicted Inflow', color='tab:blue', linewidth=1.5, alpha=0.8)
    plt.title(f"LSTM V3 Actual vs Predicted Inflow (Average across all reservoirs) - {h} Horizon", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Inflow (original units)", fontsize=12)
    plt.legend()
    plt.savefig(f"{OUT_FIGS}/v3_actual_vs_predicted_{h}.png")
    plt.close()

# Figure 2: Prediction Scatter Plots
for h in ['1d', '3d', '7d']:
    plt.figure(figsize=(7, 7))
    actual = preds_df[f'target_{h}_actual']
    pred = preds_df[f'target_{h}_prediction']
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    
    plt.scatter(actual, pred, alpha=0.4, color='tab:blue', s=10)
    
    # Ideal y=x line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Ideal (y=x)")
    
    plt.title(f"LSTM V3 Prediction Scatter - {h} Horizon\nR² = {r2:.3f}, MAE = {mae:.3f}", fontsize=14)
    plt.xlabel("Actual Inflow", fontsize=12)
    plt.ylabel("Predicted Inflow", fontsize=12)
    plt.legend()
    plt.savefig(f"{OUT_FIGS}/v3_scatter_{h}.png")
    plt.close()

# Figure 3: Training History
plt.figure(figsize=(10, 6))
plt.plot(hist_df['epoch'], hist_df['train_loss'], label="Training Loss", color="tab:blue")
plt.plot(hist_df['epoch'], hist_df['val_loss'], label="Validation Loss", color="tab:orange")
best_epoch = hist_df.loc[hist_df['val_loss'].idxmin(), 'epoch']
best_loss = hist_df['val_loss'].min()
plt.axvline(best_epoch, color='red', linestyle='--', label=f"Best Model (Epoch {int(best_epoch)})")
plt.scatter([best_epoch], [best_loss], color='red', zorder=5)
plt.title("LSTM V3 Training History", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Masked Huber Loss (Log Space)", fontsize=12)
plt.legend()
plt.savefig(f"{OUT_FIGS}/v3_training_history.png")
plt.close()

# Figure 4: Model Comparison (R2 and MAE)
models = ['Persistence', 'Rolling_Mean', 'HistGradientBoosting', 'LSTM_V3']
display_models = ['Persistence', 'Rolling Mean', 'HistGradientBoosting', 'LSTM V3']
comp_r2 = comp_df.pivot(index='target', columns='model', values='R2')[models]
comp_mae = comp_df.pivot(index='target', columns='model', values='MAE')[models]

comp_r2.columns = display_models
comp_r2.index = ['1d', '3d', '7d']
comp_r2.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Comparison: Overall Test R²", fontsize=14)
plt.ylabel("R² (Higher is better)", fontsize=12)
plt.xlabel("Forecast Horizon", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Model", loc="best")
plt.savefig(f"{OUT_FIGS}/model_comparison_r2.png")
plt.close()

comp_mae.columns = display_models
comp_mae.index = ['1d', '3d', '7d']
comp_mae.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Comparison: Overall Test MAE", fontsize=14)
plt.ylabel("MAE (Lower is better)", fontsize=12)
plt.xlabel("Forecast Horizon", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Model", loc="best")
plt.savefig(f"{OUT_FIGS}/model_comparison_mae.png")
plt.close()

# Figure 5: GNN Ablation Story
gnn_data = {
    'Model': ['LSTM V3', 'Geographic GCN', 'Identity GCN', 'Correlation GCN', 'Gated GCN-LSTM'],
    'target_1d': [0.759, -0.706, 0.638, -0.098, 0.629],
    'target_3d': [0.654, -0.802, 0.586, -0.089, 0.582],
    'target_7d': [0.501, -0.487, 0.531, 0.057, 0.511]
}
gnn_df = pd.DataFrame(gnn_data).set_index('Model')
gnn_df.columns = ['1d', '3d', '7d']

ax = gnn_df.T.plot(kind='bar', figsize=(12, 6), colormap='tab10')
plt.title("Evaluated Spatial-Model Ablations (Overall Test R²)", fontsize=14)
plt.ylabel("R² (Higher is better)", fontsize=12)
plt.xlabel("Forecast Horizon", fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.xticks(rotation=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='lower right', title="Architecture")
plt.ylim(max(gnn_df.min().min() - 0.1, -1.0), 1.0)
plt.savefig(f"{OUT_FIGS}/spatial_ablation_r2.png")
plt.close()

# Figure 6: Period Comparison
period_pivot_r2 = period_df.pivot(index='target', columns='period', values='R2')
period_pivot_r2.index = ['1d', '3d', '7d']
period_pivot_r2.plot(kind='bar', figsize=(9, 6), color=['tab:blue', 'tab:orange'])
plt.title("LSTM V3 Period Comparison: Test R²", fontsize=14)
plt.ylabel("R²", fontsize=12)
plt.xlabel("Forecast Horizon", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Period")
plt.savefig(f"{OUT_FIGS}/v3_period_comparison_r2.png")
plt.close()

period_pivot_mae = period_df.pivot(index='target', columns='period', values='MAE')
period_pivot_mae.index = ['1d', '3d', '7d']
period_pivot_mae.plot(kind='bar', figsize=(9, 6), color=['tab:blue', 'tab:orange'])
plt.title("LSTM V3 Period Comparison: Test MAE", fontsize=14)
plt.ylabel("MAE", fontsize=12)
plt.xlabel("Forecast Horizon", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Period")
plt.savefig(f"{OUT_FIGS}/v3_period_comparison_mae.png")
plt.close()

# Figure 7: Per-Reservoir Performance & Table
res_metrics = []
for res, grp in preds_df.groupby('reservoir'):
    for h in ['1d', '3d', '7d']:
        a = grp[f'target_{h}_actual']
        p = grp[f'target_{h}_prediction']
        res_metrics.append({
            'Reservoir': res,
            'Horizon': h,
            'MAE': mean_absolute_error(a, p),
            'RMSE': np.sqrt(mean_squared_error(a, p)),
            'R2': r2_score(a, p)
        })
res_df = pd.DataFrame(res_metrics)
res_df.to_csv(f"{OUT_TABLES}/v3_per_reservoir_metrics.csv", index=False)

for h in ['1d', '3d', '7d']:
    sub_df = res_df[res_df['Horizon'] == h].sort_values('R2', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sub_df, y='Reservoir', x='R2', palette='coolwarm_r')
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"LSTM V3 Per-Reservoir R² ({h} Horizon)", fontsize=14)
    plt.xlabel("R²")
    plt.ylabel("")
    plt.savefig(f"{OUT_FIGS}/v3_reservoir_r2_{h}.png")
    plt.close()

# Figure 8: Residual Analysis
for h in ['1d', '3d', '7d']:
    plt.figure(figsize=(10, 5))
    residuals = preds_df[f'target_{h}_actual'] - preds_df[f'target_{h}_prediction']
    sns.histplot(residuals, bins=50, kde=True, color='tab:blue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"LSTM V3 Residual Distribution (Actual - Predicted) - {h} Horizon", fontsize=14)
    plt.xlabel("Error (Original Units)")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUT_FIGS}/v3_residuals_{h}.png")
    plt.close()

# Figure 9: Extreme-flow Limitation
max_idx = preds_df['target_7d_actual'].idxmax()
max_date = preds_df.loc[max_idx, 'date']
max_res = preds_df.loc[max_idx, 'reservoir']

res_ts = preds_df[preds_df['reservoir'] == max_res].sort_values('date')
res_ts = res_ts.set_index('date').resample('D').asfreq().reset_index()
plt.figure(figsize=(12, 5))
plt.plot(res_ts['date'], res_ts['target_7d_actual'], label='Actual Inflow', color='black', linewidth=2)
plt.plot(res_ts['date'], res_ts['target_7d_prediction'], label='Predicted Inflow', color='tab:red', linewidth=2)
plt.title(f"Extreme-Flow Limitation (LSTM V3, 7d Horizon, Reservoir: {max_res})", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Inflow (original units)", fontsize=12)
plt.legend()
plt.savefig(f"{OUT_FIGS}/v3_extreme_flow_limitation.png")
plt.close()

# Figure 10: June 2025 Discontinuity
agg_df = preds_df.groupby('date').apply(lambda x: mean_absolute_error(x['target_7d_actual'], x['target_7d_prediction'])).reset_index(name='MAE')
agg_df = agg_df.set_index('date').resample('D').asfreq()
agg_df['Rolling_MAE'] = agg_df['MAE'].rolling(window=7, center=True, min_periods=1).mean()
agg_df = agg_df.reset_index()

plt.figure(figsize=(12, 5))
plt.plot(agg_df['date'], agg_df['Rolling_MAE'], color='tab:red', linewidth=2)
plt.axvspan(pd.to_datetime("2025-06-01"), agg_df['date'].max(), color='gray', alpha=0.2, label="Monsoon Period (Observed Discontinuity)")
plt.title("Observed Structural Discontinuity: Rolling 7-Day MAE (7d Horizon)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Rolling Mean Absolute Error", fontsize=12)
plt.legend()
plt.savefig(f"{OUT_FIGS}/june_2025_structural_discontinuity.png")
plt.close()

# Save final CSV Tables
v3_metrics = []
for h in ['1d', '3d', '7d']:
    a = preds_df[f'target_{h}_actual']
    p = preds_df[f'target_{h}_prediction']
    v3_metrics.append({
        'Horizon': f"target_{h}",
        'MAE': mean_absolute_error(a, p),
        'RMSE': np.sqrt(mean_squared_error(a, p)),
        'R2': r2_score(a, p),
        'Bias': np.mean(p - a),
        'Negative_Count': (p < 0).sum(),
        'Negative_Percentage': 100 * (p < 0).sum() / len(p)
    })
pd.DataFrame(v3_metrics).to_csv(f"{OUT_TABLES}/final_v3_metrics.csv", index=False)
comp_df.to_csv(f"{OUT_TABLES}/final_model_comparison.csv", index=False)
period_df.to_csv(f"{OUT_TABLES}/final_period_comparison.csv", index=False)

print("Visualizations and tables generated successfully.")
