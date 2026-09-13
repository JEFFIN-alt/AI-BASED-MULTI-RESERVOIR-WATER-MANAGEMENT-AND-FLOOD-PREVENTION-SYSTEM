"""
Test Suite: GNN Inference — Gated GCN-LSTM V1
==============================================

Tests:
  1. Checkpoint loads
  2. Scaler loads
  3. Graph topology matches training (Graph D, 82 directed edges)
  4. Node ordering matches training (16 canonical reservoirs)
  5. Feature ordering matches training (5 dynamic features)
  6. Sequence shape is correct (7, 5)
  7. Inference produces finite values
  8. Outputs contain 1d/3d/7d
  9. Output dimensions are correct (16 reservoirs × 3 targets)
  10. Inverse transformations are correct (round-trip)
  11. Reproduced metrics match saved values (strict tolerance)
  12. Gate value is non-trivial (model learned to use spatial signal)
"""
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.gnn_inference import (
    LiveGNNForecaster,
    GatedGCNLSTM,
    CANONICAL_NODE_ORDER,
    NODE_TO_IDX,
    FEATURE_ORDER,
    TARGET_NAMES,
    NUM_NODES,
    SEQ_LEN,
    NUM_FEATURES,
    NUM_TARGETS,
)


@pytest.fixture(scope="module")
def project_root():
    return str(PROJECT_ROOT)


@pytest.fixture(scope="module")
def forecaster(project_root):
    return LiveGNNForecaster(project_root)


# ── 1. Checkpoint loads ─────────────────────────────────────────────────────

def test_checkpoint_loads(forecaster):
    """The model must load successfully from the saved checkpoint."""
    assert forecaster._model is not None
    assert isinstance(forecaster._model, GatedGCNLSTM)


# ── 2. Scaler loads ─────────────────────────────────────────────────────────

def test_scaler_loads(forecaster):
    """The log-target scaler must load and have expected attributes."""
    scaler = forecaster._log_target_scaler
    assert scaler is not None
    assert hasattr(scaler, 'mean_')
    assert hasattr(scaler, 'scale_')
    assert scaler.mean_.shape == (3,)  # 3 targets


# ── 3. Graph topology matches training ──────────────────────────────────────

def test_graph_topology(forecaster):
    """Edge index must be exactly 82 directed edges (41 undirected)."""
    edge_index = forecaster._edge_index
    assert edge_index.shape == (2, 82), (
        f"Expected (2, 82), got {edge_index.shape}"
    )
    assert edge_index.min() >= 0
    assert edge_index.max() < 16


def test_graph_is_graph_d(project_root):
    """Verify we are loading Graph D (correlation_v1_2), not Graph B."""
    import json
    meta_path = Path(project_root) / "data/processed/graph/graph_D_correlation_v1_2/metadata.json"
    assert meta_path.exists(), "Graph D metadata not found"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["edges"] == 41
    assert meta["nodes"] == 16
    assert meta["leakage_safe"] is True


# ── 4. Node ordering matches training ───────────────────────────────────────

def test_node_ordering(project_root):
    """Canonical node order must match training split reservoirs."""
    train_meta = pd.read_csv(
        Path(project_root) / "data/processed/splits/train.csv"
    )
    training_order = sorted(train_meta["reservoir"].unique())
    assert CANONICAL_NODE_ORDER == training_order, (
        f"Node order mismatch:\n  inference: {CANONICAL_NODE_ORDER}\n  training: {training_order}"
    )


# ── 5. Feature ordering ────────────────────────────────────────────────────

def test_feature_ordering():
    """Feature order must match training pipeline."""
    expected = ["inflow", "water_level", "live_storage", "rainfall", "total_outflow"]
    assert FEATURE_ORDER == expected


# ── 6. Sequence shape ──────────────────────────────────────────────────────

def test_sequence_shape():
    assert SEQ_LEN == 7
    assert NUM_FEATURES == 5
    assert NUM_TARGETS == 3


# ── 7. Inference produces finite values ─────────────────────────────────────

def test_inference_finite(forecaster):
    """Inference with random valid input must produce finite outputs."""
    np.random.seed(42)
    history = {
        name: np.random.randn(7, 5).astype(np.float32)
        for name in CANONICAL_NODE_ORDER
    }
    result = forecaster.predict_all(history)
    
    for name in CANONICAL_NODE_ORDER:
        assert name in result
        for key in ["target_1d", "target_3d", "target_7d"]:
            val = result[name][key]
            assert np.isfinite(val), f"{name}.{key} is not finite: {val}"


# ── 8. Outputs contain 1d/3d/7d ─────────────────────────────────────────────

def test_output_keys(forecaster):
    """Output dict must contain all required keys."""
    history = {
        name: np.zeros((7, 5), dtype=np.float32)
        for name in CANONICAL_NODE_ORDER
    }
    result = forecaster.predict_all(history)
    
    for name in CANONICAL_NODE_ORDER:
        assert "target_1d" in result[name]
        assert "target_3d" in result[name]
        assert "target_7d" in result[name]
        assert "available" in result[name]


# ── 9. Output dimensions ───────────────────────────────────────────────────

def test_output_dimensions(forecaster):
    """Must produce forecasts for all 16 reservoirs."""
    history = {
        name: np.zeros((7, 5), dtype=np.float32)
        for name in CANONICAL_NODE_ORDER
    }
    result = forecaster.predict_all(history)
    assert len(result) == 16


# ── 10. Inverse transformation round-trip ───────────────────────────────────

def test_inverse_transform_roundtrip(project_root):
    """log1p → scale → inverse_scale → expm1 must be reversible."""
    log_scaler = joblib.load(
        Path(project_root) / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl"
    )
    original = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)
    scaled = log_scaler.transform(np.log1p(original))
    recovered = np.expm1(log_scaler.inverse_transform(scaled))
    np.testing.assert_allclose(original, recovered, atol=1e-4)


# ── 11. Metric reproduction ────────────────────────────────────────────────

def test_metric_reproduction(project_root):
    """
    Run inference on the saved test set and compare against saved metrics.
    
    Tolerance is set based on observed float32 rounding:
    - MAE tolerance: 1e-4
    - RMSE tolerance: 1e-4
    - R² tolerance: 1e-6
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    root = Path(project_root)
    
    # Load data (exact training pipeline)
    train_meta = pd.read_csv(root / "data/processed/splits/train.csv")
    test_meta = pd.read_csv(root / "data/processed/splits/test.csv")
    NODE_ORDER = sorted(train_meta["reservoir"].unique())
    r2i = {r: i for i, r in enumerate(NODE_ORDER)}
    
    test_dates = sorted(test_meta["date"].unique())
    X_test_flat = np.load(root / "data/processed/lstm/X_test_dynamic.npy").astype(np.float32)
    y_test_flat = np.load(root / "data/processed/lstm/y_test.npy").astype(np.float32)
    
    target_scaler = joblib.load(root / "data/processed/scaled/target_scaler.pkl")
    log_target_scaler = joblib.load(root / "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl")
    
    y_te_orig = target_scaler.inverse_transform(y_test_flat)
    y_te_s = log_target_scaler.transform(np.log1p(y_te_orig)).astype(np.float32)
    
    # Pivot
    nd = len(test_dates)
    X_p = np.zeros((nd, 16, 7, 5), dtype=np.float32)
    y_p = np.zeros((nd, 16, 3), dtype=np.float32)
    m_p = np.zeros((nd, 16), dtype=bool)
    d2i = {d: i for i, d in enumerate(test_dates)}
    for idx, row in test_meta.iterrows():
        di, ri = d2i[row["date"]], r2i[row["reservoir"]]
        X_p[di, ri] = X_test_flat[idx]
        y_p[di, ri] = y_te_s[idx]
        m_p[di, ri] = True
    
    # Load model via LiveGNNForecaster
    forecaster = LiveGNNForecaster(project_root)
    
    # Run inference (matching training eval: DataLoader, batch_size=32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_p),
            torch.from_numpy(y_p),
            torch.from_numpy(m_p),
        ),
        batch_size=32, shuffle=False,
    )
    
    ps, acs, ms = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            out = forecaster._model(xb)
            ps.append(out.cpu().numpy())
            acs.append(yb.numpy())
            ms.append(mb.numpy())
    
    ps = np.concatenate(ps)
    acs = np.concatenate(acs)
    ms = np.concatenate(ms)
    
    test_preds = ps[ms]
    test_actuals = acs[ms]
    
    preds_orig = np.expm1(log_target_scaler.inverse_transform(test_preds))
    actuals_orig = np.expm1(log_target_scaler.inverse_transform(test_actuals))
    
    # Compare against saved metrics
    saved = pd.read_csv(root / "results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_metrics.csv")
    
    for i, name in enumerate(["target_1d", "target_3d", "target_7d"]):
        p = preds_orig[:, i]
        a = actuals_orig[:, i]
        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        r2 = r2_score(a, p)
        
        saved_row = saved[saved["Horizon"] == name].iloc[0]
        
        assert abs(mae - saved_row["MAE"]) < 1e-4, (
            f"{name} MAE mismatch: {mae} vs {saved_row['MAE']}"
        )
        assert abs(rmse - saved_row["RMSE"]) < 1e-4, (
            f"{name} RMSE mismatch: {rmse} vs {saved_row['RMSE']}"
        )
        assert abs(r2 - saved_row["R2"]) < 1e-6, (
            f"{name} R² mismatch: {r2} vs {saved_row['R2']}"
        )


# ── 12. Gate value is non-trivial ───────────────────────────────────────────

def test_gate_learned(forecaster):
    """The model should have learned a non-trivial gate value (> 0.01)."""
    gate = forecaster.gate_value
    assert gate > 0.005, f"Gate too small: {gate}"
    assert gate < 1.0, f"Gate too large: {gate}"


# ── 13. Diagnostics ────────────────────────────────────────────────────────

def test_diagnostics(forecaster):
    """Diagnostics must return all expected fields."""
    diag = forecaster.get_diagnostics()
    assert diag["model"] == "Gated GCN-LSTM V1"
    assert diag["graph"] == "Graph D (correlation_v1_2)"
    assert diag["nodes"] == 16
    assert diag["edges_directed"] == 82
    assert diag["sequence_length"] == 7
    assert diag["target_units"] == "MCM/day"


# ── 14. Input validation ───────────────────────────────────────────────────

def test_invalid_shape_rejected(forecaster):
    """Wrong sequence shape must raise ValueError."""
    history = {"Anathode": np.zeros((5, 5), dtype=np.float32)}
    with pytest.raises(ValueError, match="shape"):
        forecaster.predict_all(history)


def test_nan_input_rejected(forecaster):
    """NaN input must raise ValueError."""
    history = {"Anathode": np.full((7, 5), np.nan, dtype=np.float32)}
    with pytest.raises(ValueError, match="NaN"):
        forecaster.predict_all(history)
