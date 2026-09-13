"""
GNN Inference Module — Gated GCN-LSTM V1 (Production)
======================================================

Loads the trained Gated GCN-LSTM V1 checkpoint and provides live inference
for all 16 canonical Kerala reservoirs simultaneously.

EXACT TRAINING/INFERENCE PARITY
--------------------------------
This module reproduces the EXACT preprocessing pipeline from
``src/modeling/train_gcn_lstm_gated_v1.py``:

- **Graph:** Graph D (correlation_v1_2), 16 nodes, 41 undirected edges
  → 82 directed edges. Correlation ≥ 0.55, ≥ 365 overlap days, train-only.
- **Node order:** Alphabetically sorted 16 canonical reservoirs.
- **Features:** 5 dynamic features per node per timestep:
  [inflow, water_level, live_storage, rainfall, total_outflow]
  (StandardScaler-normalized, matching ``data/processed/scaled/feature_scaler.pkl``)
- **Sequence length:** 7 days.
- **Input tensor:** ``(B, 16, 7, 5)`` — batch × nodes × timesteps × features.
- **Target transformation:** ``log1p(MCM/day) → StandardScaler``
  (matching ``models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl``).
- **Inverse transform:** ``StandardScaler.inverse_transform → expm1``
  to recover original MCM/day values.
- **Missing nodes:** Zero-padded (same as training; zero-feature nodes
  contribute zero vectors through GCNConv aggregation).

SAFETY
------
- The model is loaded in ``eval()`` mode and ``torch.no_grad()`` context.
- This module NEVER modifies any saved model weights, scalers, or data files.
- If a required artifact is missing, a clear ``FileNotFoundError`` is raised.
- If input dimensions are wrong, a clear ``ValueError`` is raised.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric is required for GNN inference. "
        "Install with: pip install torch_geometric"
    )


# ─── Canonical 16-node reservoir ordering (alphabetical, matches training) ───

CANONICAL_NODE_ORDER: List[str] = [
    "Anathode", "Anayirankal", "Banasura Sagar", "Chenkulam",
    "Idamalayar", "Idukki", "Kakkayam", "Kallarkutty",
    "Kundala", "Mattupetty", "Moozhiyar", "Pamba",
    "Pambla", "Ponmudi", "Poringalkuthu", "Sholayar",
]

NODE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(CANONICAL_NODE_ORDER)}
NUM_NODES = 16
SEQ_LEN = 7
NUM_FEATURES = 5
NUM_TARGETS = 3  # target_1d, target_3d, target_7d

# Feature order (matches prepare_lstm_sequences.py)
FEATURE_ORDER: List[str] = [
    "inflow", "water_level", "live_storage", "rainfall", "total_outflow"
]

TARGET_NAMES: List[str] = ["target_1d", "target_3d", "target_7d"]


# ─── Model Architecture (exact copy from train_gcn_lstm_gated_v1.py) ─────────

class GatedGCNLSTM(nn.Module):
    """
    Dual-branch architecture with learnable scalar gate.

    Local branch:  LSTM(input=5, hidden=64) — processes raw features per node
    Spatial branch: GCNConv(5->32) + ReLU -> LSTM(input=32, hidden=64)
    Fusion: h = h_local + sigmoid(alpha) * h_spatial
    Head:  FC(64->32) -> ReLU -> FC(32->3)

    Gate alpha initialized to -4.595 so sigmoid(alpha) ≈ 0.01.
    """
    def __init__(self, edge_index: torch.Tensor):
        super().__init__()
        self.edge_index = edge_index

        # LOCAL branch
        self.local_lstm = nn.LSTM(
            input_size=5, hidden_size=64, num_layers=1, batch_first=True
        )

        # SPATIAL branch
        self.gcn = GCNConv(5, 32, add_self_loops=True, normalize=True)
        self.gcn_relu = nn.ReLU()
        self.spatial_lstm = nn.LSTM(
            input_size=32, hidden_size=64, num_layers=1, batch_first=True
        )

        # Scalar gate
        self.alpha = nn.Parameter(torch.tensor(-4.595))

        # Shared FC head
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F = x.shape  # (B, 16, 7, 5)

        # LOCAL BRANCH
        local_in = x.view(B * N, T, F)
        local_out, _ = self.local_lstm(local_in)
        h_local = local_out[:, -1, :]  # (B*16, 64)

        # SPATIAL BRANCH
        x_gcn = x.permute(0, 2, 1, 3)  # (B, T, N, F)
        gcn_out = self.gcn_relu(self.gcn(x_gcn, self.edge_index))  # (B, T, N, 32)
        spatial_in = gcn_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, 32)
        spatial_out, _ = self.spatial_lstm(spatial_in)
        h_spatial = spatial_out[:, -1, :]  # (B*16, 64)

        # GATED FUSION
        gate = torch.sigmoid(self.alpha)
        h = h_local + gate * h_spatial  # (B*16, 64)

        # FC HEAD
        out = self.relu(self.fc1(h))
        out = self.output(out)  # (B*16, 3)
        return out.view(B, N, 3)


# ─── GNN Inference Engine ────────────────────────────────────────────────────

class LiveGNNForecaster:
    """
    Production inference interface for the Gated GCN-LSTM V1 model.

    Usage::

        forecaster = LiveGNNForecaster(project_root="/path/to/project")
        result = forecaster.predict_all(history_dict)

    Where ``history_dict`` maps reservoir names to DataFrames containing
    7 consecutive daily observations with columns:
    ``[inflow, water_level, live_storage, rainfall, total_outflow]``.
    """

    # Relative paths (from project root) to required artifacts
    MODEL_CHECKPOINT = "models/gcn_lstm_gated_v1/best_model.pt"
    LOG_TARGET_SCALER = "models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl"
    GRAPH_EDGES = "data/processed/graph/graph_D_correlation_v1_2/edges.csv"
    GRAPH_METADATA = "data/processed/graph/graph_D_correlation_v1_2/metadata.json"

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.device = torch.device("cpu")  # Training was CPU-only

        # 1. Build graph topology
        self._edge_index = self._build_edge_index()

        # 2. Load model
        self._model = self._load_model()

        # 3. Load scalers
        self._log_target_scaler = self._load_log_target_scaler()

        # Cache for latest inference result
        self._last_result: Optional[Dict] = None
        self._last_inference_time: float = 0.0

    def _build_edge_index(self) -> torch.Tensor:
        """Reconstruct the exact Graph D edge_index used during training."""
        edges_path = self.project_root / self.GRAPH_EDGES
        if not edges_path.exists():
            raise FileNotFoundError(
                f"Graph D edges not found: {edges_path}\n"
                f"Required for GNN inference."
            )

        edges_df = pd.read_csv(edges_path)

        # Validate all edge nodes are in canonical ordering
        all_nodes = set(edges_df["source"]) | set(edges_df["target"])
        unknown = all_nodes - set(CANONICAL_NODE_ORDER)
        assert not unknown, (
            f"Graph D contains nodes not in canonical ordering: {unknown}"
        )

        # Build bidirectional edge_index (matching training exactly)
        eil = []
        for _, row in edges_df.iterrows():
            u = NODE_TO_IDX[row["source"]]
            v = NODE_TO_IDX[row["target"]]
            eil.append([u, v])
            eil.append([v, u])

        edge_index = np.unique(
            np.array(eil, dtype=np.int64).T, axis=1
        )

        # Training asserts exactly 82 directed edges
        assert edge_index.shape == (2, 82), (
            f"Expected edge_index shape (2, 82), got {edge_index.shape}. "
            f"Graph D topology mismatch."
        )

        return torch.from_numpy(edge_index).long().to(self.device)

    def _load_model(self) -> GatedGCNLSTM:
        """Load the trained Gated GCN-LSTM checkpoint."""
        ckpt_path = self.project_root / self.MODEL_CHECKPOINT
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"GNN checkpoint not found: {ckpt_path}\n"
                f"Train the model first with train_gcn_lstm_gated_v1.py"
            )

        model = GatedGCNLSTM(edge_index=self._edge_index).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_log_target_scaler(self):
        """Load the log-target StandardScaler used during training."""
        scaler_path = self.project_root / self.LOG_TARGET_SCALER
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Log-target scaler not found: {scaler_path}"
            )
        return joblib.load(scaler_path)

    @property
    def gate_value(self) -> float:
        """Current learned gate value sigmoid(alpha)."""
        return float(torch.sigmoid(self._model.alpha).item())

    @property
    def last_inference_time_ms(self) -> float:
        """Latency of last inference call in milliseconds."""
        return self._last_inference_time * 1000.0

    def predict_all(
        self,
        history_dict: Dict[str, np.ndarray],
        available_mask: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run GNN inference for all 16 reservoirs simultaneously.

        Parameters
        ----------
        history_dict : dict
            Maps canonical reservoir name → numpy array of shape (7, 5).
            The 5 features must be in FEATURE_ORDER:
            [inflow, water_level, live_storage, rainfall, total_outflow].
            These must be ALREADY SCALED (StandardScaler-normalized),
            matching ``data/processed/scaled/feature_scaler.pkl``.
        available_mask : dict, optional
            Maps reservoir name → bool indicating whether data is available.
            Missing or False entries are zero-padded (matching training).

        Returns
        -------
        dict
            Maps reservoir name → {
                "target_1d": float (MCM/day),
                "target_3d": float (MCM/day),
                "target_7d": float (MCM/day),
                "available": bool,
            }
        """
        t0 = time.perf_counter()

        # Build input tensor (1, 16, 7, 5)
        X = np.zeros((1, NUM_NODES, SEQ_LEN, NUM_FEATURES), dtype=np.float32)

        for res_name, idx in NODE_TO_IDX.items():
            if res_name in history_dict:
                arr = history_dict[res_name]
                if arr.shape != (SEQ_LEN, NUM_FEATURES):
                    raise ValueError(
                        f"History for {res_name} has shape {arr.shape}, "
                        f"expected ({SEQ_LEN}, {NUM_FEATURES})"
                    )
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"History for {res_name} contains NaN/Inf values"
                    )
                X[0, idx] = arr
            # else: zero-padded (matches training missing-value handling)

        X_t = torch.from_numpy(X).to(self.device)

        # Run inference
        with torch.no_grad():
            y_pred_scaled = self._model(X_t).cpu().numpy()  # (1, 16, 3)

        # Inverse transform: scaled → log1p domain → MCM/day
        y_flat = y_pred_scaled.reshape(-1, NUM_TARGETS)  # (16, 3)
        y_orig = np.expm1(
            self._log_target_scaler.inverse_transform(y_flat)
        )  # (16, 3) in MCM/day

        # Build result dict
        result = {}
        for res_name, idx in NODE_TO_IDX.items():
            is_available = (
                available_mask.get(res_name, True) if available_mask else True
            ) and (res_name in history_dict)

            result[res_name] = {
                "target_1d": float(y_orig[idx, 0]),
                "target_3d": float(y_orig[idx, 1]),
                "target_7d": float(y_orig[idx, 2]),
                "available": is_available,
            }

        self._last_inference_time = time.perf_counter() - t0
        self._last_result = result
        return result

    def predict_from_flat_arrays(
        self,
        X_flat: np.ndarray,
        meta_df: pd.DataFrame,
        target_date: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict using pre-built flat arrays (matching training data format).

        This is used for evaluation parity testing.

        Parameters
        ----------
        X_flat : np.ndarray
            Shape (N_samples, 7, 5) — scaled feature sequences.
        meta_df : pd.DataFrame
            Must have columns ['date', 'reservoir'] matching X_flat rows.
        target_date : str
            The date to predict for.

        Returns
        -------
        dict
            Same format as predict_all().
        """
        # Filter to the target date
        date_mask = meta_df["date"] == target_date
        date_meta = meta_df[date_mask]
        date_X = X_flat[date_mask.values]

        # Build history_dict
        history_dict = {}
        for i, (_, row) in enumerate(date_meta.iterrows()):
            res_name = row["reservoir"]
            if res_name in NODE_TO_IDX:
                history_dict[res_name] = date_X[i]

        return self.predict_all(history_dict)

    def get_diagnostics(self) -> Dict:
        """Return model diagnostics for monitoring."""
        return {
            "model": "Gated GCN-LSTM V1",
            "graph": "Graph D (correlation_v1_2)",
            "nodes": NUM_NODES,
            "edges_directed": 82,
            "edges_undirected": 41,
            "gate_alpha": float(self._model.alpha.item()),
            "gate_sigmoid": self.gate_value,
            "sequence_length": SEQ_LEN,
            "features": FEATURE_ORDER,
            "targets": TARGET_NAMES,
            "target_units": "MCM/day",
            "checkpoint": self.MODEL_CHECKPOINT,
            "last_inference_ms": self.last_inference_time_ms,
        }
