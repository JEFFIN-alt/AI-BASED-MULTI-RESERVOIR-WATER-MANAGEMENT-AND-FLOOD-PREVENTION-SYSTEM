import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

from src.modeling.v3_feature_contract import (
    DYNAMIC_FEATURES,
    FEATURE_UNITS,
    FORECAST_UNITS,
    HORIZONS,
    HISTORY_DAYS,
    MODEL_VERSION,
    TARGET_COLUMNS,
    ForecastStatus,
    verify_scaler_contract,
)

# Filter out scikit-learn version mismatch warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ForecastUnavailableError(RuntimeError):
    """
    Raised when a required V3 input feature is not legitimately available.

    Stage 5: the live pipeline must FAIL LOUDLY rather than silently substitute
    a different physical quantity (e.g. a storage percentage in a slot trained
    on metres) or zero-fill a missing measurement.
    """


class SimpleReservoirLSTM(nn.Module):
    """
    Identical architecture to the one defined in train_lstm_pytorch_v3_logtarget.py.
    """
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, output_size)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_hidden = sequence_output[:, -1, :]
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class LiveForecaster:
    """
    Clean inference interface for the validated LSTM V3 logtarget model.
    Reproduces training-time preprocessing exactly.

    STAGE 5 — PROVENANCE
    --------------------
    The numeric path (scaling -> LSTM -> unscale -> expm1) is UNCHANGED. What
    changed is that the pipeline is now explicit about WHERE ITS INPUTS CAME
    FROM, so a demonstration forecast can never be mistaken for the validated
    real-data evaluation.

    The model, the feature scaler and the log-target scaler are loaded
    READ-ONLY and are never modified, re-fitted or rescaled.
    """
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / "models" / "lstm_pytorch_v3_logtarget"
        self.scaler_dir = self.project_root / "data" / "processed" / "scaled"

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleReservoirLSTM().to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / "best_model.pt", map_location=self.device, weights_only=True))
        self.model.eval()

        # Load scalers
        self.log_target_scaler = joblib.load(self.model_dir / "log_target_scaler.pkl")
        feature_scaler_bundle = joblib.load(self.scaler_dir / "feature_scaler.pkl")
        self.feature_scaler = feature_scaler_bundle["feature_scaler"]

        # Exact feature order required by V3 training pipeline (frozen contract)
        self.dynamic_features = list(DYNAMIC_FEATURES)
        self.expected_cols = self.feature_scaler.feature_names_in_

        # FAIL LOUDLY if the frozen scaler does not match the frozen contract.
        # A silent column reordering would corrupt every prediction while still
        # producing plausible-looking numbers.
        self.scaler_contract = verify_scaler_contract(self.feature_scaler)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_input_provenance(self, input_provenance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decide the forecast status from the INPUT provenance.

        A forecast is only ``VALIDATED`` when every input feature is a real
        historical measurement. Anything else — simulation-derived, synthetic
        demonstration placeholders, or unspecified — is
        ``DEMONSTRATION_ONLY``.
        """
        if not input_provenance:
            return {
                "forecast_status": ForecastStatus.DEMONSTRATION_ONLY.value,
                "forecast_provenance": "UNSPECIFIED_INPUT_PROVENANCE",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "input_provenance": None,
                "note": (
                    "No input provenance was supplied, so this forecast is NOT "
                    "claimed to be validated. Reported V3 test metrics do not apply."
                ),
            }

        all_real = bool(input_provenance.get("all_real_measurements", False))
        status = (
            ForecastStatus.VALIDATED if all_real
            else ForecastStatus.DEMONSTRATION_ONLY
        )
        return {
            "forecast_status": status.value,
            "forecast_provenance": (
                "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL" if all_real
                else "SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL"
            ),
            "is_simulated": not all_real,
            "validated_metrics_apply": all_real,
            "input_provenance": input_provenance,
            "note": (
                "Inputs are real measurements; reported V3 evaluation metrics apply."
                if all_real else
                "Inputs are simulated and/or synthetic demonstration placeholders. "
                "This is a DEMONSTRATION forecast and is NOT equivalent to the "
                "validated real-data V3 evaluation."
            ),
        }

    def predict(self, history_df: pd.DataFrame,
                input_provenance: Optional[Dict[str, Any]] = None) -> dict:
        """
        Input: 7 consecutive historical observations.
        Columns required: water_level, live_storage, inflow, rainfall, total_outflow

        Output: Dict with forecast_1d, 3d, 7d and full provenance metadata.

        Parameters
        ----------
        input_provenance : dict, optional
            Provenance summary from
            ``src.modeling.v3_feature_contract.feature_provenance_summary``.
            When supplied and it declares UNAVAILABLE features, this method
            raises ``ForecastUnavailableError`` instead of silently proceeding.
        """
        if len(history_df) != HISTORY_DAYS:
            raise ValueError(f"Expected exactly {HISTORY_DAYS} rows of history, got {len(history_df)}.")

        required_cols = list(self.dynamic_features)
        for col in required_cols:
            if col not in history_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # STAGE 5: refuse to run when a required physical quantity is not
        # legitimately available. No substitution, no zero-fill.
        if input_provenance:
            unavailable = list(input_provenance.get("unavailable", []))
            if unavailable:
                raise ForecastUnavailableError(
                    "Cannot produce a forecast: required V3 input feature(s) are "
                    f"UNAVAILABLE in the live simulation: {unavailable}. "
                    "Use an explicit SYNTHETIC_DEMO placeholder or leave the "
                    "forecast unavailable — do not fabricate the quantity."
                )

        # Check chronological ordering.
        # The frozen model consumes a CONTIGUOUS 7-day window. The previous
        # implementation emitted a warning on the first iteration unconditionally
        # and never actually tested consecutiveness, so a gapped window passed
        # silently. It now validates for real and refuses a gapped window.
        if "date" in history_df.columns:
            dates = pd.to_datetime(history_df["date"]).dt.normalize()
            day_gaps = dates.diff().dropna().dt.days
            if not (day_gaps == 1).all():
                raise ValueError(
                    "History dates are not consecutive daily observations: "
                    f"{[str(d.date()) for d in dates]}. The frozen V3 model "
                    "requires a contiguous 7-day window."
                )

        # Check finite values
        if not np.isfinite(history_df[required_cols].astype(float).values).all():
            raise ValueError("History contains NaN or infinite values.")

        # 2. Build flattened row for feature scaler
        flat_row = {}
        for feat in self.dynamic_features:
            for day_idx in range(1, HISTORY_DAYS + 1): # day_1 to day_7
                # history_df index 0 is day_1 (oldest), index 6 is day_7 (latest)
                flat_row[f"{feat}_day_{day_idx}"] = float(history_df.iloc[day_idx - 1][feat])

        # The frozen scaler expects 51 columns. The 16 static / availability
        # columns are scaled here only so the scaler can run; the LSTM then
        # DISCARDS them (its input tensor is the 35 dynamic values only).
        # Filling them is therefore numerically inert — pinned by
        # test_static_columns_cannot_influence_the_forecast.
        for col in self.expected_cols:
            if col not in flat_row:
                flat_row[col] = 0.0

        flat_df = pd.DataFrame([flat_row], columns=self.expected_cols)

        # 3. Scale features
        scaled_flat = self.feature_scaler.transform(flat_df)
        scaled_flat_df = pd.DataFrame(scaled_flat, columns=self.expected_cols)

        # 4. Build dynamic tensor (1, 7, 5)
        tensor = np.zeros((1, HISTORY_DAYS, len(self.dynamic_features)), dtype=np.float32)
        for feat_idx, feat in enumerate(self.dynamic_features):
            cols = [f"{feat}_day_{day_idx}" for day_idx in range(1, HISTORY_DAYS + 1)]
            tensor[0, :, feat_idx] = scaled_flat_df[cols].to_numpy(dtype=np.float32)[0]

        tensor_pt = torch.from_numpy(tensor).to(self.device)

        # 5. Inference
        with torch.no_grad():
            pred_scaled = self.model(tensor_pt).cpu().numpy()

        # 6. Target Inverse Transform
        pred_log = self.log_target_scaler.inverse_transform(pred_scaled)
        pred_orig = np.expm1(pred_log)[0]

        if not np.isfinite(pred_orig).all():
            raise ValueError("Model produced non-finite predictions.")

        # Get forecast issue date
        forecast_date = str(history_df.iloc[-1]["date"])[:10] if "date" in history_df.columns else str(datetime.now().date())

        provenance = self._resolve_input_provenance(input_provenance)

        return {
            "forecast_1d": float(pred_orig[0]),
            "forecast_3d": float(pred_orig[1]),
            "forecast_7d": float(pred_orig[2]),
            "model_version": MODEL_VERSION,
            "forecast_date": forecast_date,
            "source": "LIVE_INFERENCE",

            # ── Stage 5 provenance contract ──────────────────────────────
            "forecast_source": "FROZEN_LSTM_V3",
            "forecast_unit": FORECAST_UNITS,
            "horizons": list(HORIZONS),
            "target_columns": list(TARGET_COLUMNS),
            "feature_order": list(self.dynamic_features),
            "feature_units": {name: FEATURE_UNITS[name] for name in self.dynamic_features},
            "history_days": HISTORY_DAYS,
            **provenance,
        }
