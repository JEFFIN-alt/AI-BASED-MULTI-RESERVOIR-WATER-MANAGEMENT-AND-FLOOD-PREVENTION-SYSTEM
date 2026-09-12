import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import warnings
from datetime import datetime

# Filter out scikit-learn version mismatch warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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
        
        # Exact feature order required by V3 training pipeline
        self.dynamic_features = [
            "inflow",
            "water_level",
            "live_storage",
            "rainfall",
            "total_outflow",
        ]
        self.expected_cols = self.feature_scaler.feature_names_in_

    def predict(self, history_df: pd.DataFrame) -> dict:
        """
        Input: 7 consecutive historical observations.
        Columns required: water_level, live_storage, inflow, rainfall, total_outflow
        
        Output: Dict with forecast_1d, 3d, 7d and metadata.
        """
        # 1. Validation
        if len(history_df) != 7:
            raise ValueError(f"Expected exactly 7 rows of history, got {len(history_df)}.")
        
        required_cols = ["water_level", "live_storage", "inflow", "rainfall", "total_outflow"]
        for col in required_cols:
            if col not in history_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Check chronological ordering
        if "date" in history_df.columns:
            dates = pd.to_datetime(history_df["date"]).dt.date.tolist()
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days != 1:
                    raise ValueError("Dates in history_df are not consecutive.")
                    
        # Check finite values
        if not np.isfinite(history_df[required_cols].astype(float).values).all():
            raise ValueError("History contains NaN or infinite values.")

        # 2. Build flattened row for feature scaler
        flat_row = {}
        for feat in self.dynamic_features:
            for day_idx in range(1, 8): # day_1 to day_7
                # history_df index 0 is day_1 (oldest), index 6 is day_7 (latest)
                flat_row[f"{feat}_day_{day_idx}"] = float(history_df.iloc[day_idx - 1][feat])
        
        # Fill static features with 0 (StandardScaler scales column-independently, 
        # and the LSTM doesn't use the static features anyway, but we need 51 cols for the scaler)
        for col in self.expected_cols:
            if col not in flat_row:
                flat_row[col] = 0.0
                
        flat_df = pd.DataFrame([flat_row], columns=self.expected_cols)
        
        # 3. Scale features
        scaled_flat = self.feature_scaler.transform(flat_df)
        scaled_flat_df = pd.DataFrame(scaled_flat, columns=self.expected_cols)
        
        # 4. Build dynamic tensor (1, 7, 5)
        tensor = np.zeros((1, 7, len(self.dynamic_features)), dtype=np.float32)
        for feat_idx, feat in enumerate(self.dynamic_features):
            cols = [f"{feat}_day_{day_idx}" for day_idx in range(1, 8)]
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
        
        return {
            "forecast_1d": float(pred_orig[0]),
            "forecast_3d": float(pred_orig[1]),
            "forecast_7d": float(pred_orig[2]),
            "model_version": "LSTM_V3_LOGTARGET",
            "forecast_date": forecast_date,
            "source": "LIVE_INFERENCE"
        }
