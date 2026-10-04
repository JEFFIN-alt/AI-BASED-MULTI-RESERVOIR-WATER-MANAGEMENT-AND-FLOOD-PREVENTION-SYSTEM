import asyncio
import json
import logging
import math
from src.controller.storage_reserve import reserve_bounds, verify_reserve_action
import os
import time
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from src.dashboard.sim_bridge import SimBridge
from src.common import units
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
from src.dashboard.api.notifications import DownstreamNotificationManager
from src.modeling.inference import LiveForecaster, ForecastUnavailableError
from src.modeling.gnn_inference import LiveGNNForecaster
from src.modeling.gnn_advisory import (
    build_gnn_advisory,
    empty_block as empty_gnn_advisory,
)
from src.modeling.v3_feature_contract import (
    DEMO_PLACEHOLDERS,
    HISTORY_DAYS,
    ForecastStatus,
    build_live_feature_inputs,
    feature_provenance_summary,
    UNAVAILABLE_IN_LIVE_SIMULATION,
)
from src.network_env.gnn_forecast_adapter import GNNForecastAdapter
from src.network_env.live_forecast_adapter import LiveForecastAdapter
from src.network_env.v3_forecast_adapter import V3ForecastAdapter
from src.controller.live_mpc_orchestrator import (
    SAFETY_STATUS_NOT_APPLIED_ADAPTER_ERROR,
    ControllerStatus,
    LiveControlDecision,
    LiveMPCOrchestrator,
)
from src.controller.downstream_capacity_guard import (
    DOWNSTREAM_STATUS_NOT_APPLIED_ADAPTER_ERROR,
)

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent

#: STAGE 19 (P4) — module logger. Only used for the deadline-loop overrun
#: DEBUG line; no existing logging behaviour is changed anywhere by Stage 19.
logger = logging.getLogger(__name__)

CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

#: ── STAGE 9 — EXPLICIT FORECAST INVENTORY ───────────────────────────────────
#: Live reservoirs that the frozen LSTM V3 live pipeline does NOT forecast.
#: For these, ``LiveForecastAdapter`` emits an explicit ``FORECAST_UNAVAILABLE``
#: record (``issue = MISSING_FORECAST``); the value is never invented. The
#: Stage 7 provenance gate therefore blocks the coordinated four-reservoir MPC.
#:
#: This is a deliberate, named policy — not a silent ``continue`` — so the
#: absence of a D forecast is auditable rather than looking like legacy
#: "don't touch D" behaviour.
LIVE_FORECAST_EXCLUDED = ("Virtual Reservoir D",)

#: ── STAGE 18 — FORECAST SOURCE SELECTION (opt-in; default is unchanged) ──────
#: Which forecast source is allowed to feed the controller.
#:
#: ``SIMULATION`` (DEFAULT — the behaviour every existing test pins)
#:     The frozen LSTM V3 model run on the LIVE simulation's own state, using
#:     explicitly-labelled synthetic placeholders for the two features the live
#:     simulation cannot legitimately produce (``water_level`` in metres,
#:     ``rainfall`` in mm). Because those inputs are NOT real measurements,
#:     ``LiveForecaster._resolve_input_provenance`` correctly reports
#:     ``DEMONSTRATION_ONLY`` / ``validated_metrics_apply = False``, and Reservoir
#:     D has no live forecast at all (``LIVE_FORECAST_EXCLUDED``). The Stage 7
#:     provenance gate therefore BLOCKS the MPC, AUTO reports ``AUTO_BLOCKED``
#:     and the CURRENT gates are held. That is the honest live behaviour and it
#:     is deliberately NOT changed by this stage.
#:
#: ``VALIDATED_REPLAY`` (explicit operator selection)
#:     The frozen V3 model's own predictions on its held-out REAL historical test
#:     split, read READ-ONLY through ``V3ForecastAdapter`` from the SAME artifact
#:     ``scripts/run_phase15_3_validation.py`` consumed. The inputs to those
#:     predictions are real measurements, so the V3 evaluation metrics genuinely
#:     DO apply to them and the Stage 5 contract legitimately reports
#:     ``VALIDATED`` / ``validated_metrics_apply = True`` for all four reservoirs.
#:
#:     The provenance gate is NOT bypassed, relaxed or special-cased: it is
#:     genuinely satisfied, and the real MPCController -> SafetyLayer ->
#:     DownstreamCapacityGuard chain then produces the FINAL_SAFE_CONTROL_ACTION
#:     that moves the gates.
#:
#:     These are HISTORICAL replay forecasts, NOT forecasts of current live
#:     conditions. The payload states that explicitly
#:     (``replay_is_historical_not_live = True``) and the twin displays it.
FORECAST_SOURCE_SIMULATION = "SIMULATION"
FORECAST_SOURCE_VALIDATED_REPLAY = "VALIDATED_REPLAY"
FORECAST_SOURCES = (FORECAST_SOURCE_SIMULATION, FORECAST_SOURCE_VALIDATED_REPLAY)

#: ── STAGE 18 — AUTO CONTROL STATES ───────────────────────────────────────────
#: The state of backend AUTOMATIC control, derived ONLY from authoritative state.
#: The most specific applicable state wins:
#:
#:   AUTO_DISABLED         control mode is MANUAL — the backend is not deciding.
#:   AUTO_READY            AUTO armed, but no control decision has been made yet.
#:   AUTO_BLOCKED          the provenance gate refused the forecast; gates HELD.
#:   AUTO_ERROR            the controller could not run (UNAVAILABLE).
#:   AUTO_CONTROL_APPLIED  the controller ran and its safe action moved >=1 gate.
#:   AUTO_RUNNING          the controller ran, required no gate change, loop advancing.
#:   AUTO_HOLD             the controller ran, required no gate change, loop idle.
AUTO_DISABLED = "AUTO_DISABLED"
AUTO_READY = "AUTO_READY"
AUTO_RUNNING = "AUTO_RUNNING"
AUTO_CONTROL_APPLIED = "AUTO_CONTROL_APPLIED"
AUTO_HOLD = "AUTO_HOLD"
AUTO_BLOCKED = "AUTO_BLOCKED"
AUTO_ERROR = "AUTO_ERROR"

#: A gate move smaller than this (gate PERCENT) is not a control action worth
#: reporting; it is the backend's own deadband, not a UI rounding choice.
GATE_CHANGE_EPSILON_PCT = 1e-6

#: Bounded live event log length (real backend events only).
EVENT_LOG_MAXLEN = 250

# ── STAGE 4 — single authoritative simulation ────────────────────────────────
#: Every GlobalSimulationState constructed in THIS process registers itself here.
#: The FastAPI Digital Twin must have exactly one; the integration tests assert
#: it, so an accidental second live simulation is caught rather than silently
#: becoming a competing state producer.
_LIVE_INSTANCES: list = []


def authoritative_instance_count() -> int:
    """Number of live simulation instances constructed in this process."""
    return len(_LIVE_INSTANCES)


def get_authoritative_state_manager() -> "GlobalSimulationState":
    """
    Return THE one authoritative live simulation instance.

    This is the single object that (a) processes every command and (b) produces
    every state payload published over the WebSocket. See also
    ``src.dashboard.api.routes``, which imports the same ``sim_state``.
    """
    return sim_state


class GlobalSimulationState:
    def __init__(self):
        _LIVE_INSTANCES.append(self)
        self.bridge = SimBridge(str(CONFIG_PATH), str(THRESH_PATH))

        self.storm_intensity = 0.0
        self.mode = "MANUAL"
        self.running = False
        self.sim_speed = 1.0
        # ── STAGE 18 — which forecast source feeds the controller. DEFAULT is
        # the unchanged live-simulation source, so every existing provenance
        # guarantee (and the Stage 16 hard gate) still holds.
        self.forecast_source = FORECAST_SOURCE_SIMULATION
        #: The last REAL gate movement the controller caused, per reservoir:
        #: [{"node": ..., "previous_pct": ..., "new_pct": ..., "label": ...}].
        #: Written only from the authoritative decision, never from the UI.
        self.last_gate_transitions: list = []
        #: The REAL gates immediately before the last AUTO step (per node id).
        self.last_previous_gate_pct: dict = {}
        #: Risk transitions the backend itself detected on the most recent
        #: evaluated step. Used to explain WHY the controller acted, and never to
        #: invent a reason that did not occur.
        self.last_risk_transitions: list = []
        #: The backend's own event log (real events, real timestamps).
        self.event_log = deque(maxlen=EVENT_LOG_MAXLEN)
        #: Risk status per twin reservoir key, from the PREVIOUS authoritative
        #: step, so a NORMAL -> HIGH transition can be reported truthfully.
        self._last_risk = {}
        self._last_downstream_status = None
        self.notification_manager = DownstreamNotificationManager(on_update=self._schedule_notification_broadcast)
        self._notification_loop = None
        #: Lazily constructed READ-ONLY reader of the frozen V3 held-out
        #: prediction artifact (VALIDATED_REPLAY source only).
        self._v3_replay_adapter = None
        # Baseline inflows (MCM/day). MUST stay below each reservoir's
        # max_release_capacity_mcm_day (A:5, B:10, C:150, D:200) so that a
        # fully-open gate can actually drain the reservoir — including routed
        # inflow from upstream (A max release 5 → B local 4 + routed 5 = 9 < 10;
        # C local 90 + routed 10 = 100 < 150). At storm=0 the multipliers below
        # keep every total inflow under its release capacity.
        self.manual_inflows = {
            "Virtual Reservoir A": 3.0,
            "Virtual Reservoir B": 4.0,
            "Virtual Reservoir C": 90.0,
            "Virtual Reservoir D": 0.0
        }
        self.manual_gates = {
            "Virtual Reservoir A": 40.0,
            "Virtual Reservoir B": 35.0,
            "Virtual Reservoir C": 50.0,
            # ── STAGE 9 — THE LEGACY "D PINNED AT 100%" IS REMOVED ─────────
            # D (Idukki) used to be hardcoded at 100.0 (fully open) while the
            # other three baselines were 40/35/50 — AND it could not be changed
            # through the API, because ``POST /api/gate/{id}`` only accepted
            # ``reservoir_1..3``. That made D's gate a permanently fixed
            # decision. It is now an ordinary operator baseline like every other
            # reservoir, and it IS commandable (``reservoir_4``).
            #
            # This is a MANUAL-MODE operator command, not a controller output:
            # in AI mode all four values are replaced by the MPC + SafetyLayer
            # decision for the same four reservoirs.
            "Virtual Reservoir D": 50.0,
        }
        self.classroom_demo = False
        self.final_safety = {"checked": False}
        self.simulation_error = None
        self.clients = set()
        self.loop_task = None
        self._update_inflows()

        # ── STAGE 5 ────────────────────────────────────────────────────────
        # Number of AUTHORITATIVE simulation steps taken. The live forecast
        # buffer requires 7 GENUINELY DISTINCT steps; it is never seeded with
        # repeated copies of t=0 (that would be a fabricated 7-day history).
        self.sim_step_index = 0
        # ── STAGE 11 — LIVE MASS-BALANCE INTEGRITY ─────────────────────────
        # Whether the action that was WRITTEN to ReservoirNetwork is the
        # controller's FINAL_SAFE_CONTROL_ACTION (Stage 10's single eligible
        # record). Populated after every authoritative step; ``None`` means
        # "not checked", never "fine".
        self._applied_action_verification: dict = {
            "controller_action_checked": False,
            "matches_final_safe_control_action": None,
            "checked_action_source": None,
            "action_check_note": "NO_ACTION_APPLIED_YET",
        }

        # The live simulation cannot produce water_level (metres) or rainfall
        # (mm). See src/modeling/v3_feature_contract.py. By default the Digital
        # Twin demonstration uses EXPLICITLY LABELLED synthetic placeholders so
        # the 3D demo still renders a forecast. Set
        # AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS=0 to disable and have the live
        # forecast report FORECAST_UNAVAILABLE instead.
        self.allow_synthetic_demo_inputs = (
            os.environ.get("AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS", "1") != "0"
        )
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        self.res_mapping = {
            node_id: res_cfg["repository_derived_source"] 
            for node_id, res_cfg in config["reservoirs"].items()
        }
        
        # ── ML Integration ────────────────────────────────────────────────
        # STAGE 5: the VALIDATED LSTM path and the ADVISORY GNN path are
        # initialised INDEPENDENTLY. Previously a single try/except meant a
        # failing experimental GNN disabled the validated LSTM forecasts too.
        try:
            self.lstm_forecaster = LiveForecaster(str(_PROJECT_ROOT))
            self.lstm_ready = True
        except Exception as e:
            print(f"[StateManager] LSTM (validated) initialization failed: {e}")
            self.lstm_forecaster = None
            self.lstm_ready = False

        self.gnn_adapter = GNNForecastAdapter(self.res_mapping)
        #: STAGE 14 — the structured ADVISORY result. Displayed by the frontends,
        #: never consumed by the controller (which does not read this attribute).
        self.gnn_advisory = empty_gnn_advisory("NO_INFERENCE_YET")
        # ── STAGE 19 (P3) — per-authoritative-step memoisation of the DISPLAY
        # forecast/advisory work.
        #
        # `_run_ml_pipeline()` used to be recomputed on EVERY state read: once
        # inside `step()`'s own `get_adapted_state()`, once more in
        # `broadcast_state()`'s `get_adapted_state()`, and once per REST
        # `/api/state` read — even though the authoritative inputs had not
        # changed. The DISPLAY path is now memoised; see
        # `_display_pipeline_cache_key()` / `_display_forecasts()`.
        #
        # The CONTROL path is deliberately untouched: `step()` keeps calling
        # `_run_ml_pipeline()` directly so it can never consume stale display
        # data, and the Stage 7 provenance gate is enforced exactly as before.
        self._pipeline_cache_key = None
        self._pipeline_cache = None
        # ── STAGE 19 (P4) — deadline-loop overrun accounting (diagnostic only;
        # no physics, no controller timing semantics are read from these).
        self.loop_overrun_count: int = 0
        self.last_loop_overrun_s: float = 0.0
        self.max_loop_overrun_s: float = 0.0
        try:
            self.gnn_forecaster = LiveGNNForecaster(str(_PROJECT_ROOT))
            self.gnn_ready = True
        except Exception as e:
            print(f"[StateManager] GNN (advisory/experimental) initialization failed: {e}")
            self.gnn_forecaster = None
            self.gnn_ready = False

        #: Retained for backwards compatibility; the validated LSTM path is the
        #: one that matters for control.
        self.ml_ready = self.lstm_ready
            
        # ── STAGE 7 — ONE authoritative live controller path ───────────────
        # The validated Phase 15.3 MPC determines AI-mode gate decisions,
        # behind an explicit forecast-provenance gate.
        # ── STAGE 8 — the SafetyLayer IS integrated ────────────────────────
        # ``LiveMPCOrchestrator.decide()`` passes the MPC's raw proposal through
        # the EXISTING validated ``SafetyLayer`` and returns ITS output. The gate
        # commands applied below are therefore safety-validated, never raw MPC.
        self.mpc_orchestrator = LiveMPCOrchestrator()
        self.last_control_decision: Optional[LiveControlDecision] = None

        # Keep HISTORY_DAYS of simulated history for inference.
        # NOT pre-seeded: a 7-day window is only "available" once 7 genuinely
        # distinct simulation steps have actually been recorded.
        self.history_buffers = {v: [] for k, v in self.res_mapping.items()}

    # ------------------------------------------------------------------
    # STAGE 18 — CONTROL MODE, FORECAST SOURCE AND THE LIVE EVENT LOG
    # ------------------------------------------------------------------

    def log_event(self, kind: str, text: str, **extra) -> dict:
        """
        Append ONE real backend event, timestamped by the backend.

        Callers must pass something the backend actually observed (a mode change
        it performed, a transition it measured). Nothing here is invented, and no
        controller action is ever logged that did not happen.
        """
        event = {
            "seq": (self.event_log[-1]["seq"] + 1) if self.event_log else 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "kind": str(kind),
            "text": str(text),
        }
        for key, value in extra.items():
            if value is not None:
                event[key] = value
        self.event_log.append(event)
        return event

    def event_log_payload(self) -> list:
        """The bounded window of real events, oldest first."""
        return [dict(e) for e in self.event_log]

    def set_mode(self, mode: str) -> None:
        """
        Set MANUAL/AI and record the change as a real event.

        AUTO -> MANUAL handover: the operator baseline is set to the gates the
        controller is CURRENTLY holding, so control is handed over continuously
        instead of the gates snapping back to whatever baseline was stored
        before AUTO took over. The authoritative state itself (storages, flows,
        routing) is never touched, so nothing else about MANUAL mode changes.
        """
        mode = str(mode).upper()
        if mode == self.mode:
            return
        previous = self.mode
        self.mode = mode
        # STAGE 19 (P3) — a mode change re-arms the display pipeline. The
        # authoritative inputs are still fingerprinted, but the invalidation is
        # explicit so a stale advisory can never be shown after a handover.
        self._invalidate_pipeline_cache()
        if mode == "MANUAL" and previous == "AI":
            try:
                self.manual_gates.update(self._current_gate_pct())
            except Exception:  # pragma: no cover - network not ready
                pass
        if mode == "AI":
            self.log_event("auto_enabled",
                           f"AUTO control ENABLED (was {previous})",
                           previous_mode=previous, mode=mode)
        else:
            self.log_event("auto_disabled",
                           f"AUTO control DISABLED — MANUAL (was {previous})",
                           previous_mode=previous, mode=mode)

    def set_forecast_source(self, source: str) -> None:
        """Select the forecast source and record it as a real event."""
        source = str(source).upper()
        if source not in FORECAST_SOURCES:
            raise ValueError(f"unknown forecast source: {source!r}")
        previous = self.forecast_source
        self.forecast_source = source
        # STAGE 19 (P3) — a source change is a different pipeline entirely
        # (replay CSV vs live LSTM + GNN); never serve the old one.
        self._invalidate_pipeline_cache()
        if source != previous:
            self.log_event("forecast_source",
                           f"Forecast source: {previous} -> {source}",
                           previous_source=previous, forecast_source=source)

    #: The frozen V3 artifact holding the model's HELD-OUT test predictions.
    V3_PREDICTIONS_ARTIFACT = (
        "results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv"
    )

    def _replay_adapter(self) -> V3ForecastAdapter:
        """The shared, READ-ONLY V3 artifact reader (constructed once)."""
        if self._v3_replay_adapter is None:
            self._v3_replay_adapter = V3ForecastAdapter(
                str(_PROJECT_ROOT), artifact_path=self.V3_PREDICTIONS_ARTIFACT
            )
        return self._v3_replay_adapter

    def _replay_date(self, adapter: V3ForecastAdapter) -> Optional[str]:
        """One historical calendar day per simulated day; never repeat a stale issue."""
        dates = sorted(str(d) for d in adapter.available_dates)
        if not dates:
            return None
        date = (pd.Timestamp(dates[0]) + pd.Timedelta(days=self.bridge.cascade.network.timestep)).strftime("%Y-%m-%d")
        if date not in dates:
            return None
        snapshot = adapter.get_network_snapshot(date, node_ids=list(adapter.reservoir_mapping))
        return date if all(all(fc.is_available(h) for h in ("1d", "3d", "7d"))
                           for fc in snapshot.forecasts.values()) else None

    @staticmethod
    def _replay_unavailable(reason: str, note: str, live_names) -> dict:
        """Explicit UNAVAILABLE payload for every reservoir — never a substitute."""
        return {
            live: {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.UNAVAILABLE.value,
                "forecast_provenance": reason,
                "is_simulated": False,
                "validated_metrics_apply": False,
                "unavailable_features": ["validated_replay_artifact"],
                "note": note,
            }
            for live in live_names
        }

    def _validated_replay_forecasts(self) -> dict:
        """
        STAGE 18 — controller forecasts from the frozen V3 model's predictions on
        its HELD-OUT REAL historical test split.

        Read READ-ONLY through the SAME ``V3ForecastAdapter`` the Phase 15.3
        validation script uses; neither the adapter nor the artifact is modified.

        The declared provenance is the honest one for this source: the INPUTS to
        those predictions are real measurements, so the V3 evaluation metrics DO
        apply (``validated_metrics_apply = True``) and the declared status is
        ``VALIDATED``. That is what makes the Stage 7 provenance gate genuinely
        SATISFIABLE for this source, rather than bypassed.

        Reservoir D is included like every other reservoir. Anything unavailable
        is reported UNAVAILABLE — never zero-filled, averaged or carried forward.
        """
        live_names = list(self.res_mapping)
        try:
            adapter = self._replay_adapter()
            date = self._replay_date(adapter)
        except Exception as exc:  # artifact missing / unreadable
            return self._replay_unavailable(
                f"REPLAY_SOURCE_ERROR:{type(exc).__name__}",
                "The frozen V3 held-out prediction artifact could not be read. "
                "Nothing was substituted.",
                live_names,
            )

        if date is None:
            return self._replay_unavailable(
                "REPLAY_ARTIFACT_INCOMPLETE",
                "No date in the frozen V3 artifact carries all three horizons for "
                "all four reservoirs. Nothing was substituted.",
                live_names,
            )

        node_by_repo = {v: k for k, v in adapter.reservoir_mapping.items()}
        out: dict = {}
        for live_name, repo_name in self.res_mapping.items():
            node_id = node_by_repo.get(repo_name)
            record = adapter.get_forecast(node_id, date) if node_id else None
            if record is None or not all(
                record.is_available(h) for h in ("1d", "3d", "7d")
            ):
                out[live_name] = {
                    "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                    "forecast_source": "FROZEN_LSTM_V3",
                    "forecast_status": ForecastStatus.UNAVAILABLE.value,
                    "forecast_provenance": "REPLAY_HORIZON_UNAVAILABLE",
                    "is_simulated": False,
                    "validated_metrics_apply": False,
                    "unavailable_features": ["validated_replay_horizons"],
                }
                continue
            out[live_name] = {
                "forecast_1d": record.target_1d,
                "forecast_3d": record.target_3d,
                "forecast_7d": record.target_7d,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.VALIDATED.value,
                "forecast_provenance": "MODEL_PREDICTION_ON_HELD_OUT_HISTORICAL_DATA",
                "is_simulated": False,
                "validated_metrics_apply": True,
                "forecast_unit": "MCM/day",
                "horizons": ["forecast_1d", "forecast_3d", "forecast_7d"],
                "forecast_date": date,
                "replay_is_historical_not_live": True,
                "replay_artifact": self.V3_PREDICTIONS_ARTIFACT,
                "v3_reservoir_name": record.v3_reservoir_name,
                "note": (
                    "Frozen LSTM V3 prediction on HELD-OUT REAL historical data. "
                    "The inputs are real measurements, so the reported V3 "
                    "evaluation metrics apply. This is a HISTORICAL REPLAY, not a "
                    "forecast of current live conditions."
                ),
            }
        return out

    def _record_history(self):
        """
        Record ONE authoritative simulation step into the 7-day rolling window.

        STAGE 5 — only features the authoritative ReservoirNetwork can
        legitimately produce are recorded:

            inflow        <- ReservoirState.inflow_local    (MCM/day)
            live_storage  <- ReservoirState.storage         (MCM)
            total_outflow <- ReservoirState.total_outflow   (MCM/day)

        ``water_level`` (metres) and ``rainfall`` (mm) are deliberately NOT
        recorded here: they are produced (or not) at forecast time by
        ``_forecast_one_reservoir``, so that their provenance is explicit.

        Previously this method fed ``manual_inflows`` (a *commanded* baseline)
        as ``inflow``, a storage PERCENTAGE as ``water_level`` (a slot trained
        on metres), a hard 0.0 as ``rainfall``, and fabricated 50.0 defaults
        when a reservoir was missing. All of that is gone.
        """
        self.sim_step_index += 1
        for v_name, p_name in self.res_mapping.items():
            res_obj = self.bridge.cascade.reservoirs.get(v_name)
            if res_obj is None:
                continue
            state = res_obj.state
            self.history_buffers[p_name].append({
                "step_index": self.sim_step_index,
                "inflow": float(state.inflow_mcm_day),
                "live_storage": float(state.storage_mcm),
                "total_outflow": float(state.release_mcm_day),
            })
            if len(self.history_buffers[p_name]) > HISTORY_DAYS:
                self.history_buffers[p_name].pop(0)

    def _update_inflows(self):
        if getattr(self, "classroom_demo", False):
            self.manual_inflows = dict(zip(self.bridge.cascade.network.processing_order, [1.0, .5, 8.0, 0.0]))
            return
        # Storm multiplier scales baseline inflow up to 4x (1 + 3*storm).
        # Baselines are chosen so that at storm=0 every inflow is below the
        # reservoir's max release capacity — full gate can then drain it.
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            default_inflow = 3.0 if "A" in res else (4.0 if "B" in res else 90.0)
            self.manual_inflows[res] = default_inflow * (1.0 + (self.storm_intensity * 3.0))

    # ------------------------------------------------------------------
    # STAGE 5 — live forecast construction with explicit provenance
    # ------------------------------------------------------------------

    def _live_feature_inputs(self, v_name: str, p_name: str) -> dict:
        """Provenance record for the 5 frozen V3 features, from the LATEST state."""
        res_obj = self.bridge.cascade.reservoirs.get(v_name)
        state = res_obj.state if res_obj else None
        return build_live_feature_inputs(
            inflow_local=None if state is None else state.inflow_mcm_day,
            storage=None if state is None else state.storage_mcm,
            total_outflow=None if state is None else state.release_mcm_day,
            mapped_reservoir=p_name,
            allow_synthetic_demo=self.allow_synthetic_demo_inputs,
        )

    def _feature_frame(self, p_name: str, inputs: dict):
        """
        Build the 7-row HISTORY_DAYS window in the FROZEN feature order.

        The three simulation-derived features come from the recorded history.
        ``water_level`` / ``rainfall`` are constant across the window (they are
        placeholders, so they carry no simulated dynamics) and come from the
        provenance record — never from a conversion of storage.
        """
        buf = self.history_buffers.get(p_name, [])
        water_level = inputs["water_level"].value
        rainfall = inputs["rainfall"].value
        rows = []
        for r in buf:
            rows.append({
                "inflow": r["inflow"],
                "water_level": water_level,
                "live_storage": r["live_storage"],
                "rainfall": rainfall,
                "total_outflow": r["total_outflow"],
            })
        df = pd.DataFrame(rows)
        df["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(rows)).strftime("%Y-%m-%d")
        return df

    def _forecast_one_reservoir(self, v_name: str, p_name: str) -> dict:
        """Produce one forecast with an explicit, unambiguous provenance record."""
        buf = self.history_buffers.get(p_name, [])
        distinct_steps = len({r["step_index"] for r in buf})

        # 1. Warm-up: we require HISTORY_DAYS GENUINELY DISTINCT simulation
        #    steps. We never fabricate a history by repeating t=0.
        if len(buf) < HISTORY_DAYS or distinct_steps < HISTORY_DAYS:
            return {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.WARMUP.value,
                "forecast_provenance": "INSUFFICIENT_HISTORY",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "steps_collected": distinct_steps,
                "steps_required": HISTORY_DAYS,
                "note": (
                    "Fewer than 7 distinct authoritative simulation steps have been "
                    "recorded. No history is fabricated to fill the gap."
                ),
            }

        inputs = self._live_feature_inputs(v_name, p_name)
        summary = feature_provenance_summary(inputs)

        # 2. A required physical quantity that cannot be produced legitimately.
        if summary["unavailable"]:
            return {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.UNAVAILABLE.value,
                "forecast_provenance": "REQUIRED_FEATURE_UNAVAILABLE",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "unavailable_features": summary["unavailable"],
                "unavailable_reasons": {
                    f: UNAVAILABLE_IN_LIVE_SIMULATION.get(f, "unavailable")
                    for f in summary["unavailable"]
                },
                "input_provenance": summary,
                "note": (
                    "The live simulation cannot legitimately produce the listed "
                    "feature(s). Nothing is substituted."
                ),
            }

        # 3. Validated path unavailable (model could not be loaded).
        if not self.lstm_ready:
            return {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.UNAVAILABLE.value,
                "forecast_provenance": "MODEL_NOT_LOADED",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "input_provenance": summary,
            }

        df = self._feature_frame(p_name, inputs)
        try:
            fc = self.lstm_forecaster.predict(df, input_provenance=summary)
        except ForecastUnavailableError as exc:
            return {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.UNAVAILABLE.value,
                "forecast_provenance": "REQUIRED_FEATURE_UNAVAILABLE",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "unavailable_reasons": {"error": str(exc)},
                "input_provenance": summary,
            }
        except Exception as exc:  # non-finite history, scaling failure, ...
            return {
                "forecast_1d": None, "forecast_3d": None, "forecast_7d": None,
                "forecast_source": "FROZEN_LSTM_V3",
                "forecast_status": ForecastStatus.UNAVAILABLE.value,
                "forecast_provenance": "INFERENCE_ERROR",
                "is_simulated": True,
                "validated_metrics_apply": False,
                "unavailable_reasons": {"error": f"{type(exc).__name__}: {exc}"},
                "input_provenance": summary,
            }
        return fc

    def _build_gnn_history(self) -> dict:
        """
        Build SCALED (7, 5) windows for the ADVISORY GNN.

        The GNN's documented input contract requires already-scaled features in
        FEATURE_ORDER; the previous live path passed raw (unscaled) values.
        This only feeds the experimental advisory model — never control.
        """
        if self.lstm_forecaster is None:
            return {}
        history = {}
        for v_name, p_name in self.res_mapping.items():
            buf = self.history_buffers.get(p_name, [])
            if len({r["step_index"] for r in buf}) < HISTORY_DAYS:
                continue
            inputs = self._live_feature_inputs(v_name, p_name)
            if feature_provenance_summary(inputs)["unavailable"]:
                continue
            try:
                frame = self._feature_frame(p_name, inputs)
                flat = {}
                for feat in self.lstm_forecaster.dynamic_features:
                    for i in range(HISTORY_DAYS):
                        flat[f"{feat}_day_{i + 1}"] = float(frame.iloc[i][feat])
                for col in self.lstm_forecaster.expected_cols:
                    flat.setdefault(col, 0.0)
                ordered = pd.DataFrame([flat], columns=self.lstm_forecaster.expected_cols)
                scaled = self.lstm_forecaster.feature_scaler.transform(ordered)
                window = scaled[0, : HISTORY_DAYS * len(self.lstm_forecaster.dynamic_features)]
                # Scaler columns are feature-major: each feature contains seven days.
                history[p_name] = window.reshape(
                    len(self.lstm_forecaster.dynamic_features), HISTORY_DAYS
                ).T.astype(np.float32)
            except Exception:
                continue
        return history

    def _display_pipeline_cache_key(self) -> tuple:
        """
        STAGE 19 (P3) — fingerprint of every authoritative input the DISPLAY
        forecast path depends on.

        The key deliberately includes MORE than the suggested triple
        (``sim_step_index``, ``forecast_source``, ``storm_intensity``):

        * the `ReservoirNetwork` OBJECT (identity comparison, `is`) — `reset()`
          rebuilds the cascade while `sim_step_index` keeps counting, and
          storing the object here keeps a strong reference so a freed object's
          address can never be recycled into a false hit;
        * `network.timestep` — the physics advanced;
        * the gate vector and the storage vector — four floats each; including
          them makes the fingerprint robust against any gate-only or
          storage-only change between reads (belt and braces: including them
          can only cause an extra recomputation, never a stale read).

        `_run_ml_pipeline()` reads no other authoritative mutable state
        (history buffers advance only via `step()`, which advances
        `sim_step_index` first).

        STAGE 19 (P3 fix) — the MODEL-AVAILABILITY inputs are part of the key
        too. `_run_ml_pipeline()` branches on `gnn_ready` / `lstm_ready` and
        calls the loaded forecaster objects, so toggling a model (as the GNN
        A/B inertness tests do) is an input change: without these the memo
        would serve the previous advisory (`AVAILABLE` where the pipeline
        currently produces `UNAVAILABLE`). Objects are compared by identity.
        """
        network = self.bridge.cascade.network
        return (
            network,
            int(self.sim_step_index),
            str(self.forecast_source),
            round(float(self.storm_intensity), 6),
            int(network.timestep),
            tuple(round(float(network.nodes[nid].state.gate_position), 6)
                  for nid in network.processing_order),
            tuple(round(float(network.nodes[nid].state.storage), 9)
                  for nid in network.processing_order),
            bool(self.gnn_ready),
            self.gnn_forecaster,
            bool(self.lstm_ready),
            self.lstm_forecaster,
        )

    def _display_forecasts(self) -> dict:
        """
        STAGE 19 (P3) — control forecasts for DISPLAY/state adaptation only.

        Memoised per `_display_pipeline_cache_key()`; recomputed whenever ANY
        authoritative input changed. The CONTROL path in `step()` calls
        `_run_ml_pipeline()` directly and never reads this cache.
        """
        key = self._display_pipeline_cache_key()
        if self._pipeline_cache_key == key and self._pipeline_cache is not None:
            return self._pipeline_cache
        computed = self._run_ml_pipeline()
        self._pipeline_cache_key = key
        self._pipeline_cache = computed
        return computed

    def _invalidate_pipeline_cache(self) -> None:
        """Drop the display-pipeline memoisation (reset / mode / source)."""
        self._pipeline_cache_key = None
        self._pipeline_cache = None

    def _run_ml_pipeline(self) -> dict:
        """
        Produce control forecasts.

        CONTROL POLICY: FROZEN LSTM V3 ONLY. The experimental GNN is computed
        separately for ADVISORY display and is DELIBERATELY not used for control
        (Req. Stage 5 / 20).

        RESERVOIR INVENTORY (STAGE 9)
        -----------------------------
        The live pipeline forecasts the reservoirs that have a live feature
        source. Reservoir D (Idukki) is deliberately NOT skipped silently: it is
        excluded here and therefore reaches the controller as an EXPLICIT
        ``FORECAST_UNAVAILABLE`` entry (``issue = MISSING_FORECAST``) produced by
        ``LiveForecastAdapter``. It is never zero-filled, averaged, carried
        forward or otherwise fabricated, and the Stage 7 provenance gate blocks
        the coordinated MPC because of it. See ``LIVE_FORECAST_EXCLUDED``.
        """
        if self.forecast_source == FORECAST_SOURCE_VALIDATED_REPLAY:
            # ── STAGE 18 — explicitly-selected VALIDATED source ──────────────
            # The frozen V3 model's predictions on its held-out REAL historical
            # test split, read-only, for ALL FOUR reservoirs (D included). This
            # source genuinely satisfies the Stage 7 provenance gate — the gate
            # is not bypassed, relaxed or special-cased.
            lstm_forecasts = self._validated_replay_forecasts()
        else:
            # ── DEFAULT — unchanged live behaviour ───────────────────────────
            # The frozen LSTM V3 model run on THIS simulation's own state, with
            # Reservoir D deliberately absent (explicit FORECAST_UNAVAILABLE,
            # never fabricated).
            lstm_forecasts = {}
            for v_name, p_name in self.res_mapping.items():
                if v_name in LIVE_FORECAST_EXCLUDED:
                    continue
                lstm_forecasts[v_name] = self._forecast_one_reservoir(v_name, p_name)

        # Advisory/experimental GNN — never feeds the control path.
        # STAGE 14 — the inference result is now ADDRESSED as a structured
        # advisory block that reaches the API/UI for display. It is stored on a
        # separate attribute that the controller never reads.
        if self.gnn_ready:
            try:
                gnn_history = self._build_gnn_history()
                if gnn_history:
                    gnn_result = self.gnn_forecaster.predict_all(gnn_history)
                    self.gnn_adapter.update_from_inference(
                        gnn_result,
                        inference_time_ms=self.gnn_forecaster.last_inference_time_ms,
                        gate_value=self.gnn_forecaster.gate_value,
                    )
                    # Learned per-node representations (advisory spatial context).
                    # These reproduce the model's own fused hidden state; they are
                    # not hydraulic, causal or control information.
                    embeddings, gate = self.gnn_forecaster.node_representations(gnn_history)
                    self.gnn_advisory = build_gnn_advisory(
                        embeddings=embeddings,
                        gate_value=gate,
                        live_input_nodes=sorted(gnn_history),
                        graph_provenance=self.gnn_forecaster.graph_provenance(),
                        inference_latency_ms=self.gnn_forecaster.last_inference_time_ms,
                    )
                else:
                    self.gnn_advisory = empty_gnn_advisory("INSUFFICIENT_HISTORY_FOR_GNN")
            except Exception as e:
                print(f"[StateManager] GNN advisory inference failed: {e}")
                self.gnn_advisory = empty_gnn_advisory(
                    f"INFERENCE_ERROR:{type(e).__name__}"
                )
        else:
            self.gnn_advisory = empty_gnn_advisory("MODEL_NOT_LOADED")

        # Production control policy (validated LSTM V3 baseline).
        ctrl_forecasts = self.gnn_adapter.get_control_forecasts(lstm_forecasts, policy="lstm_primary")
        # STAGE 14 — FAIL-CLOSED proof that the advisory cannot leak into control.
        self._assert_control_forecasts_are_lstm_only(ctrl_forecasts, lstm_forecasts)

        # Carry the Stage 5 provenance through to the control/display payload.
        for name, fc in lstm_forecasts.items():
            if name in ctrl_forecasts:
                for key in ("forecast_status", "forecast_provenance", "forecast_source",
                            "is_simulated", "validated_metrics_apply",
                            "unavailable_features", "unavailable_reasons",
                            "steps_collected", "note", "input_provenance",
                            "forecast_unit", "horizons", "target_columns",
                            "model_version", "feature_order", "feature_units",
                            "history_days", "forecast_date",
                            # STAGE 18 — honest labelling of the replay source
                            "replay_is_historical_not_live", "replay_artifact",
                            "v3_reservoir_name"):
                    if key in fc:
                        ctrl_forecasts[name][key] = fc[key]
        return ctrl_forecasts

    # ------------------------------------------------------------------
    # STAGE 7 — the ONE authoritative live controller path
    # ------------------------------------------------------------------

    #: Forecast horizons that the controller consumes.
    _CONTROL_HORIZONS = ("forecast_1d", "forecast_3d", "forecast_7d")

    def _assert_control_forecasts_are_lstm_only(
        self,
        ctrl_forecasts: dict,
        lstm_forecasts: dict,
    ) -> None:
        """
        STAGE 14 — the advisory boundary, enforced fail-closed.

        The GNN adapter is able to build control forecasts under other,
        unvalidated policies; those paths exist for offline research and have
        never been closed-loop validated. This guard re-reads every horizon the
        controller is about to consume and refuses the cycle unless each one is
        the validated frozen-LSTM value.

        Failing loudly is deliberate: silently controlling with an unvalidated
        forecast source would be a far worse outcome than a refused step, and
        this is the mechanism that makes "the GNN cannot reach control" a
        property of the running system rather than a reading of its source.
        """
        for name, fc in ctrl_forecasts.items():
            reference = lstm_forecasts.get(name)
            if reference is None:
                continue
            for horizon in self._CONTROL_HORIZONS:
                used = fc.get(horizon)
                validated = reference.get(horizon)
                if used is None and validated is None:
                    continue
                if used != validated:
                    raise RuntimeError(
                        "CONTROL FORECAST NOT FROM THE VALIDATED MODEL: "
                        f"{name}.{horizon} = {used!r} but the frozen LSTM V3 "
                        f"forecast is {validated!r}. The experimental GNN is "
                        "advisory only; refusing to control with an "
                        "unvalidated forecast source."
                    )

    def _forecast_date(self) -> str:
        """Issue date used for the live forecast snapshot."""
        return str(pd.Timestamp.now().date())

    def _build_live_forecast_bundle(self, ctrl_forecasts: dict):
        """
        Adapt the live forecast payload into the validated MPC contract.

        The adapter is bound to the LIVE network's node ids, because the MPC
        looks forecasts up by the node ids of the network it is asked to control.
        """
        network = self.bridge.cascade.network
        adapter = LiveForecastAdapter.for_network(network, project_root=str(_PROJECT_ROOT))
        issue_dates = {v.get("forecast_date") for v in ctrl_forecasts.values()
                       if isinstance(v, dict) and v.get("forecast_date")}
        if len(issue_dates) > 1:
            raise ValueError("Mixed forecast issue dates")
        return adapter.build_bundle(ctrl_forecasts, next(iter(issue_dates), self._forecast_date()))

    def _apply_ai_control(self, ctrl_forecasts: dict, gate_commands: dict) -> dict:
        """
        Apply the authoritative MPC decision in AI mode.

        The validated ``ForecastAwareController`` rule-based advisor is NOT used
        here any more — it is no longer the live controller (Stage 7).

        If the provenance gate blocks the forecast, the MPC is NOT invoked and
        the current gates are held; ``control_applied`` is False and the reason
        is reported explicitly. Nothing is fabricated.

        STAGE 8 — ``decision.gate_positions_pct`` holds the **SafetyLayer's**
        output (the MPC's raw proposal having been passed through the existing
        validated layer inside the orchestrator). The SafetyLayer is therefore
        upstream of every gate command applied to ``ReservoirNetwork``, and no
        other code path may write gate commands in AI mode.
        """
        try:
            bundle = self._build_live_forecast_bundle(ctrl_forecasts)
        except Exception as exc:
            gate_commands = self._current_gate_pct()
            decision = LiveControlDecision(
                controller_status=ControllerStatus.BLOCKED.value,
                forecast_control_eligible=False,
                blocked_reason=f"FORECAST_ADAPTER_ERROR:{type(exc).__name__}",
                reasons=[str(exc)],
                safety_layer_status=SAFETY_STATUS_NOT_APPLIED_ADAPTER_ERROR,
                safety_is_safe=False,
                # Stage 10 — no action was produced, so the downstream-capacity
                # boundary did not run and does not fabricate one.
                downstream_status=DOWNSTREAM_STATUS_NOT_APPLIED_ADAPTER_ERROR,
                downstream_reason=(
                    "the live forecast adapter raised; no action was produced and "
                    "no downstream evaluation was performed"
                ),
                final_safe_control_action_pct=dict(gate_commands),
                final_safe_control_action_source="HELD_CURRENT_GATES",
                gate_positions_pct=dict(gate_commands),
                control_applied=False,
                mpc_status="NOT_INVOKED",
            )
            self.last_control_decision = decision
            return gate_commands

        decision = self.mpc_orchestrator.decide(
            self.bridge.cascade.network,
            bundle=bundle,
            # STAGE 10 — pass the exogenous inflows that will actually be applied
            # in this step (``self.bridge.step(self.manual_inflows, ...)`` below).
            # The downstream-capacity boundary predicts with these, so its
            # guarantee is about the step that really happens rather than about
            # the previous step's inflow trace.
            current_inflows=dict(self.manual_inflows),
        )
        self.last_control_decision = decision

        # Apply whatever the orchestrator returned (MPC gates when active, the
        # held current gates when blocked). Never a fabricated value.
        for name, pct in decision.gate_positions_pct.items():
            if name in gate_commands:
                gate_commands[name] = pct
        return gate_commands

    # ------------------------------------------------------------------
    # STAGE 18 — AUTO CONTROL: real gates, real transitions, real states
    # ------------------------------------------------------------------

    def _current_gate_pct(self) -> dict:
        """
        The REAL gate positions of the authoritative network, in EXTERNAL
        percent.

        ``ReservoirState.gate_position`` is the canonical FRACTION in [0, 1];
        the percent conversion goes through the single unit boundary
        (``src/common/units``) rather than an inline ``* 100``.
        """
        network = self.bridge.cascade.network
        return {
            str(nid): float(units.gate_fraction_to_percent(
                network.nodes[nid].state.gate_position
            ))
            for nid in network.processing_order
        }

    @staticmethod
    def _gate_transitions(before_pct: dict, after_pct: dict) -> list:
        """
        The gates the controller ACTUALLY moved, measured before/after.

        A move below ``GATE_CHANGE_EPSILON_PCT`` is not a control action. Sorted
        into network order so the payload is deterministic.
        """
        out = []
        for nid in after_pct:
            if nid not in before_pct:
                continue
            previous = float(before_pct[nid])
            new = float(after_pct[nid])
            if abs(new - previous) > GATE_CHANGE_EPSILON_PCT:
                out.append({
                    "node": str(nid),
                    "previous_pct": previous,
                    "new_pct": new,
                    "delta_pct": new - previous,
                })
        return out

    #: Risk severity order used ONLY to read the direction of a transition the
    #: backend already classified. The twin never re-classifies risk itself.
    _RISK_ORDER = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}

    #: MPC proposal / safety output / final action, as percent, None-safe.
    @staticmethod
    def _pct_or_none(values, node_ids) -> dict:
        out = {}
        for nid in node_ids:
            if nid not in values:
                continue
            raw = values[nid]
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            out[str(nid)] = number
        return out

    def _auto_control_block(self) -> dict:
        """
        STAGE 18 — the backend's own answer to "what is AUTO doing right now?".

        Derived ONLY from the authoritative control mode, the last REAL control
        decision and the REAL gate transitions. No reason is invented and no
        status is embellished: every string originates in the controller's own
        provenance (``blocked_reason``, ``reasons``, the layer statuses), and the
        MPC proposal / SafetyLayer output / FINAL_SAFE_CONTROL_ACTION are the
        controller's own recorded values.
        """
        decision = self.last_control_decision
        ctl = self.mpc_orchestrator.status_dict()
        cstatus = str(getattr(decision, "controller_status", "")
                      or ctl.get("controller_status") or "")
        transitions = [dict(t) for t in (self.last_gate_transitions or [])]
        risk_transitions = [dict(r) for r in (self.last_risk_transitions or [])]
        escalated = [
            r for r in risk_transitions
            if self._RISK_ORDER.get(str(r.get("risk")), 0)
            > self._RISK_ORDER.get(str(r.get("previous_risk")), 0)
        ]

        if self.mode != "AI":
            state, reason = AUTO_DISABLED, (
                "Control mode is MANUAL — the backend is not deciding gate positions."
            )
        elif decision is None:
            state, reason = AUTO_READY, (
                "AUTO armed. No control decision has been made yet."
            )
        elif cstatus == ControllerStatus.BLOCKED.value:
            state, reason = AUTO_BLOCKED, (
                str(getattr(decision, "blocked_reason", "")
                    or "FORECAST_NOT_ELIGIBLE_FOR_CONTROL")
            )
        elif cstatus == ControllerStatus.UNAVAILABLE.value:
            state, reason = AUTO_ERROR, (
                str(getattr(decision, "blocked_reason", "") or "CONTROLLER_UNAVAILABLE")
            )
        elif transitions:
            state = AUTO_CONTROL_APPLIED
            # Give the ACTUAL reason the backend has evidence for, rather than a
            # generic sentence: if the backend detected a risk escalation, say so.
            if escalated:
                reason = "Reservoir risk increased: " + ", ".join(
                    f"{r['reservoir']} {r['previous_risk']} -> {r['risk']}"
                    for r in escalated
                )
            else:
                reason = "Control action applied to the authoritative reservoir network."
        elif self.running:
            state, reason = AUTO_RUNNING, "No control action required this step."
        else:
            state, reason = AUTO_HOLD, (
                "No control action required; the simulation loop is idle."
            )

        network = self.bridge.cascade.network
        node_ids = [str(n) for n in network.processing_order]
        provenance = getattr(decision, "forecast_provenance", {}) or {}
        nodes = provenance.get("nodes", {}) if isinstance(provenance, dict) else {}
        forecast_validated = bool(nodes) and all(
            (n or {}).get("declared_status") == "VALIDATED" for n in nodes.values()
        )

        proposed = self._pct_or_none(
            {k: (None if v is None else v * 100.0)
             for k, v in (getattr(decision, "proposed_gate_positions_fraction", {}) or {}).items()},
            node_ids,
        )
        safety_pct = self._pct_or_none(
            getattr(decision, "safety_layer_gate_positions_pct", {}) or {}, node_ids
        )
        final_pct = self._pct_or_none(
            getattr(decision, "final_safe_control_action_pct", {}) or {}, node_ids
        )
        previous = {str(k): float(v) for k, v in (self.last_previous_gate_pct or {}).items()}

        actions = [
            {
                "node": nid,
                "previous_pct": previous.get(nid),
                "mpc_proposal_pct": proposed.get(nid),
                "safety_pct": safety_pct.get(nid),
                "final_pct": final_pct.get(nid),
            }
            for nid in node_ids
        ]

        return {
            "state": state,
            "reason": reason,
            "control_mode": self.mode,
            "auto_enabled": self.mode == "AI",
            "loop_running": bool(self.running),
            "controller_type": str(getattr(decision, "controller_type", "MPC")),
            "controller_status": cstatus or "UNKNOWN",
            "forecast_control_eligible": bool(
                getattr(decision, "forecast_control_eligible", False)
            ),
            "forecast_validated": forecast_validated,
            "control_applied": bool(getattr(decision, "control_applied", False)),
            "safety_layer_status": str(
                getattr(decision, "safety_layer_status", "UNKNOWN")
            ),
            "safety_modified_by_controller": bool(getattr(decision, "safety_modified", False)),
            "downstream_status": str(getattr(decision, "downstream_status", "UNKNOWN")),
            "downstream_protection_modified": bool(
                getattr(decision, "downstream_protection_modified", False)
            ),
            "final_safe_control_action_source": str(
                getattr(decision, "final_safe_control_action_source", "NOT_APPLIED")
            ),
            "forecast_source": self.forecast_source,
            "mpc_status": str(getattr(decision, "mpc_status", "")),
            "gate_transitions": transitions,
            "risk_transitions": risk_transitions,
            "actions": actions,
            "reasons": list(getattr(decision, "reasons", []) or []),
            "controller_provenance": {
                "forecast_date": provenance.get("forecast_date"),
                "rule": provenance.get("rule"),
                "reason_strings": list(provenance.get("reason_strings", []) or []),
            },
        }

    def _record_control_events(self, state: dict) -> None:
        """
        STAGE 18 — emit REAL events for one authoritative step.

        Every event is derived from data the backend itself produced: the
        controller's own decision provenance, the gate transitions measured on
        the authoritative network, and the risk/flow classifications the backend
        computed. Nothing is emitted for something that did not happen, so the
        activity log can never show a controller action that was not applied.
        """
        reservoirs = state.get("reservoirs") or {}

        # ── 1. Risk transitions (NORMAL -> HIGH etc.), as classified by the
        #       backend's own risk engine. Only a real CHANGE is reported.
        current_risk = {
            key: str((res or {}).get("risk") or "normal").upper()
            for key, res in reservoirs.items()
        }
        for key, status in current_risk.items():
            previous = self._last_risk.get(key)
            if previous is not None and previous != status:
                res = reservoirs.get(key) or {}
                label = res.get("repository_name") or res.get("node_id") or key
                self.log_event(
                    "risk_change",
                    f"{label} risk {previous} -> {status}",
                    reservoir=label, previous_risk=previous, risk=status,
                    reason=res.get("risk_reason"),
                )
        # Persist the transitions so the AUTO block can explain WHY the
        # controller acted, using only transitions the backend really detected.
        self.last_risk_transitions = [
            {
                "reservoir": (reservoirs.get(key) or {}).get("repository_name") or key,
                "node_id": (reservoirs.get(key) or {}).get("node_id"),
                "previous_risk": self._last_risk.get(key),
                "risk": status,
            }
            for key, status in current_risk.items()
            if self._last_risk.get(key) is not None
            and self._last_risk.get(key) != status
        ]
        self._last_risk = current_risk

        # ── 2. Controller events. Emitted ONLY when the backend produced a
        #       decision for THIS step, and always from its own fields.
        if self.mode != "AI":
            return
        decision = self.last_control_decision
        if decision is None:
            return

        cstatus = str(getattr(decision, "controller_status", ""))
        if cstatus == ControllerStatus.BLOCKED.value:
            self.log_event(
                "control_blocked",
                "MPC control decision BLOCKED — holding current gates",
                controller_status=cstatus,
                reason=str(getattr(decision, "blocked_reason", "")),
            )
            return
        if cstatus == ControllerStatus.UNAVAILABLE.value:
            self.log_event(
                "control_error",
                "Controller UNAVAILABLE — holding current gates",
                controller_status=cstatus,
                reason=str(getattr(decision, "blocked_reason", "")),
            )
            return
        if cstatus != ControllerStatus.ACTIVE.value:
            return

        self.log_event(
            "mpc_decision",
            f"{getattr(decision, 'controller_type', 'MPC')} control decision generated",
            controller_status=cstatus,
            objective_score=getattr(decision, "mpc_objective_score", None),
            candidates_evaluated=getattr(decision, "candidates_evaluated", None),
        )
        self.log_event(
            "safety_validation",
            f"Safety validation: {getattr(decision, 'safety_layer_status', 'UNKNOWN')}",
            safety_layer_status=str(getattr(decision, "safety_layer_status", "")),
            safety_modified=bool(getattr(decision, "safety_modified", False)),
        )
        self.log_event(
            "downstream_protection",
            f"Downstream protection: {getattr(decision, 'downstream_status', 'UNKNOWN')}",
            downstream_status=str(getattr(decision, "downstream_status", "")),
            downstream_modified=bool(
                getattr(decision, "downstream_protection_modified", False)
            ),
        )

        # ── 3. The gates that ACTUALLY moved (measured, not claimed).
        for transition in self.last_gate_transitions or []:
            self.log_event(
                "gate_change",
                f"{transition['node']} gate: "
                f"{transition['previous_pct']:.0f}% -> {transition['new_pct']:.0f}%",
                reservoir=transition["node"],
                previous_pct=transition["previous_pct"],
                new_pct=transition["new_pct"],
            )
        if not self.last_gate_transitions:
            self.log_event(
                "control_hold",
                "Controller evaluated the network; no gate change required",
                controller_status=cstatus,
            )

    def reset_auto_control_state(self) -> None:
        """
        STAGE 18 — clear the AUTO-mode transient state after a RESET.

        A reset begins a NEW run, so the previous run's controller decision, gate
        transitions, risk memory and event log would all be stale. Clearing them
        is what makes ``AUTO`` report honestly (``AUTO_READY``) afterwards instead
        of showing a decision that belonged to the discarded run.

        No loop is created or stopped here: the ONE authoritative loop lives in
        ``simulation_loop()`` and simply keeps advancing whatever state exists.
        There is therefore no second loop to leak, and nothing to double-run.
        """
        self.last_control_decision = None
        self.mpc_orchestrator.last_decision = None
        self.last_gate_transitions = []
        self.last_previous_gate_pct = {}
        self.last_risk_transitions = []
        self._last_risk = {}
        self.event_log.clear()
        for history in self.history_buffers.values():
            history.clear()
        self.final_safety = {"checked": False}
        self.simulation_error = None
        # STAGE 19 (P3) — a RESET rebuilds the cascade and starts a new run:
        # the display pipeline must be recomputed for it, never reused.
        self._invalidate_pipeline_cache()
        self.log_event("reset", "Simulation RESET — authoritative state re-initialised")

    def step(self):
        self.simulation_error = None
        self._update_inflows()

        ctrl_forecasts = self._run_ml_pipeline()

        gate_commands = self.manual_gates.copy()
        action_source = "MANUAL_OPERATOR_GATES"
        # STAGE 18 — the REAL gates before this step's action, read from the
        # authoritative network. This is what the AI decision panel reports as
        # "previous gate" and what the 3D twin animates from.
        previous_gates = self._current_gate_pct()
        if self.mode == "AI":
            gate_commands = self._apply_ai_control(ctrl_forecasts, gate_commands)
            # STAGE 11 — name the provenance of the action that is about to be
            # applied. The orchestrator reported FINAL_SAFE_CONTROL_ACTION; the
            # mass-balance diagnostic records this alongside it so the audit can
            # be tied to the action that was really written to the physics.
            decision = self.last_control_decision
            action_source = getattr(decision, "final_safe_control_action_source", None) \
                or "UNKNOWN"

        # Final boundary for EVERY live path: manual, MPC, unavailable forecast,
        # and adapter failure. No gate reaches physics before this check.
        gate_commands = self._final_action_boundary(gate_commands)
        self.bridge.step(self.manual_inflows, gate_commands, action_source=action_source)
        self._record_history()
        # STAGE 11 — verify (after the step, on the audit of that step) that the
        # action applied to ReservoirNetwork is the controller's
        # FINAL_SAFE_CONTROL_ACTION. Never repairs and never re-applies.
        self._applied_action_verification = self._verify_applied_action(gate_commands)

        # ── STAGE 18 — record what ACTUALLY happened, from backend data only.
        if self.mode == "AI":
            self.last_previous_gate_pct = dict(previous_gates)
            self.last_gate_transitions = self._gate_transitions(
                previous_gates, gate_commands
            )
        state = self.get_adapted_state()
        # Observe only authoritative step results. Email delivery runs on a
        # bounded worker and cannot hold up simulation/control/WebSocket work.
        self.notification_manager.observe(state)
        state["notification"] = self.notification_manager.payload()
        self._record_control_events(state)
        return state

    def _final_action_boundary(self, requested):
        network = self.bridge.cascade.network
        ids = network.processing_order
        current = {n: network.nodes[n].state.gate_position for n in ids}
        proposed = {n: requested[n] / 100.0 for n in ids if n in requested}
        safety = self.mpc_orchestrator.safety.validate(proposed, current, ids)
        mpc = getattr(self.mpc_orchestrator, 'mpc', None)
        fraction = getattr(getattr(mpc, 'config', None), 'minimum_storage_fraction', None)
        reserve = reserve_bounds(network, self.manual_inflows,
            fraction) if self.mode == 'AI' and fraction is not None else None
        decision = self.last_control_decision if self.mode == "AI" else None
        if decision is not None and decision.control_applied:
            # Reuse THIS step's already checked forecast trajectory; verify the
            # exact proposed vector survived the final bounds/rate validation.
            checked_action = decision.final_safe_control_action_fraction
            if set(safety.validated_gates) != set(checked_action) or any(
                not math.isclose(value, checked_action[node], rel_tol=0.0, abs_tol=1e-11)
                for node, value in safety.validated_gates.items()
            ):
                raise RuntimeError("Final action diverged from the checked MPC action")
            # Apply the original checked vector without a percent round-trip.
            applied = dict(checked_action)
            downstream = dict(decision.downstream_capacity_protection)
            if downstream.get("predicted_flow_mcm_day") is None:
                raise RuntimeError("Cannot advance: MPC downstream prediction unavailable")
        else:
            proposal = dict(safety.validated_gates)
            if reserve:
                proposal = {n: min(g, reserve['gate_upper_bounds'][n]) for n, g in proposal.items()}
            checked = self.mpc_orchestrator.downstream_guard.evaluate(
                network, action_fraction=proposal,
                current_fraction=current, node_ids=ids,
                max_gate_change=self.mpc_orchestrator.safety.max_gate_change,
                inflows=dict(self.manual_inflows),
                **({'gate_upper_bounds': reserve['gate_upper_bounds']} if reserve else {}))
            if not checked.trajectory_mcm_day:
                raise RuntimeError("Cannot advance: downstream prediction unavailable")
            applied = checked.action_fraction
            downstream = checked.to_dict()
        if not self.mpc_orchestrator.downstream_guard.safety_layer_feasible(
                applied, ids, current, self.mpc_orchestrator.safety.max_gate_change):
            raise RuntimeError("Cannot advance: final gate bounds/rate check failed")
        verify_reserve_action(applied, reserve)
        self.final_safety = {"checked": True, "mode": self.mode,
            "gate_status": safety.status, "violations": safety.violations,
            "requested_gate_pct": dict(requested),
            "applied_gate_pct": {n: g * 100.0 for n, g in applied.items()},
            "downstream": downstream,
            "storage_reserve": dict(reserve, verified=True) if reserve else {"enabled": False, "reason": "Manual mode or reserve not configured"}}
        return dict(self.final_safety["applied_gate_pct"])

    def load_classroom_demo(self):
        """Deterministic initial conditions; all subsequent physics is unchanged."""
        self.running = False
        self.mode = "MANUAL"
        self.storm_intensity = 0.0
        self.classroom_demo = True
        self.simulation_error = None
        self.bridge.init_cascade(50.0)
        self.manual_gates = {n: 0.0 for n in self.bridge.cascade.network.processing_order}
        self.set_forecast_source("SIMULATION")
        self.reset_auto_control_state()
        self._update_inflows()
        self.log_event("classroom_demo", "Classroom preset: fixed local inflows; gates closed; daily steps")

    def _classroom_telemetry(self):
        network = self.bridge.cascade.network
        return {"active": self.classroom_demo, "day": network.timestep,
            "units": "storage MCM; flow MCM/day; one step = one day",
            "spill_policy": "Upstream spill exits through separate lateral outlets; controlled release routes downstream.",
            "reservoirs": [{"node": n, "storage": node.state.storage,
                "local_inflow": node.state.inflow_local, "routed_inflow": node.state.inflow_routed,
                "controlled_release": node.state.controlled_release, "spill": node.state.spill,
                "gate_pct": node.state.gate_position * 100.0}
                for n, node in network.nodes.items()],
            "routes": [{"source": c.source, "destination": c.destination,
                "delay_days": c.delay, "attenuation": c.attenuation,
                "queued_mcm": sum(c.queue)} for c in network.connections]}

    def _schedule_notification_broadcast(self):
        """Publish asynchronous mail completion without blocking its SMTP worker."""
        loop = self._notification_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast_state()))

    def close_resolved_notification_after_reset(self):
        """Close a prior incident from the reset cascade's existing status."""
        reset_state = self.bridge.get_state({})
        adapted = adapt_state_for_twin(reset_state, self.mode, self.storm_intensity)
        adapted["state_identity"] = {"network_timestep": int(self.bridge.cascade.network.timestep)}
        downstream = adapted.get("downstream") or {}
        self.notification_manager.close_if_resolved(downstream.get("status"), state=adapted)

    #: Tolerance for the applied-vs-final-action comparison (gate PERCENT).
    ACTION_MATCH_TOLERANCE_PERCENT = 1e-9

    def _verify_applied_action(self, gate_commands: dict) -> dict:
        """
        STAGE 11 — the check must validate the action that was APPLIED.

        Compares the gate commands handed to ``ReservoirNetwork`` with the
        ``FINAL_SAFE_CONTROL_ACTION`` the orchestrator reported for this step.
        A mismatch is reported, never corrected.
        """
        info = {
            "controller_action_checked": False,
            "matches_final_safe_control_action": None,
            "checked_action_source": None,
            "action_check_note": "",
        }
        decision = self.last_control_decision
        if self.mode != "AI" or decision is None or not getattr(decision, "control_applied", False):
            info["action_check_note"] = (
                "NO_CONTROLLER_ACTION_FOR_THIS_STEP: manual operator gates were applied"
                if self.mode == "AI" else
                "NO_CONTROLLER_ACTION_FOR_THIS_STEP: MANUAL mode applies operator gates"
            )
            return info

        final_pct = dict(getattr(decision, "final_safe_control_action_pct", {}) or {})
        matches = set(final_pct) == set(gate_commands) and all(
            abs(float(gate_commands[nid]) - float(final_pct[nid]))
            <= self.ACTION_MATCH_TOLERANCE_PERCENT
            for nid in final_pct
        )
        info["controller_action_checked"] = True
        info["checked_action_source"] = getattr(decision, "final_safe_control_action_source", None)
        info["matches_final_safe_control_action"] = bool(matches)
        info["action_check_note"] = (
            "the gates applied to ReservoirNetwork are the controller's "
            "FINAL_SAFE_CONTROL_ACTION"
            if matches else
            "MISMATCH: the gates applied to ReservoirNetwork differ from the "
            "controller's FINAL_SAFE_CONTROL_ACTION"
        )
        return info

    def get_adapted_state(self):
        # ── STAGE 19 (P3) — DISPLAY forecasts are memoised per authoritative
        # step. This method is called by `step()`, by `broadcast_state()` and by
        # REST `/api/state`; each call used to re-run the whole forecast/GNN
        # advisory pipeline even when nothing had changed. The CONTROL path in
        # `step()` still calls `_run_ml_pipeline()` directly and is untouched.
        ctrl_forecasts = self._display_forecasts()
        current_state = self.bridge.get_state(ctrl_forecasts)
        current_state["storm_intensity"] = self.storm_intensity

        # STAGE 7 — authoritative controller provenance (MPC / SafetyLayer /
        # downstream boundary) reaches the API/UI.
        current_state["control"] = self.mpc_orchestrator.status_dict()

        # ── STAGE 18 — AUTO CONTROL STATE, FORECAST SOURCE AND LIVE EVENT LOG
        # All three are produced by the BACKEND. The twin renders them; it never
        # derives a control state, a reason or an event of its own.
        current_state["auto_control"] = self._auto_control_block()
        current_state["forecast_source"] = self.forecast_source
        current_state["event_log"] = self.event_log_payload()

        # ── STAGE 12 — AUTHORITATIVE STATE IDENTITY ────────────────────────
        # What identifies a state update: the simulation progress the backend
        # itself maintains. NO new clock is introduced.
        #   sim_step_index   — steps taken by THIS one live simulation instance
        #   network_timestep — ReservoirNetwork.timestep, incremented ONLY by an
        #                      authoritative ReservoirNetwork.step()
        # A RESET is distinguishable: network_timestep returns to 0 while
        # sim_step_index keeps counting.
        _network_timestep = int(self.bridge.cascade.network.timestep)
        current_state["state_identity"] = {
            "sim_step_index": int(self.sim_step_index),
            "network_timestep": _network_timestep,
            "state_id": f"step{int(self.sim_step_index)}-t{_network_timestep}",
            "source": (
                "GlobalSimulationState.sim_step_index + ReservoirNetwork.timestep "
                "(authoritative; no browser clock)"
            ),
        }
        # STAGE 12 — whether the authoritative simulation is running, and at what
        # speed. The frontend used to print a hardcoded "RUNNING"; it now displays
        # the backend's own answer (and `--` when it has none).
        current_state["simulation"] = {
            "running": bool(self.running),
            "speed": float(self.sim_speed),
            "error": self.simulation_error,
            "source": "GlobalSimulationState.running / sim_speed",
        }

        # ── STAGE 11 — LIVE MASS-BALANCE INTEGRITY ─────────────────────────
        # The audit of the last authoritative step reaches the API/WebSocket/twin
        # exactly as the physics produced it, plus the applied-action check. When
        # no step has been audited the block says NOT_CHECKED / checked=False and
        # NO applied-action verdict is attached (never "fine").
        mass_balance = dict(current_state.get("mass_balance") or {})
        if mass_balance.get("checked") is True:
            mass_balance.update(self._applied_action_verification)
        else:
            mass_balance.update({
                "controller_action_checked": False,
                "matches_final_safe_control_action": None,
                "checked_action_source": None,
                "action_check_note": "NO_AUDITED_STEP_YET",
            })
        current_state["mass_balance"] = mass_balance

        # STAGE 14 — the GNN advisory reaches the API/WebSocket/twin for DISPLAY
        # only. It is attached after the control path has already produced its
        # forecast, and nothing downstream of this point feeds a decision.
        current_state["gnn_advisory"] = self.gnn_advisory

        # Requests are presentation metadata, never a replacement for physical gates.
        if self.mode == "MANUAL":
            for res, gate_val in self.manual_gates.items():
                if res in current_state["reservoirs"]:
                    current_state["reservoirs"][res]["requested_gate_pct"] = gate_val

        adapted = adapt_state_for_twin(current_state, self.mode, self.storm_intensity)
        adapted["notification"] = self.notification_manager.payload()
        adapted["final_safety"] = self.final_safety
        adapted["classroom_demo"] = self._classroom_telemetry()
        return adapted

    def _record_loop_overrun(self, overrun_s: float) -> None:
        """
        STAGE 19 (P4) — record that one loop iteration missed its deadline.

        Diagnostic only: it is written so an operator can SEE cadence pressure
        (a slow controller, a loaded host). Nothing in the physics or control
        path reads it, and an overrun never causes a catch-up step.
        """
        self.loop_overrun_count += 1
        self.last_loop_overrun_s = float(overrun_s)
        if overrun_s > self.max_loop_overrun_s:
            self.max_loop_overrun_s = float(overrun_s)
        logger.debug(
            "simulation_loop deadline overrun by %.3fs (count=%d)",
            overrun_s, self.loop_overrun_count,
        )

    async def broadcast_state(self):
        if not self.clients:
            return
        loop = asyncio.get_running_loop()
        if getattr(self, "_broadcast_loop", None) is not loop:
            self._broadcast_loop = loop
            self._broadcast_lock = asyncio.Lock()

        async def send(client, state_json):
            try:
                await asyncio.wait_for(client.send_text(state_json), timeout=0.5)
            except Exception:
                self.clients.discard(client)
                # Removing a slow socket from broadcasts must also trigger the
                # browser's reconnect path; heartbeats alone could look healthy.
                if hasattr(client, "close"):
                    try:
                        await asyncio.wait_for(client.close(code=1013), timeout=0.1)
                    except Exception:
                        pass

        # Order snapshots, serialize once, and bound each client's backpressure.
        async with self._broadcast_lock:
            state_json = json.dumps(self.get_adapted_state(), allow_nan=False)
            await asyncio.gather(*(send(client, state_json) for client in list(self.clients)))
                
    def record_simulation_error(self, exc):
        """Pause visibly after a rejected step; keep the loop available for recovery."""
        self.running = False
        self.simulation_error = str(exc)
        self.log_event("simulation_error", f"Simulation paused: {exc}")
        logger.exception("Simulation step failed; playback paused")

    async def simulation_loop(self):
        """
        STAGE 19 (P4) — DEADLINE-BASED scheduling (no cadence drift).

        The old loop was ``step(); broadcast(); sleep(1/sim_speed)``, so the
        real period was ``1/sim_speed + computation_time`` — at speed 1.0 the
        twin updated every ~1.28 s instead of every 1.0 s, and the gap grew
        with controller cost. This version advances a deadline by one period
        per authoritative step and sleeps only for the REMAINING time:

            next_deadline += 1.0 / sim_speed
            sleep(max(0, next_deadline - now))

        * When the system keeps up, the cadence is ``1/sim_speed`` — at
          speed 1.0 that is approximately one authoritative step per second.
        * If a step overruns its deadline, the loop does NOT run catch-up
          steps, does NOT duplicate or skip authoritative steps, and does NOT
          burst: the missed slot is dropped, the overrun is recorded, and the
          schedule resumes from the next period. The authoritative state
          therefore remains exactly one step per iteration.
        * Physics, controller timing semantics and `sim_speed`'s meaning
          (steps per second) are unchanged: only WHEN the sleep happens moved.
        """
        self._notification_loop = asyncio.get_running_loop()
        # Monotonic clock for the deadline arithmetic.
        clock = time.monotonic
        next_deadline = clock()
        while True:
            if not self.running:
                # Idle: drop the schedule so resuming starts fresh (no burst).
                next_deadline = clock()
                await asyncio.sleep(0.5)
                continue

            # ── ONE authoritative step, exactly as before.
            try:
                self.step()
            except Exception as exc:
                self.record_simulation_error(exc)
            # ── Broadcast the resulting authoritative state.
            await self.broadcast_state()

            # ── Deadline arithmetic: advance one period per step taken.
            period = 1.0 / self.sim_speed
            next_deadline += period
            remaining = next_deadline - clock()
            if remaining > 0.0:
                await asyncio.sleep(remaining)
            else:
                # OVERRUN: the computation took longer than the period. Record
                # it (diagnostic only) and DROP the missed slot instead of
                # running catch-up steps — the next iteration simply starts now
                # and advances the deadline by one period from there.
                self._record_loop_overrun(-remaining)
                next_deadline = clock()
                # STAGE 19 (P4) — release the event loop WITHOUT delaying.
                # `broadcast_state()` returns immediately when no client is
                # connected and `step()` is synchronous, so the overrun branch
                # would otherwise be a tight, never-yielding loop that starves
                # every other task (WebSocket, REST, and this loop's own
                # cancellation). A zero-delay yield adds no step, no delay and
                # no catch-up; it only guarantees the loop stays cooperative.
                await asyncio.sleep(0)

# ── STAGE 4 — THE one authoritative live simulation instance ─────────────────
# Both the REST command routes (src/dashboard/api/routes.py) and the WebSocket
# feed (src/dashboard/api/app.py) import THIS object, so the state that is
# published is always the state produced by the simulation that processed the
# commands. No other live producer exists.
sim_state = GlobalSimulationState()
