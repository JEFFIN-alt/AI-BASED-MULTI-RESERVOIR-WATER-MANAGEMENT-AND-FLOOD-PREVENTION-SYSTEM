import asyncio
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from src.dashboard.sim_bridge import SimBridge
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
from src.modeling.inference import LiveForecaster, ForecastUnavailableError
from src.modeling.gnn_inference import LiveGNNForecaster
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
                history[p_name] = window.reshape(
                    HISTORY_DAYS, len(self.lstm_forecaster.dynamic_features)
                ).astype(np.float32)
            except Exception:
                continue
        return history

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
        lstm_forecasts = {}
        for v_name, p_name in self.res_mapping.items():
            if v_name in LIVE_FORECAST_EXCLUDED:
                continue
            lstm_forecasts[v_name] = self._forecast_one_reservoir(v_name, p_name)

        # Advisory/experimental GNN — never feeds the control path.
        if self.gnn_ready:
            try:
                gnn_history = self._build_gnn_history()
                if gnn_history:
                    gnn_result = self.gnn_forecaster.predict_all(gnn_history)
                    self.gnn_adapter.update_from_inference(gnn_result)
            except Exception as e:
                print(f"[StateManager] GNN advisory inference failed: {e}")

        # Production control policy (validated LSTM V3 baseline).
        ctrl_forecasts = self.gnn_adapter.get_control_forecasts(lstm_forecasts, policy="lstm_primary")

        # Carry the Stage 5 provenance through to the control/display payload.
        for name, fc in lstm_forecasts.items():
            if name in ctrl_forecasts:
                for key in ("forecast_status", "forecast_provenance", "forecast_source",
                            "is_simulated", "validated_metrics_apply",
                            "unavailable_features", "unavailable_reasons",
                            "steps_collected", "note", "input_provenance",
                            "forecast_unit", "horizons", "target_columns",
                            "model_version", "feature_order", "feature_units",
                            "history_days", "forecast_date"):
                    if key in fc:
                        ctrl_forecasts[name][key] = fc[key]
        return ctrl_forecasts

    # ------------------------------------------------------------------
    # STAGE 7 — the ONE authoritative live controller path
    # ------------------------------------------------------------------

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
        return adapter.build_bundle(ctrl_forecasts, self._forecast_date())

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

    def step(self):
        self._update_inflows()
        self._record_history()

        ctrl_forecasts = self._run_ml_pipeline()

        gate_commands = self.manual_gates.copy()
        action_source = "MANUAL_OPERATOR_GATES"
        if self.mode == "AI":
            gate_commands = self._apply_ai_control(ctrl_forecasts, gate_commands)
            # STAGE 11 — name the provenance of the action that is about to be
            # applied. The orchestrator reported FINAL_SAFE_CONTROL_ACTION; the
            # mass-balance diagnostic records this alongside it so the audit can
            # be tied to the action that was really written to the physics.
            decision = self.last_control_decision
            action_source = getattr(decision, "final_safe_control_action_source", None) \
                or "UNKNOWN"

        self.bridge.step(self.manual_inflows, gate_commands, action_source=action_source)
        # STAGE 11 — verify (after the step, on the audit of that step) that the
        # action applied to ReservoirNetwork is the controller's
        # FINAL_SAFE_CONTROL_ACTION. Never repairs and never re-applies.
        self._applied_action_verification = self._verify_applied_action(gate_commands)
        return self.get_adapted_state()

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
        ctrl_forecasts = self._run_ml_pipeline()
        current_state = self.bridge.get_state(ctrl_forecasts)
        current_state["storm_intensity"] = self.storm_intensity

        # STAGE 7 — authoritative controller provenance (MPC / SafetyLayer /
        # downstream boundary) reaches the API/UI.
        current_state["control"] = self.mpc_orchestrator.status_dict()

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

        # Inject live manual gate state into current_state for immediate visual feedback
        if self.mode == "MANUAL":
            for res, gate_val in self.manual_gates.items():
                if res in current_state["reservoirs"]:
                    current_state["reservoirs"][res]["gate_position_pct"] = gate_val

        return adapt_state_for_twin(current_state, self.mode, self.storm_intensity)

    async def broadcast_state(self):
        if not self.clients:
            return
        state = self.get_adapted_state()
        state_json = json.dumps(state)
        for client in list(self.clients):
            try:
                await client.send_text(state_json)
            except Exception:
                self.clients.remove(client)
                
    async def simulation_loop(self):
        while True:
            if self.running:
                self.step()
                await self.broadcast_state()
                await asyncio.sleep(1.0 / self.sim_speed)
            else:
                await asyncio.sleep(0.5)

# ── STAGE 4 — THE one authoritative live simulation instance ─────────────────
# Both the REST command routes (src/dashboard/api/routes.py) and the WebSocket
# feed (src/dashboard/api/app.py) import THIS object, so the state that is
# published is always the state produced by the simulation that processed the
# commands. No other live producer exists.
sim_state = GlobalSimulationState()

