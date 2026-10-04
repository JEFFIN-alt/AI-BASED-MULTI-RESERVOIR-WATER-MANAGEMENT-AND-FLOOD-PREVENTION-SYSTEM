"""Independent Discord delivery; no network or chart work on the simulation thread."""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
from threading import Event, RLock, Thread
import time
from typing import Callable, Protocol, runtime_checkable

import httpx


@runtime_checkable
class Notifier(Protocol):
    def notify(self, state: dict, incident: dict | None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class GmailAdapter:
    """Delegate to the existing email incident handler, preserving its behaviour."""
    def __init__(self, observe: Callable[[dict], None]):
        self._observe = observe

    def notify(self, state, incident=None):
        self._observe(state)

    def start(self):
        pass

    def stop(self):
        pass


_WEBHOOK = re.compile(
    r"https://(?:(?:canary|ptb)\.)?(?:discord\.com|discordapp\.com)"
    r"/api/(?:v\d+/)?webhooks/\d+/[A-Za-z0-9._-]+(?:\?thread_id=\d+)?"
)
_WEBHOOK_IN_LOG = re.compile(
    r"https?://(?:(?:canary|ptb)\.)?(?:discord\.com|discordapp\.com)"
    r"/api/(?:v\d+/)?webhooks/[^\s'\"<>]+"
)
_WEBHOOK_PATH_IN_LOG = re.compile(r"/api/(?:v\d+/)?webhooks/[^\s'\"<>]+")
MASKED_WEBHOOK = "https://discord.com/api/webhooks/***/***"
COLOURS = {"WARNING": 0xF59E0B, "CRITICAL": 0xDC2626, "RECOVERY": 0x16A34A}
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _local_webhook():
    """Read only Discord's local secret once, without executing or logging it."""
    value = ""
    try:
        for line in _ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            key, separator, candidate = line.strip().removeprefix("export ").partition("=")
            if separator and key.strip() == "DISCORD_WEBHOOK_URL":
                candidate = candidate.strip()
                if candidate[:1] in ("'", '"'):
                    quote = candidate[0]
                    end = candidate.find(quote, 1)
                    candidate = candidate[1:end] if end > 0 else ""
                else:
                    candidate = candidate.partition(" #")[0].strip()
                value = candidate
    except (OSError, UnicodeError):
        pass
    return value


class _WebhookLogFilter(logging.Filter):
    """httpx logs request URLs even when application errors are sanitized."""
    def filter(self, record):
        record.msg = _WEBHOOK_IN_LOG.sub(MASKED_WEBHOOK, record.getMessage())
        record.msg = _WEBHOOK_PATH_IN_LOG.sub("/api/webhooks/***/***", record.msg)
        record.args = ()
        return True


for _logger_name in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11",
                     "httpcore.http2", "httpcore.proxy", "httpcore.socks"):
    logging.getLogger(_logger_name).addFilter(_WebhookLogFilter())


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _snapshot(state):
    """Keep bounded telemetry only; omit forecast arrays and unsupported drivers."""
    control = state.get("control") or {}
    return {
        "downstream": {k: (state.get("downstream") or {}).get(k)
                       for k in ("status", "flow_m3_s", "capacity_m3_s")},
        "control": {k: control.get(k) for k in
                    ("downstream_proposed_predicted_flow_mcm_day", "downstream_capacity_mcm_day")},
        "state_identity": {"network_timestep": (state.get("state_identity") or {}).get("network_timestep")},
        "reservoirs": {f"reservoir_{i}": {"gate": ((state.get("reservoirs") or {}).get(f"reservoir_{i}") or {}).get("gate")}
                       for i in range(1, 5)},
    }


def build_message(state, incident, kind="ALERT"):
    """One bounded embed, an attachment reference, and no chat components."""
    ds, control = state.get("downstream") or {}, state.get("control") or {}
    level = "RECOVERY" if kind == "RECOVERY" else incident.get("severity", "WARNING")
    fields = []
    if _number(ds.get("flow_m3_s")) and _number(ds.get("capacity_m3_s")):
        fields.append({"name": "DS flow / safe limit", "value":
                       f"{ds['flow_m3_s']:.3f} / {ds['capacity_m3_s']:.3f} m³/s"})
    predicted = control.get("downstream_proposed_predicted_flow_mcm_day")
    capacity = control.get("downstream_capacity_mcm_day")
    if not _number(capacity) and _number(ds.get("capacity_m3_s")):
        capacity = ds["capacity_m3_s"] * 86400 / 1_000_000
    if _number(predicted) and _number(capacity):
        fields.append({"name": "MPC predicted downstream / capacity (proposed)",
                       "value": f"{predicted:.3f} / {capacity:.3f} MCM/day"})
    gates = []
    for i, name in enumerate("ABCD", 1):
        gate = ((state.get("reservoirs") or {}).get(f"reservoir_{i}") or {}).get("gate")
        if _number(gate):
            gates.append(f"{name}: {gate * 100:.1f}%")
    if gates:
        fields.append({"name": "Final gate actions (applied)", "value": " · ".join(gates)})
    day = (state.get("state_identity") or {}).get("network_timestep")
    if _number(day):
        fields.append({"name": "Sim day", "value": str(day)})
    description = ("Downstream telemetry has returned to NORMAL." if kind == "RECOVERY" else
                   "Test Discord alert. No simulation state or gates were changed." if kind == "TEST" else
                   "Acknowledge this incident in AquaFlow Settings → Notifications to stop reminders.")
    embed = {"title": f"AquaFlow — {'TEST' if kind == 'TEST' else level} downstream notification",
             "description": description[:4096], "color": COLOURS.get(level, COLOURS["WARNING"]),
             "fields": fields[:25], "image": {"url": "attachment://chart.png"},
             "footer": {"text": f"Incident: {incident.get('incident_id') or 'test'} · {kind}"[:2048]}}
    return {"embeds": [embed], "attachments": [{"id": 0, "filename": "chart.png"}],
            "allowed_mentions": {"parse": []}}


def downstream_chart(history):
    """Plot available real samples, including a single sample; never fabricate data."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(6, 2.5), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    valid = [sample for sample in history if _number(sample[1]) and _number(sample[2])]
    if valid:
        ax.plot(range(len(valid)), [p[1] for p in valid], marker=".", color="#0284c7", label="DS flow")
        ax.plot(range(len(valid)), [p[2] for p in valid], linestyle="--", color="#dc2626", label="Safe limit")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "No downstream samples available", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Recent backend samples (oldest → newest)", ylabel="m³/s")
    ax.grid(alpha=.2)
    output = BytesIO()
    fig.savefig(output, format="png")
    return output.getvalue()


class DiscordAdapter:
    """Bounded, independent worker with backend-owned incident/acknowledgement state.

    Reminders use wall time, so acknowledgement and recovery also work while the
    simulator is paused. Records are in-memory, like the existing Gmail incidents.
    """
    def __init__(self, transport=None, on_update=None, *, clock=time.monotonic,
                 cooldown_seconds=30, repeat_seconds=300, chart=downstream_chart):
        self._transport = transport
        self._on_update = on_update
        self._clock = clock
        self.cooldown_seconds = max(1, float(cooldown_seconds))
        self.repeat_seconds = max(self.cooldown_seconds, float(repeat_seconds))
        self._chart = chart
        self.enabled = True
        self._lock = RLock()
        self._queue = Queue(maxsize=32)
        self._records = {}
        self._current = None
        self._latest = None
        self._last_test_attempt = float("-inf")
        self._history = deque(maxlen=60)
        self._last_identity = None
        self._rate_limit_until = 0.0
        self._invalid_webhook = None
        self._stop = Event()
        self._thread = None
        self._local_webhook = _local_webhook()

    def _config(self):
        value = os.environ.get("DISCORD_WEBHOOK_URL", self._local_webhook).strip()
        return value if _WEBHOOK.fullmatch(value) else None

    @staticmethod
    def _public(record):
        return {k: v for k, v in record.items() if not k.startswith("_")}

    def settings_payload(self):
        configured = self._config() is not None
        with self._lock:
            last = self._latest or {}
            return {"configured": configured, "status": "Configured" if configured else "Not configured",
                    "masked_webhook": MASKED_WEBHOOK if configured else None, "enabled": self.enabled,
                    "last_delivery": last.get("status", "NOT_CONFIGURED"),
                    "last_delivery_timestamp": last.get("timestamp"), "error": last.get("error"),
                    "repeat_seconds": self.repeat_seconds, "cooldown_seconds": self.cooldown_seconds}

    def active_incidents(self):
        with self._lock:
            return [self._public(r) for r in self._records.values() if not r["closed"]]

    def acknowledge(self, incident_id):
        with self._lock:
            record = self._records.get(incident_id)
            if record is None:
                raise KeyError("Incident not found")
            if record["closed"]:
                raise ValueError("Incident is already resolved")
            if not record["acknowledged"]:
                record.update(acknowledged=True, acknowledged_at=_now())
            result = self._public(record)
        self._changed()
        return result

    def set_enabled(self, enabled):
        with self._lock:
            self.enabled = bool(enabled)
            if self.enabled and self._current and not self._current["_attempted"]:
                self._current["_due"] = self._clock()
        self._changed()

    def _changed(self):
        if self._on_update:
            try:
                self._on_update()
            except Exception:
                pass

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(target=lambda: asyncio.run(self._run()), name="aquaflow-discord", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        while True:
            try:
                record, _ = self._queue.get_nowait()
            except Empty:
                break
            self._finish(record, "FAILED", "Discord worker stopped before delivery")
            self._queue.task_done()

    def notify(self, state, incident):
        snapshot = _snapshot(state)
        severity = snapshot["downstream"]["status"]
        with self._lock:
            identity = (state.get("state_identity") or {}).get("state_id")
            ds = snapshot["downstream"]
            identity = identity if identity is not None else (snapshot["state_identity"]["network_timestep"],
                                                            ds["flow_m3_s"], ds["capacity_m3_s"])
            if identity != self._last_identity:
                self._history.append((identity, ds["flow_m3_s"], ds["capacity_m3_s"]))
                self._last_identity = identity
            if severity == "NORMAL":
                self._close_locked(snapshot)
            elif severity in ("WARNING", "CRITICAL") and incident:
                incident_id = incident["incident_id"]
                record = self._records.get(incident_id)
                if record is None:
                    if self._current and self._current["incident_id"] != incident_id:
                        # UNKNOWN telemetry can split Gmail's contiguous incident.
                        # Supersede the old record without inventing a recovery.
                        self._current.update(closed=True, _recovery_needed=False)
                    # Only closed, idle records are evicted; the delivery queue is bounded.
                    for old_id in list(self._records):
                        if len(self._records) < 128:
                            break
                        old = self._records[old_id]
                        if old["closed"] and not old["_pending"]:
                            del self._records[old_id]
                    record = {"incident_id": incident_id, "severity": severity, "timestamp": _now(),
                              "acknowledged": False, "acknowledged_at": None, "closed": False,
                              "status": "NOT_CONFIGURED", "error": None, "_pending": False,
                              "_attempted": False, "_sent": False, "_due": self._clock(),
                              "_last_attempt": float("-inf"), "_recovery_needed": False}
                    self._records[incident_id] = record
                if record["closed"]:
                    return  # An old incident cannot be reopened by stale observations.
                if record["severity"] != severity and severity == "CRITICAL":
                    record["_due"] = min(record["_due"], record["_last_attempt"] + self.cooldown_seconds)
                    record["severity"] = severity
                record["_snapshot"] = snapshot
                self._current = record
                self._schedule_due_locked()

    def close(self, state=None):
        with self._lock:
            self._close_locked(_snapshot(state or {}))
            self._history.clear()
            self._last_identity = None
        self._changed()

    def _close_locked(self, snapshot):
        if self._current:
            record = self._current
            record.update(closed=True, recovered_at=_now(), _snapshot=snapshot, _recovery_needed=True,
                          _recovery_history=list(self._history))
            self._current = None
            self._schedule_due_locked()

    def send_test(self, state=None):
        with self._lock:
            if self._latest and self._latest.get("kind") == "TEST" and self._latest.get("_pending"):
                return self._public(self._latest)
            if self._clock() - self._last_test_attempt < self.cooldown_seconds:
                raise RuntimeError("Discord test cooldown is active")
            record = {"kind": "TEST", "timestamp": _now(), "_pending": False,
                      "_snapshot": _snapshot(state or {}), "_attempted": False, "_sent": False}
            self._latest = record
            self._last_test_attempt = self._clock()
            self._enqueue_locked(record, "TEST")
            return self._public(record)

    def _schedule_due_locked(self):
        now = self._clock()
        for record in self._records.values():
            if record["_pending"]:
                continue
            if record["closed"]:
                if (record["_recovery_needed"] and record["_sent"] and
                        now >= record["_last_attempt"] + self.cooldown_seconds):
                    self._enqueue_locked(record, "RECOVERY")
            elif not record["acknowledged"] and now >= record["_due"]:
                self._enqueue_locked(record, "REMINDER" if record["_attempted"] else "ALERT")

    def _enqueue_locked(self, record, kind):
        self._latest = record
        if not self._config():
            record.update(status="NOT_CONFIGURED", error=None)
            return
        if kind != "TEST" and not self.enabled:
            record.update(status="DISABLED", error=None)
            return
        if self._stop.is_set():
            record.update(status="FAILED", error="Discord worker stopped")
            return
        record.update(status="PENDING", error=None, kind=kind, _pending=True)
        self._latest = record
        try:
            self._queue.put_nowait((record, kind))
        except Full:
            record.update(status="FAILED", error="Discord delivery queue full", _pending=False)
            if "_due" in record:
                record["_due"] = self._clock() + self.repeat_seconds

    def _finish(self, record, status, error=None):
        with self._lock:
            record.update(status=status, error=error, timestamp=_now(), _pending=False)
            if "_due" in record:
                record["_due"] = self._clock() + self.repeat_seconds
        self._changed()

    async def _wait(self, seconds):
        deadline = self._clock() + seconds
        while self._clock() < deadline and not self._stop.is_set():
            await asyncio.sleep(min(.1, deadline - self._clock()))

    @staticmethod
    def _retry_after(response):
        try:
            value = response.json().get("retry_after")
        except (ValueError, AttributeError):
            value = None
        if not _number(value):
            try:
                value = float(response.headers.get("Retry-After", "1"))
            except ValueError:
                value = 1
        return max(0, value) if _number(value) else 1

    async def _deliver(self, record, kind, client):
        try:
            with self._lock:
                cancelled = kind not in ("TEST", "RECOVERY") and (record["closed"] or record["acknowledged"])
                snapshot = record["_snapshot"]
                incident = self._public(record)
                history = record.get("_recovery_history", []) if kind == "RECOVERY" else list(self._history)
                if kind == "RECOVERY":
                    record["_recovery_needed"] = False
            if cancelled:
                self._finish(record, "CANCELLED")
                return
            payload, image = build_message(snapshot, incident, kind), self._chart(history)
            url = self._config()
            if not url:
                self._finish(record, "NOT_CONFIGURED")
                return
            if url == self._invalid_webhook:
                self._finish(record, "FAILED", "Discord webhook is unavailable; replace its configuration")
                return
            for attempt in range(2):
                await self._wait(max(0, self._rate_limit_until - self._clock()))
                with self._lock:
                    cancelled = (kind not in ("TEST", "RECOVERY") and
                                 (record["closed"] or record["acknowledged"]))
                    disabled = kind != "TEST" and not self.enabled
                if self._stop.is_set() or cancelled or disabled:
                    self._finish(record, "CANCELLED" if not disabled else "DISABLED")
                    return
                with self._lock:
                    record["_attempted"] = True
                    record["_last_attempt"] = self._clock()
                response = await client.post(httpx.URL(url).copy_merge_params({"wait": "true"}),
                                             data={"payload_json": json.dumps(payload, allow_nan=False)},
                                             files={"files[0]": ("chart.png", image, "image/png")})
                if response.status_code == 429:
                    self._rate_limit_until = self._clock() + self._retry_after(response)
                    if attempt == 0:
                        continue
                    self._finish(record, "FAILED", "Discord rate limit persisted after one retry")
                    return
                if response.status_code in (401, 403, 404):
                    self._invalid_webhook = url
                if not 200 <= response.status_code < 300:
                    self._finish(record, "FAILED", f"Discord rejected delivery (HTTP {response.status_code})")
                    return
                confirmation = response.json()
                if not isinstance(confirmation, dict) or not confirmation.get("id"):
                    self._finish(record, "FAILED", "Discord did not confirm delivery")
                    return
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    try:
                        delay = float(response.headers.get("X-RateLimit-Reset-After", "0"))
                        if _number(delay):
                            self._rate_limit_until = self._clock() + max(0, delay)
                    except ValueError:
                        pass
                with self._lock:
                    record["_sent"] = True
                self._finish(record, "SENT")
                return
        except Exception:
            # Never expose exceptions, response bodies, request URLs or webhook tokens.
            self._finish(record, "FAILED", "Discord delivery failed or timed out")

    async def deliver_pending(self, client):
        """Single-consumer pump, also exercised with mocked clients in tests."""
        with self._lock:
            self._schedule_due_locked()
        while not self._stop.is_set():
            try:
                record, kind = self._queue.get_nowait()
            except Empty:
                break
            try:
                await self._deliver(record, kind, client)
            finally:
                self._queue.task_done()

    async def _run(self):
        async with httpx.AsyncClient(transport=self._transport, timeout=httpx.Timeout(3, connect=2),
                                     follow_redirects=False, trust_env=False) as client:
            while not self._stop.is_set():
                await self.deliver_pending(client)
                await asyncio.sleep(.1)
