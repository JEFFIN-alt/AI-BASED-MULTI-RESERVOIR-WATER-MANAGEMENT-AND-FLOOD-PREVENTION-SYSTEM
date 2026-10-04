"""Telegram delivery and authenticated polling, outside the control loop.

Bot API reference: https://core.telegram.org/bots/api
Run one application process per bot (getUpdates has a single consumer).
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
import time
from typing import Protocol
from urllib.request import Request, urlopen
import uuid


class Notifier(Protocol):
    def notify(self, state: dict, incident: dict | None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


def _now():
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def alert_message(state, incident):
    ds, control = state.get("downstream") or {}, state.get("control") or {}
    lines = ["🔴 AQUAFLOW DOWNSTREAM ALERT", "", f"Status: {incident['severity']} RISK",
             f"Incident: {incident['incident_id']}", f"Time: {incident['timestamp']}", "", "DOWNSTREAM"]
    for label, key, unit in (("Flow", "flow_m3_s", " m³/s"), ("Safe limit", "capacity_m3_s", " m³/s"),
                             ("Utilisation", "utilisation", "")):
        if _number(ds.get(key)):
            lines.append(f"{label}: {ds[key]:.3g}{unit}")
    reason = control.get("downstream_reason") or ds.get("reason")
    if reason:
        lines.extend(["", "Reason:", str(reason)[:220]])
    lines.extend(["", "CONTROL"])
    mode = state.get("controller_mode")
    if mode:
        lines.append(f"Mode: {'AUTO' if mode == 'AI' else mode}")
    source = (state.get("forecast_source_selection") or {}).get("selected")
    if source:
        lines.append(f"Forecast source: {source}")
    predicted = control.get("downstream_proposed_predicted_flow_mcm_day")
    if _number(predicted):
        label = "MPC predicted downstream (proposed)" if control.get("controller_type") == "MPC" else "Proposed predicted downstream"
        lines.append(f"{label}: {predicted:.3g} MCM/day")
    lines.extend(["", "FINAL GATE ACTIONS"])
    for index, label in enumerate("ABCD", 1):
        gate = ((state.get("reservoirs") or {}).get(f"reservoir_{index}") or {}).get("gate")
        if _number(gate):
            lines.append(f"{label}: {gate * 100:.1f}%")
    protection = control.get("downstream_status")
    if protection is not None:
        lines.append(f"Downstream protection: {protection}")
    return "\n".join(lines)[:4000]


def downstream_chart(history):
    """At least three distinct real backend samples, with their actual limits."""
    valid = [p for p in history if _number(p[1]) and _number(p[2])]
    if len(valid) < 3:
        return None
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6, 2.5), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(range(len(valid)), [p[1] for p in valid], label="Downstream flow", color="#007f99")
    ax.plot(range(len(valid)), [p[2] for p in valid], label="Safe limit", color="#dc3545", linestyle="--")
    ax.set(xlabel="Recent backend samples (oldest → newest)", ylabel="m³/s")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.2)
    output = BytesIO()
    fig.savefig(output, format="png")
    return output.getvalue()


def _request(token, method, fields, photo=None):
    """Never propagate provider bodies, URLs or exceptions containing credentials."""
    try:
        if photo is None:
            body, content_type = json.dumps(fields).encode(), "application/json"
        else:
            boundary = uuid.uuid4().hex
            parts = []
            for key, value in fields.items():
                encoded = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{encoded}\r\n'.encode())
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="downstream.png"\r\nContent-Type: image/png\r\n\r\n'.encode() + photo + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode())
            body, content_type = b"".join(parts), f"multipart/form-data; boundary={boundary}"
        request = Request(f"https://api.telegram.org/bot{token}/{method}", data=body,
                          headers={"Content-Type": content_type}, method="POST")
        with urlopen(request, timeout=8) as response:
            result = json.load(response)
        if result.get("ok") is not True:
            raise RuntimeError()
        return result.get("result")
    except Exception:
        raise RuntimeError("Telegram provider failed or timed out") from None


class TelegramAdapter:
    def __init__(self, transport=None, on_update=None):
        self._transport = transport or (lambda *args: _request(*args))
        self._on_update = on_update
        self.enabled = True
        self._lock = RLock()
        self._queue = Queue(maxsize=64)
        self._stop = Event()
        self._thread = None
        self._records = deque(maxlen=256)
        self._latest = None
        self._history = deque(maxlen=60)
        self._last_identity = None
        self._offset = 0
        self.callback_status = "IDLE"

    def _config(self):
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        try:
            allowed = {int(part.strip()) for part in raw.split(",") if part.strip()}
            if any(user <= 0 for user in allowed):
                allowed = set()
        except ValueError:
            allowed = set()
        return token, chat, allowed

    def settings_payload(self):
        token, chat, allowed = self._config()
        with self._lock:
            last = dict(self._latest or {})
            return {"configured": bool(token and chat and allowed),
                    "status": "Configured" if token and chat and allowed else "Not configured",
                    "chat_id": "***" + chat[-2:] if len(chat) > 2 else ("***" if chat else None),
                    "enabled": self.enabled, "deduplication": "ONE TELEGRAM NOTIFICATION PER INCIDENT",
                    "last_delivery": last.get("status", "NOT_CONFIGURED"),
                    "last_delivery_timestamp": last.get("timestamp"), "error": last.get("error"),
                    "incident_id": last.get("incident_id"), "acknowledged": last.get("acknowledged", False),
                    "callback_status": self.callback_status}

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
            self._thread = Thread(target=self._run, name="aquaflow-telegram", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        while True:
            try:
                record, _, _, _ = self._queue.get_nowait()
            except Empty:
                break
            self._finish(record, "FAILED", "Telegram worker stopped before delivery")
            self._queue.task_done()

    def notify(self, state, incident):
        with self._lock:
            identity = (state.get("state_identity") or {}).get("state_id")
            if identity is not None and identity != self._last_identity:
                ds = state.get("downstream") or {}
                self._history.append((identity, ds.get("flow_m3_s"), ds.get("capacity_m3_s")))
                self._last_identity = identity
            if not incident:
                for record in self._records:
                    record["closed"] = True
                return
            if any(r.get("incident_id") == incident["incident_id"] for r in self._records):
                return
            record = {"incident_id": incident["incident_id"], "timestamp": _now(),
                      "acknowledged": False, "closed": False}
            self._records.append(record)
            self._latest = record
            # Copy only notification fields, avoiding forecast arrays and other
            # unrelated simulation data on this nonblocking dispatch path.
            snapshot = {key: deepcopy(state.get(key)) for key in
                        ("downstream", "controller_mode", "forecast_source_selection")}
            snapshot["control"] = {key: (state.get("control") or {}).get(key) for key in
                                   ("downstream_reason", "downstream_status", "controller_type",
                                    "downstream_proposed_predicted_flow_mcm_day")}
            snapshot["reservoirs"] = {key: {"gate": value.get("gate")} for key, value in
                                      (state.get("reservoirs") or {}).items()}
            self._enqueue(record, snapshot, dict(incident), list(self._history), self.enabled)

    def close(self):
        with self._lock:
            for record in self._records:
                record["closed"] = True
            self._history.clear()
            self._last_identity = None

    def send_test(self):
        with self._lock:
            if self._latest and self._latest.get("status") == "PENDING":
                return dict(self._latest)
            record = {"timestamp": _now(), "kind": "TEST"}
            self._latest = record
            self._enqueue(record, None, None, [], True)
            return dict(record)

    def _enqueue(self, record, state, incident, history, enabled):
        token, chat, allowed = self._config()
        if not (token and chat and allowed):
            record.update(status="NOT_CONFIGURED", error=None)
            return
        if not enabled:
            record.update(status="NOT_CONFIGURED", error="Telegram alerts disabled")
            return
        if self._stop.is_set():
            record.update(status="FAILED", error="Telegram worker stopped")
            return
        record.update(status="PENDING", error=None)
        try:
            self._queue.put_nowait((record, state, incident, history))
        except Full:
            record.update(status="FAILED", error="Telegram delivery queue full")

    def _finish(self, record, status, error=None):
        with self._lock:
            record.update(status=status, timestamp=_now(), error=error)
        self._changed()

    def _deliver(self, record, state, incident, history):
        try:
            token, chat, allowed = self._config()
            if not (token and chat and allowed):
                self._finish(record, "NOT_CONFIGURED")
                return
            text = alert_message(state, incident) if incident else "AquaFlow Telegram notification test. No simulation state or gates were changed."
            photo = downstream_chart(history) if incident else None
            fields = {"chat_id": chat}
            if incident:
                fields["reply_markup"] = {"inline_keyboard": [[{"text": "ACKNOWLEDGE", "callback_data": "ack:" + incident["incident_id"]}]]}
            if photo:
                fields["caption"] = text[:1024]
                result = self._transport(token, "sendPhoto", fields, photo)
            else:
                fields["text"] = text
                result = self._transport(token, "sendMessage", fields)
            if not isinstance(result, dict) or not result.get("message_id"):
                raise RuntimeError()
            self._finish(record, "SENT")
        except Exception:
            self._finish(record, "FAILED", "Telegram provider failed or timed out")

    def handle_callback(self, callback):
        """Only called on provider-authenticated updates; never exposed via REST."""
        _, chat, allowed = self._config()
        user = (callback.get("from") or {}).get("id")
        message_chat = ((callback.get("message") or {}).get("chat") or {}).get("id")
        data = callback.get("data", "")
        with self._lock:
            if type(user) is not int or user not in allowed or str(message_chat) != chat:
                return "Unauthorized"
            record = next((r for r in self._records if "ack:" + r.get("incident_id", "") == data), None)
            if not record or record.get("closed") or record.get("status") != "SENT":
                return "Incident unavailable"
            if not record.get("acknowledged"):
                record.update(acknowledged=True, acknowledged_at=_now())
        self._changed()
        return "Incident acknowledged"

    def _poll(self):
        token, chat, allowed = self._config()
        if not (token and chat and allowed):
            return
        try:
            updates = self._transport(token, "getUpdates", {"offset": self._offset, "timeout": 0,
                                      "limit": 10, "allowed_updates": ["callback_query"]})
            for update in updates or []:
                if self._stop.is_set():
                    break
                self._offset = max(self._offset, int(update["update_id"]) + 1)
                callback = update.get("callback_query")
                if callback:
                    answer = self.handle_callback(callback)
                    self._transport(token, "answerCallbackQuery", {"callback_query_id": callback["id"], "text": answer})
            self.callback_status = "POLLING"
        except Exception:
            self.callback_status = "FAILED"

    def _run(self):
        next_poll = time.monotonic() + 2
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=.1)
            except Empty:
                item = None
            if item:
                self._deliver(*item)
                self._queue.task_done()
            if not self._stop.is_set() and time.monotonic() >= next_poll:
                self._poll()
                next_poll = time.monotonic() + 2
