"""Downstream and test-email notifications, isolated from simulation control."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.message import EmailMessage
from threading import Lock
import os
import smtplib
import ssl
import uuid
from src.notifications.telegram_alerts import TelegramAdapter
from src.notifications.discord_alerts import DiscordAdapter, GmailAdapter


def _config(recipient_override=None):
    host = os.getenv("AQUAFLOW_SMTP_HOST", "smtp.gmail.com").strip()
    try:
        port = int(os.getenv("AQUAFLOW_SMTP_PORT", "587"))
    except ValueError:
        port = 0
    username = os.getenv("AQUAFLOW_SMTP_USERNAME", "").strip()
    password = os.getenv("AQUAFLOW_SMTP_PASSWORD", "")
    sender = os.getenv("AQUAFLOW_ALERT_EMAIL_FROM", "").strip()
    recipient = (recipient_override or os.getenv("AQUAFLOW_ALERT_EMAIL_TO", "")).strip()
    if all((host, port, username, password, sender, recipient)) and 0 < port < 65536:
        return host, port, username, password, sender, recipient
    return None


def _send(config, subject, body):
    host, port, username, password, sender, recipient = config
    message = EmailMessage()
    message["Subject"], message["From"], message["To"] = subject, sender, recipient
    message.set_content(body)
    context = ssl.create_default_context()
    client = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    kwargs = {"context": context} if port == 465 else {}
    with client(host, port, timeout=20, **kwargs) as smtp:
        if port != 465:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(message)


def _safe_error(exc):
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "SMTP authentication failed; check Gmail username and App Password"
    if isinstance(exc, (TimeoutError, OSError)):
        return "SMTP connection failed or timed out"
    return f"Email delivery failed ({type(exc).__name__})"


def _mask(address):
    if "@" not in address:
        return "configured"
    name, domain = address.rsplit("@", 1)
    return (name[:1] + "***" if name else "***") + "@" + domain


def _email_data(state, incident_id, timestamp):
    ds = state.get("downstream") or {}
    reasons = [r.get("risk_reason") for r in (state.get("reservoirs") or {}).values() if r.get("risk_reason")]
    reason = ((state.get("control") or {}).get("downstream_reason") or "; ".join(dict.fromkeys(reasons))
              or "No separate downstream reason is supplied by authoritative state")
    lines = ["AquaFlow downstream flood-risk incident", f"Incident: {incident_id}",
             f"Timestamp: {timestamp}", f"Severity: {ds.get('status', 'UNKNOWN')}", f"Risk reason: {reason}",
             f"Downstream flow: {ds.get('flow_m3_s')} m³/s", f"Downstream capacity: {ds.get('capacity_m3_s')} m³/s",
             f"Mode: {state.get('controller_mode', 'UNKNOWN')}",
             f"Forecast status: {state.get('forecast_summary', {}).get('status', 'UNKNOWN')}",
             f"Forecast source: {state.get('forecast_source_selection', {}).get('selected', 'UNKNOWN')}",
             f"Downstream protection: {state.get('downstream_protection', 'UNKNOWN')}",
             f"Mass-balance status: {(state.get('mass_balance') or {}).get('status', 'UNKNOWN')}" ]
    for key, res in (state.get("reservoirs") or {}).items():
        name = res.get("repository_name") or key
        applied = None if res.get("gate") is None else res["gate"] * 100
        lines.extend([f"{name} storage/level: {res.get('water_level')} (ratio)",
                      f"{name} Requested Gate: {res.get('requested_gate_pct')}%",
                      f"{name} Applied Gate: {applied}%",
                      f"{name} Controlled release: {res.get('controlled_release')} m³/s",
                      f"{name} Current spill: {res.get('spill_mcm')} MCM"])
    return "\n".join(lines)


class DownstreamNotificationManager:
    """One notification per contiguous authoritative warning/critical period."""
    def __init__(self, transport=None, on_update=None):
        self.transport = transport or _send
        self.on_update = on_update
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aquaflow-mail")
        self.current = None
        self.last_incident = None
        self._future = None
        self.preferences = {"recipient": os.getenv("AQUAFLOW_ALERT_EMAIL_TO", "").strip(),
                            "high_enabled": True, "critical_enabled": True}
        self.last_test = None
        self._test_pending = False
        self._test_lock = Lock()
        self._test_future = None
        self.telegram = TelegramAdapter(on_update=self._changed)
        self.gmail = GmailAdapter(self._observe_gmail)
        self.discord = DiscordAdapter(on_update=self._changed)

    def _oauth(self):
        from src.dashboard.api.gmail_auth import gmail_oauth
        return gmail_oauth

    def _effective_config(self):
        recipient = self.preferences["recipient"].strip()
        if self._oauth().status()["connected"] and recipient:
            return ("OAUTH", 0, self._oauth().email, "", self._oauth().email, recipient)
        config = _config(recipient)
        if config:
            return config
        return None

    def settings_payload(self):
        oauth = self._oauth().status()
        smtp = _config(self.preferences["recipient"]) is not None
        connected = oauth["connected"]
        records = [("TEST", self.last_test), ("INCIDENT", self.current or self.last_incident)]
        kind, latest = max(((k, v) for k, v in records if v),
                           key=lambda item: item[1].get("timestamp", ""), default=(None, {}))
        return {"discord": self.discord.settings_payload(), "active_incidents": self.discord.active_incidents(),
                "telegram": self.telegram.settings_payload(), "gmail": {"status": "CONNECTED" if connected else ("SMTP CONFIGURED" if smtp else "NOT CONNECTED"),
                           "connected": connected, "oauth_configured": oauth["oauth_configured"],
                           "email": oauth["email"] if connected else (os.getenv("AQUAFLOW_SMTP_USERNAME", "") if smtp else None)},
                "recipient": self.preferences["recipient"], "high_enabled": self.preferences["high_enabled"],
                "critical_enabled": self.preferences["critical_enabled"], "deduplication": "ONE EMAIL PER INCIDENT",
                "last_notification_status": latest.get("status", "NOT_CONFIGURED"),
                "last_notification_timestamp": latest.get("timestamp"), "last_notification_kind": kind,
                "last_notification_error": latest.get("error"), "test_pending": self._test_pending,
                "configuration_error": None if oauth["oauth_configured"] else
                "Google OAuth is not configured. Administrator OAuth credentials are required to connect a Google account."}

    def update_preferences(self, recipient=None, high_enabled=None, critical_enabled=None, telegram_enabled=None,
                           discord_enabled=None):
        if discord_enabled is not None:
            self.discord.set_enabled(discord_enabled)
        if telegram_enabled is not None:
            self.telegram.enabled = bool(telegram_enabled)
        if recipient is not None:
            self.preferences["recipient"] = recipient.strip()
        if high_enabled is not None:
            self.preferences["high_enabled"] = bool(high_enabled)
        if critical_enabled is not None:
            self.preferences["critical_enabled"] = bool(critical_enabled)
        return self.settings_payload()

    def _send_message(self, config, subject, body):
        if config and config[0] == "OAUTH":
            self._oauth().send(subject, body, config[5])
        else:
            self.transport(config, subject, body)

    def send_test(self):
        recipient = self.preferences["recipient"].strip()
        if "@" not in recipient:
            raise ValueError("Configure a valid alert recipient before sending a test")
        oauth = self._oauth()
        if not oauth.status()["connected"]:
            raise RuntimeError("Connect Gmail with Google OAuth before sending a test email")
        config = ("OAUTH", 0, oauth.email, "", oauth.email, recipient)
        with self._test_lock:
            if self._test_pending:
                raise RuntimeError("A test email is already pending")
            self._test_pending = True
            self.last_test = {"status": "PENDING", "timestamp": _now()}
            self._test_future = self.executor.submit(self._deliver_test, config)
        self._changed()
        return dict(self.last_test)

    def _deliver_test(self, config):
        try:
            self._send_message(config, "AquaFlow Gmail notification test",
                               "This is a test email from AquaFlow. No simulation state or reservoir gates were changed.")
            result = {"status": "SENT", "timestamp": _now(), "error": None}
        except Exception as exc:
            result = {"status": "FAILED", "timestamp": _now(), "error": _safe_error(exc)}
        with self._test_lock:
            self.last_test = result
            self._test_pending = False
        self._changed()

    def _changed(self):
        if self.on_update:
            try: self.on_update()
            except Exception: pass

    def observe(self, state):
        self.gmail.notify(state)
        try:
            self.discord.notify(state, self.current)
        except Exception:
            pass
        # Additive dispatch: Gmail has already queued independently. No I/O here.
        try:
            self.telegram.notify(state, self.current)
        except Exception:
            pass

    def _observe_gmail(self, state):
        severity = str((state.get("downstream") or {}).get("status", "UNKNOWN")).upper()
        if severity not in ("WARNING", "CRITICAL"):
            self.current = None
            return
        pref = "high_enabled" if severity == "WARNING" else "critical_enabled"
        if self.current is not None:
            if severity == "CRITICAL":
                self.current["severity"] = "CRITICAL"
            if self.preferences[pref] and self.current.get("error") == "Alert disabled in notification settings":
                config = self._effective_config()
                if config:
                    self.current.update(status="PENDING", error=None)
                    self._queue_incident(config, state, self.current)
            return
        config = self._effective_config()
        now = _now()
        incident_id = "ds-" + uuid.uuid4().hex[:12]
        enabled = self.preferences[pref]
        self.current = {"severity": "HIGH" if severity == "WARNING" else "CRITICAL",
                        "status": "PENDING" if config and enabled else "NOT_CONFIGURED", "incident_id": incident_id,
                        "timestamp": now, "recipient": _mask(config[5]) if config else None,
                        "error": None if enabled else "Alert disabled in notification settings"}
        self.last_incident = self.current
        if config and enabled:
            self._queue_incident(config, state, self.current)

    def _queue_incident(self, config, state, incident):
        body = _email_data(state, incident["incident_id"], incident["timestamp"])
        self._future = self.executor.submit(self._deliver, config, body, incident)

    def _deliver(self, config, body, incident):
        try:
            self._send_message(config, f"AquaFlow — {incident['severity']} DOWNSTREAM FLOOD RISK", body)
            incident["status"] = "SENT"
        except Exception as exc:
            incident["status"] = "FAILED"
            incident["error"] = _safe_error(exc)
        self._changed()

    def payload(self):
        latest = self.current or self.last_incident
        payload = dict(latest) if latest else {"severity": "NORMAL", "status": "NOT_CONFIGURED",
                   "incident_id": None, "timestamp": None, "recipient": None, "error": None}
        records = [("TEST", self.last_test), ("INCIDENT", self.current or self.last_incident)]
        kind, last = max(((k, v) for k, v in records if v),
                         key=lambda item: item[1].get("timestamp", ""), default=(None, {}))
        payload["last_notification"] = {"kind": kind, "status": last.get("status", "NOT_CONFIGURED"),
                                         "timestamp": last.get("timestamp"), "error": last.get("error")}
        payload["test_email"] = dict(self.last_test) if self.last_test else None
        payload["test_pending"] = self._test_pending
        payload["telegram"] = self.telegram.settings_payload()
        payload["discord"] = self.discord.settings_payload()
        payload["active_incidents"] = self.discord.active_incidents()
        return payload

    def acknowledge(self, incident_id):
        return self.discord.acknowledge(incident_id)

    def close_if_resolved(self, severity, state=None):
        if str(severity).upper() not in ("WARNING", "CRITICAL"):
            self.current = None
            self.telegram.close()
            if str(severity).upper() == "NORMAL":
                self.discord.close(state)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
