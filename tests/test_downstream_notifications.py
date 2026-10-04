from src.dashboard.api import notifications


def state(status="CRITICAL"):
    return {"downstream": {"status": status, "flow_m3_s": 80, "capacity_m3_s": 70},
            "controller_mode": "AI", "forecast_summary": {"status": "DEMONSTRATION_ONLY"},
            "reservoirs": {"reservoir_1": {"repository_name": "Test Reservoir",
                "water_level": 0.72, "gate": 0.45, "requested_gate_pct": 80,
                "controlled_release": 12, "spill_mcm": 1.2, "risk_reason": "High level"}}}


def test_incident_dedup_reset_and_payload(monkeypatch):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "operator@example.com"))
    sent = []
    manager = notifications.DownstreamNotificationManager(transport=lambda c, s, b: sent.append((s, b)))
    manager.preferences["recipient"] = "operator@example.com"
    manager.observe(state("NORMAL"))
    assert not sent
    manager.observe(state("WARNING"))
    incident = manager.payload()["incident_id"]
    manager._future.result(timeout=2)
    assert len(sent) == 1
    manager.observe(state())
    assert manager.payload()["incident_id"] == incident
    assert manager.payload()["status"] == "SENT"
    manager.observe(state())
    assert manager.payload()["incident_id"] == incident
    assert len(sent) == 1
    body = sent[0][1]
    assert "Downstream flow: 80" in body
    assert "Requested Gate: 80%" in body and "Applied Gate: 45.0%" in body
    manager.observe(state("NORMAL"))
    manager.observe(state())
    manager._future.result(timeout=2)
    assert len(sent) == 2
    assert manager.payload()["incident_id"] != incident
    manager.executor.shutdown(wait=True)


def test_failure_and_missing_configuration(monkeypatch):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "sir@example.com"))
    manager = notifications.DownstreamNotificationManager(transport=lambda *args: (_ for _ in ()).throw(RuntimeError("secret password")))
    manager.preferences["recipient"] = "sir@example.com"
    manager.observe(state())
    # Delivery has no authority over control-mode state.
    assert state()["controller_mode"] == "AI"
    manager._future.result(timeout=2)
    assert manager.payload()["status"] == "FAILED"
    assert "password" not in manager.payload()["error"]
    manager.executor.shutdown(wait=True)
    monkeypatch.setattr(notifications, "_config", lambda *args: None)
    manager = notifications.DownstreamNotificationManager()
    manager.observe(state())
    assert manager.payload()["status"] == "NOT_CONFIGURED"
    manager.observe(state("NORMAL"))
    manager.observe(state())
    assert manager.payload()["status"] == "NOT_CONFIGURED"
    manager.executor.shutdown(wait=True)


def test_authoritative_state_publishes_notification_block():
    from src.dashboard.api.state_manager import sim_state

    payload = sim_state.get_adapted_state()
    assert "notification" in payload
    assert payload["notification"]["status"] in {
        "NOT_CONFIGURED", "PENDING", "SENT", "FAILED"
    }
    sim_state._invalidate_pipeline_cache()


def test_test_email_requires_recipient_and_prevents_duplicate_pending(monkeypatch):
    from threading import Event
    class OAuth:
        email = "sender@gmail.com"
        def status(self): return {"connected": True}
        def send(self, subject, body, recipient):
            entered.set()
            release.wait(2)
    entered, release = Event(), Event()
    manager = notifications.DownstreamNotificationManager()
    manager._oauth = lambda: OAuth()
    try:
        manager.send_test()
        assert False, "recipient is required"
    except ValueError:
        pass
    manager.preferences["recipient"] = "target@example.com"
    result = manager.send_test()
    assert result["status"] == "PENDING"
    assert entered.wait(1)
    try:
        manager.send_test()
        assert False, "duplicate pending test must be rejected"
    except RuntimeError:
        pass
    release.set()
    manager._test_future.result(timeout=2)
    assert manager.last_test["status"] == "SENT"
    manager.executor.shutdown(wait=True)


def test_alert_preferences_never_reopen_active_incident(monkeypatch):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "target@example.com"))
    sent = []
    manager = notifications.DownstreamNotificationManager(transport=lambda *args: sent.append(args))
    manager.preferences.update(recipient="target@example.com", high_enabled=False)
    manager.observe(state("WARNING"))
    incident_id = manager.payload()["incident_id"]
    manager.observe(state("WARNING"))
    assert manager.payload()["incident_id"] == incident_id
    assert manager.payload()["status"] == "NOT_CONFIGURED"
    assert not sent
    manager.preferences["high_enabled"] = True
    manager.observe(state("WARNING"))
    manager._future.result(timeout=2)
    assert manager.payload()["incident_id"] == incident_id
    assert len(sent) == 1
    manager.executor.shutdown(wait=True)


def test_test_email_requires_oauth_and_reports_gmail_api_failure(monkeypatch):
    manager = notifications.DownstreamNotificationManager()
    manager.preferences["recipient"] = "operator@example.com"
    class Disconnected:
        email = None
        def status(self): return {"connected": False}
    manager._oauth = lambda: Disconnected()
    try:
        manager.send_test()
        assert False, "SMTP alone must not count as connected Gmail OAuth"
    except RuntimeError as exc:
        assert "OAuth" in str(exc)

    class FailingOAuth:
        email = "sender@gmail.com"
        def status(self): return {"connected": True}
        def send(self, *args): raise RuntimeError("private access token leaked")
    manager._oauth = lambda: FailingOAuth()
    try:
        manager.send_test()
        manager._test_future.result(timeout=2)
        assert manager.last_test["status"] == "FAILED"
        assert "access token" not in manager.last_test["error"]
    finally:
        manager.executor.shutdown(wait=True)


def test_oauth_missing_configuration_is_explicit(monkeypatch):
    from src.dashboard.api.gmail_auth import GmailOAuthManager
    for name in ("AQUAFLOW_GOOGLE_CLIENT_ID", "AQUAFLOW_GOOGLE_CLIENT_SECRET", "AQUAFLOW_GOOGLE_REDIRECT_URI"):
        monkeypatch.delenv(name, raising=False)
    oauth = GmailOAuthManager()
    assert oauth.authorization_url() is None
    assert oauth.status() == {"oauth_configured": False, "connected": False, "email": None, "auth_error": False}
