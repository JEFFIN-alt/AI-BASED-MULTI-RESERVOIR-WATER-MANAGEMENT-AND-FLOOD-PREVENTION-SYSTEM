"""Focused channel tests. Every external Telegram request is mocked."""
from copy import deepcopy
from threading import Event
import json
import time

import pytest

from src.notifications import telegram_alerts as telegram
from src.dashboard.api import notifications


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123456")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42, 43")
    # Guard against accidental real network even in application startup tests.
    monkeypatch.setattr(telegram, "_request", lambda token, method, fields, photo=None:
                        [] if method == "getUpdates" else {"message_id": 101})


def state(status="WARNING", index=1):
    return {"downstream": {"status": status, "flow_m3_s": 80, "capacity_m3_s": 90, "utilisation": .889},
            "state_identity": {"state_id": f"step{index}"}, "controller_mode": "AI",
            "control": {"downstream_reason": "Downstream capacity warning", "downstream_status": "PROTECTED"},
            "forecast_source_selection": {"selected": "SIMULATION"},
            "reservoirs": {f"reservoir_{i}": {"gate": .1 * i} for i in range(1, 5)}}


def wait_for(adapter, status="SENT"):
    end = time.monotonic() + 4
    while adapter.settings_payload()["last_delivery"] == "PENDING" and time.monotonic() < end:
        time.sleep(.01)
    assert adapter.settings_payload()["last_delivery"] == status


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "to@example.com"))
    result = notifications.DownstreamNotificationManager(transport=lambda *args: None)
    result.telegram.start()
    yield result
    result.telegram.stop()
    result.executor.shutdown(wait=True)


def test_configuration_masking_and_no_secret(manager, caplog):
    payload = manager.settings_payload()
    assert payload["telegram"]["configured"] is True
    assert payload["telegram"]["chat_id"] == "***56"
    assert "private-bot-token" not in json.dumps(payload) + caplog.text
    assert "-100123456" not in json.dumps(payload)


@pytest.mark.parametrize("missing", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USER_IDS"])
def test_missing_configuration(manager, monkeypatch, missing):
    monkeypatch.delenv(missing)
    assert manager.telegram.send_test()["status"] == "NOT_CONFIGURED"
    manager.observe(state())
    manager._future.result(2)
    assert manager.payload()["status"] == "SENT"
    assert manager.telegram.settings_payload()["configured"] is False


def test_invalid_allowlist_fails_closed(manager, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42,invalid")
    assert manager.telegram.send_test()["status"] == "NOT_CONFIGURED"


@pytest.mark.parametrize("severity", ["WARNING", "CRITICAL"])
def test_trigger_and_single_incident(manager, severity):
    calls = []
    manager.telegram._transport = lambda *args: calls.append(args) or {"message_id": 1}
    manager.observe(state(severity))
    wait_for(manager.telegram)
    incident = manager.current["incident_id"]
    for i in range(2, 6):
        manager.observe(state("CRITICAL", i))
    assert len(calls) == 1
    assert manager.telegram.settings_payload()["incident_id"] == incident
    fields = calls[0][2]
    assert fields["reply_markup"]["inline_keyboard"][0][0] == {"text": "ACKNOWLEDGE", "callback_data": "ack:" + incident}
    assert "HIGH RISK" in fields["text"] if severity == "WARNING" else "CRITICAL RISK" in fields["text"]
    manager.observe(state("NORMAL", 7))
    assert manager.current is None
    manager.observe(state(severity, 8))
    wait_for(manager.telegram)
    assert len(calls) == 2
    assert manager.current["incident_id"] != incident


@pytest.mark.parametrize("error", [RuntimeError("private-bot-token"), TimeoutError("private-bot-token")])
def test_failure_timeout_gmail_independent(manager, error, caplog):
    manager.telegram._transport = lambda *args: (_ for _ in ()).throw(error)
    manager.observe(state())
    wait_for(manager.telegram, "FAILED")
    manager._future.result(2)
    assert manager.payload()["status"] == "SENT"
    assert "private-bot-token" not in json.dumps(manager.settings_payload()) + caplog.text


def test_gmail_failure_telegram_success(manager):
    manager.transport = lambda *args: (_ for _ in ()).throw(RuntimeError())
    manager.observe(state())
    wait_for(manager.telegram)
    manager._future.result(2)
    assert manager.payload()["status"] == "FAILED"


def test_gmail_and_telegram_both_succeed(manager):
    manager.observe(state())
    wait_for(manager.telegram)
    manager._future.result(2)
    assert manager.payload()["status"] == "SENT"


def callback(manager, user=42, chat=-100123456, data=None):
    return {"id": "query1", "from": {"id": user}, "message": {"chat": {"id": chat}},
            "data": data or "ack:" + manager.current["incident_id"]}


@pytest.mark.parametrize("user,chat", [(99, -100123456), (42, 999), (True, -100123456), ("42", -100123456)])
def test_unauthorized_callback(manager, user, chat):
    manager.observe(state())
    wait_for(manager.telegram)
    assert manager.telegram.handle_callback(callback(manager, user, chat)) == "Unauthorized"
    assert not manager.telegram.settings_payload()["acknowledged"]


def test_authorized_ack_has_no_control_authority(manager):
    from src.dashboard.api.state_manager import sim_state
    before = (dict(sim_state.manual_gates), sim_state.mode, sim_state.sim_step_index,
              sim_state._current_gate_pct(), sim_state.bridge.cascade.network.timestep)
    snapshot = state()
    original = deepcopy(snapshot)
    manager.observe(snapshot)
    wait_for(manager.telegram)
    for _ in range(2):
        assert manager.telegram.handle_callback(callback(manager)) == "Incident acknowledged"
    assert manager.telegram.settings_payload()["acknowledged"] is True
    assert snapshot == original
    assert before == (dict(sim_state.manual_gates), sim_state.mode, sim_state.sim_step_index,
                      sim_state._current_gate_pct(), sim_state.bridge.cascade.network.timestep)
    manager.observe(state("NORMAL", 2))
    assert manager.telegram.handle_callback({"from": {"id": 42}, "message": {"chat": {"id": -100123456}},
                                             "data": "ack:missing"}) == "Incident unavailable"


def test_poll_authentication_route_and_response(manager):
    manager.observe(state())
    wait_for(manager.telegram)
    calls = []
    def request(token, method, fields, photo=None):
        calls.append((method, fields))
        if method == "getUpdates":
            return [{"update_id": 17, "callback_query": callback(manager, user=99)},
                    {"update_id": 18, "callback_query": callback(manager)}]
        return True
    manager.telegram._transport = request
    manager.telegram._poll()
    assert manager.telegram._offset == 19
    assert [c[1]["text"] for c in calls if c[0] == "answerCallbackQuery"] == ["Unauthorized", "Incident acknowledged"]
    assert manager.telegram.settings_payload()["acknowledged"]


def test_test_endpoint_and_settings_no_simulation_changes(manager, monkeypatch):
    from fastapi.testclient import TestClient
    from src.dashboard.api.app import app
    from src.dashboard.api.state_manager import sim_state
    monkeypatch.setattr(sim_state, "notification_manager", manager)
    before = (dict(sim_state.manual_gates), sim_state.sim_step_index, sim_state.mode)
    with TestClient(app) as client:
        response = client.post("/api/notifications/telegram/test")
        assert response.status_code == 200
        assert response.json()["status"] in {"PENDING", "SENT"}
        wait_for(manager.telegram)
        assert client.get("/api/notifications/settings").json()["telegram"]["last_delivery"] == "SENT"
        assert client.put("/api/notifications/settings", json={"telegram_enabled": False}).json()["telegram"]["enabled"] is False
        assert client.post("/api/notifications/telegram/callback", json=callback(manager, data="ack:spoof")).status_code in {404, 405}
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
        assert client.post("/api/notifications/telegram/test").json()["status"] == "NOT_CONFIGURED"
        assert "private-bot-token" not in client.get("/api/notifications/settings").text
    assert before == (dict(sim_state.manual_gates), sim_state.sim_step_index, sim_state.mode)
    assert not manager.telegram._thread.is_alive()


def test_chart_real_history_and_insufficient_history(manager):
    assert telegram.downstream_chart([]) is None
    assert telegram.downstream_chart([("a", 1, 10), ("b", 2, 10)]) is None
    history = [("a", 1, 10), ("b", 2, 10), ("c", 3, 10)]
    assert telegram.downstream_chart(history).startswith(b"\x89PNG")
    calls = []
    manager.telegram._transport = lambda *args: calls.append(args) or {"message_id": 1}
    for index in (1, 1, 2):
        manager.observe(state("NORMAL", index))
    assert len(manager.telegram._history) == 2
    manager.observe(state("WARNING", 3))
    wait_for(manager.telegram)
    assert calls[0][1] == "sendPhoto"
    assert calls[0][3].startswith(b"\x89PNG")


def test_message_contains_only_available_authoritative_fields():
    message = telegram.alert_message(state(), {"severity": "HIGH", "incident_id": "ds-a", "timestamp": "now"})
    for expected in ("80", "90", "0.889", "Mode: AUTO", "A: 10.0%", "D: 40.0%", "Downstream capacity warning"):
        assert expected in message
    for absent in ("rainfall", "upstream", "MPC predicted", "D forecast"):
        assert absent not in message
    snapshot = state()
    snapshot["control"].update(controller_type="MPC", downstream_proposed_predicted_flow_mcm_day=1.23)
    assert "MPC predicted downstream (proposed): 1.23 MCM/day" in telegram.alert_message(
        snapshot, {"severity": "CRITICAL", "incident_id": "ds-b", "timestamp": "now"})


def test_background_nonblocking_and_clean_shutdown(manager):
    entered, release = Event(), Event()
    def transport(*args):
        entered.set()
        assert release.wait(3)
        return {"message_id": 1}
    manager.telegram._transport = transport
    start = time.monotonic()
    manager.observe(state())
    assert time.monotonic() - start < .5
    assert entered.wait(1)
    assert manager.telegram.settings_payload()["last_delivery"] == "PENDING"
    release.set()
    manager.telegram.stop()
    assert not manager.telegram._thread.is_alive()
    assert manager.telegram._queue.unfinished_tasks == 0


def test_toggle_does_not_reopen_incident(manager):
    manager.telegram.enabled = False
    manager.observe(state())
    assert manager.telegram.settings_payload()["last_delivery"] == "NOT_CONFIGURED"
    manager.telegram.enabled = True
    manager.observe(state("CRITICAL", 2))
    assert manager.telegram.settings_payload()["last_delivery"] == "NOT_CONFIGURED"
    manager.observe(state("NORMAL", 3))
    manager.observe(state("CRITICAL", 4))
    wait_for(manager.telegram)


@pytest.mark.parametrize("payload", [{"ok": False, "description": "private-bot-token"}, {"ok": True, "result": {"message_id": 3}}])
def test_http_provider_confirmation_and_sanitization(monkeypatch, payload):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode()
    captured = []
    monkeypatch.setattr(telegram, "urlopen", lambda request, timeout: captured.append((request, timeout)) or Response())
    # Fixture replaces _request: load original from its captured implementation below.
    if payload["ok"]:
        assert ORIGINAL_REQUEST("private-bot-token", "sendMessage", {"text": "test"}) == {"message_id": 3}
    else:
        with pytest.raises(RuntimeError, match="Telegram provider failed or timed out") as error:
            ORIGINAL_REQUEST("private-bot-token", "sendMessage", {})
        assert "private-bot-token" not in str(error.value)
    assert captured[0][1] == 8


ORIGINAL_REQUEST = telegram._request


@pytest.mark.parametrize("provider_result", [None, {}, {"ok": True}])
def test_no_provider_confirmation_never_reports_sent(manager, provider_result):
    manager.telegram._transport = lambda *args: provider_result
    manager.telegram.send_test()
    wait_for(manager.telegram, "FAILED")


def test_pending_test_is_deduplicated(manager):
    entered, release = Event(), Event()
    calls = []
    def transport(*args):
        calls.append(args)
        entered.set()
        release.wait(2)
        return {"message_id": 1}
    manager.telegram._transport = transport
    try:
        assert manager.telegram.send_test()["status"] == "PENDING"
        assert entered.wait(1)
        assert manager.telegram.send_test()["status"] == "PENDING"
    finally:
        release.set()
    wait_for(manager.telegram)
    assert len(calls) == 1


def test_shutdown_marks_queued_work_failed():
    adapter = telegram.TelegramAdapter()
    assert adapter.send_test()["status"] == "PENDING"
    adapter.stop()
    assert adapter.settings_payload()["last_delivery"] == "FAILED"
    assert adapter._queue.unfinished_tasks == 0


def test_transport_timeout_is_sanitized(monkeypatch):
    monkeypatch.setattr(telegram, "urlopen", lambda *args, **kwargs:
                        (_ for _ in ()).throw(TimeoutError("private-bot-token")))
    with pytest.raises(RuntimeError) as error:
        ORIGINAL_REQUEST("private-bot-token", "sendMessage", {})
    assert "private-bot-token" not in str(error.value)


def test_multipart_chart_request(monkeypatch):
    captured = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"ok":true,"result":{"message_id":9}}'
    monkeypatch.setattr(telegram, "urlopen", lambda request, timeout: captured.append(request) or Response())
    assert ORIGINAL_REQUEST("private-bot-token", "sendPhoto", {"chat_id": "-1", "caption": "Alert",
                            "reply_markup": {"inline_keyboard": []}}, b"real-png-data") == {"message_id": 9}
    assert "multipart/form-data" in captured[0].headers["Content-type"]
    assert b"real-png-data" in captured[0].data


def test_closed_incident_cannot_be_acknowledged(manager):
    manager.observe(state())
    wait_for(manager.telegram)
    query = callback(manager)
    manager.close_if_resolved("NORMAL")
    assert manager.telegram.handle_callback(query) == "Incident unavailable"


def test_queue_capacity_is_bounded_and_failure_isolated():
    adapter = telegram.TelegramAdapter()
    for index in range(65):
        adapter.notify(state(index=index), {"incident_id": f"ds-{index}", "severity": "HIGH", "timestamp": "now"})
    assert adapter._queue.qsize() == 64
    assert adapter.settings_payload()["last_delivery"] == "FAILED"
    adapter.stop()
