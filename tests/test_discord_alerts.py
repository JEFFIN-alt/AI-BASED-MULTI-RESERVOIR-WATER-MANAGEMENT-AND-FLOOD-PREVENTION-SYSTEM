"""Discord contract and incident lifecycle tests; all provider traffic is mocked."""
import asyncio
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
import json
import logging
from threading import Event
import time

import httpx
import pytest

from src.dashboard.api import notifications
from src.notifications import discord_alerts as discord


WEBHOOK = "https://discord.com/api/webhooks/123456789/private-webhook-token"
PNG = b"\x89PNG\r\n\x1a\nmocked-chart"


def state(status="WARNING", day=1):
    return {"downstream": {"status": status, "flow_m3_s": 80, "capacity_m3_s": 90},
            "state_identity": {"state_id": f"step{day}-t{day}", "network_timestep": day},
            "control": {"downstream_proposed_predicted_flow_mcm_day": 55,
                        "downstream_capacity_mcm_day": 50},
            "reservoirs": {f"reservoir_{i}": {"gate": .1 * i} for i in range(1, 5)}}


class Clock:
    def __init__(self): self.value = 1000.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


@pytest.fixture
def channel(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)
    clock, calls, waits = Clock(), [], []

    async def transport(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "message-1"})

    adapter = discord.DiscordAdapter(transport=httpx.MockTransport(transport), clock=clock,
                                     chart=lambda history: PNG)

    async def wait(seconds):
        waits.append(seconds)
        clock.advance(seconds)

    adapter._wait = wait
    adapter.calls, adapter.waits, adapter.clock = calls, waits, clock
    yield adapter
    adapter.stop()


def pump(adapter):
    async def run():
        async with httpx.AsyncClient(transport=adapter._transport) as client:
            await adapter.deliver_pending(client)
    asyncio.run(run())


def incident(adapter, status="WARNING", day=1, incident_id="ds-test"):
    adapter.notify(state(status, day), {"incident_id": incident_id})


def multipart(request):
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode() + request.content)
    parts = {p.get_param("name", header="content-disposition"): p for p in message.iter_parts()}
    return json.loads(parts["payload_json"].get_payload(decode=True)), parts["files[0]"]


@pytest.mark.parametrize("url", ["", "http://discord.com/api/webhooks/1/token", "https://example.com/api/webhooks/1/token",
                                  "https://discord.com.evil.test/api/webhooks/1/token",
                                  "https://discord.com/api/webhooks/1", "https://user@discord.com/api/webhooks/1/token"])
def test_missing_or_invalid_configuration_never_sends(channel, monkeypatch, url):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
    incident(channel)
    assert channel.settings_payload()["status"] == "Not configured"
    assert channel.settings_payload()["masked_webhook"] is None
    assert channel.send_test()["status"] == "NOT_CONFIGURED"
    pump(channel)
    assert not channel.calls


@pytest.mark.parametrize("assignment", [
    'DISCORD_WEBHOOK_URL="{url}"', "DISCORD_WEBHOOK_URL='{url}'",
    "DISCORD_WEBHOOK_URL={url}", "export DISCORD_WEBHOOK_URL={url}",
    'DISCORD_WEBHOOK_URL="{url}" # local secret',
])
def test_local_env_configuration_is_masked_and_cached(monkeypatch, tmp_path, assignment):
    local = tmp_path / ".env"
    local.write_text("UNRELATED_SETTING=private\n" + assignment.format(url=WEBHOOK) + "\n", encoding="utf-8")
    monkeypatch.setattr(discord, "_ENV_FILE", local)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    adapter = discord.DiscordAdapter()
    assert adapter._config() == WEBHOOK
    assert adapter.settings_payload()["masked_webhook"] == discord.MASKED_WEBHOOK
    assert "UNRELATED_SETTING" not in discord.os.environ
    local.write_text("DISCORD_WEBHOOK_URL=\n", encoding="utf-8")
    assert adapter.settings_payload()["configured"]  # no repeated disk reads during simulation
    assert WEBHOOK not in json.dumps(adapter.settings_payload())


@pytest.mark.parametrize("override", ["", "https://discord.com/api/webhooks/123/override-token"])
def test_process_environment_takes_precedence_over_local_env(monkeypatch, tmp_path, override):
    local = tmp_path / ".env"
    local.write_text('DISCORD_WEBHOOK_URL="' + WEBHOOK + '"\n', encoding="utf-8")
    monkeypatch.setattr(discord, "_ENV_FILE", local)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", override)
    assert discord.DiscordAdapter()._config() == (override or None)


def test_missing_local_env_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(discord, "_ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert discord.DiscordAdapter().settings_payload()["configured"] is False


def test_secrets_are_masked_in_settings_records_and_library_logs(channel, caplog):
    caplog.set_level(logging.DEBUG)
    incident(channel)
    pump(channel)
    for name in ("httpx", "httpcore.http11"):
        logging.getLogger(name).info("request=%s", WEBHOOK + "?wait=true")
        logging.getLogger(name).debug("path=%s", "/api/webhooks/123456789/private-webhook-token")
    public = json.dumps([channel.settings_payload(), channel.active_incidents()])
    assert channel.settings_payload()["masked_webhook"] == discord.MASKED_WEBHOOK
    assert WEBHOOK not in public + caplog.text
    assert "private-webhook-token" not in public + caplog.text


@pytest.mark.parametrize("level", ["WARNING", "CRITICAL"])
def test_immediate_trigger_and_incident_dedup(channel, level):
    original = state(level)
    saved = deepcopy(original)
    channel.notify(original, {"incident_id": "ds-test"})
    assert channel.settings_payload()["last_delivery"] == "PENDING"
    pump(channel)
    for _ in range(20): channel.notify(original, {"incident_id": "ds-test"})
    pump(channel)
    assert len(channel.calls) == 1
    assert original == saved
    assert channel.active_incidents()[0]["severity"] == level


@pytest.mark.parametrize("level", ["NORMAL", "UNKNOWN", "ERROR"])
def test_other_levels_do_not_open_an_incident(channel, level):
    channel.notify(state(level), None)
    pump(channel)
    assert not channel.calls and not channel.active_incidents()


def test_embed_contract_multipart_and_attachment_reference(channel):
    incident(channel, "CRITICAL", 7)
    pump(channel)
    request = channel.calls[0]
    payload, photo = multipart(request)
    assert request.url.params["wait"] == "true"
    assert len(payload["embeds"]) == 1 and "components" not in payload
    embed = payload["embeds"][0]
    assert embed["color"] == discord.COLOURS["CRITICAL"]
    assert embed["image"]["url"] == "attachment://chart.png"
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    assert fields["DS flow / safe limit"] == "80.000 / 90.000 m³/s"
    assert fields["MPC predicted downstream / capacity (proposed)"] == "55.000 / 50.000 MCM/day"
    assert fields["Final gate actions (applied)"] == "A: 10.0% · B: 20.0% · C: 30.0% · D: 40.0%"
    assert fields["Sim day"] == "7"
    assert len(embed["description"]) <= 4096 and len(embed["fields"]) <= 25
    assert len(embed["footer"]["text"]) <= 2048
    assert "upstream" not in json.dumps(payload).lower() and "d-forecast" not in json.dumps(payload).lower()
    assert payload["allowed_mentions"] == {"parse": []}
    assert photo.get_filename() == "chart.png" and photo.get_content_type() == "image/png"
    assert photo.get_payload(decode=True) == PNG


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "55", True])
def test_unavailable_prediction_is_omitted(channel, value):
    snapshot = state()
    snapshot["control"]["downstream_proposed_predicted_flow_mcm_day"] = value
    payload = discord.build_message(snapshot, {"severity": "WARNING", "incident_id": "ds-test"})
    assert not any("MPC" in f["name"] for f in payload["embeds"][0]["fields"])
    assert payload["embeds"][0]["color"] == discord.COLOURS["WARNING"]


def test_capacity_conversion_and_all_embed_limits():
    snapshot = state()
    snapshot["control"].pop("downstream_capacity_mcm_day")
    payload = discord.build_message(snapshot, {"incident_id": "x" * 10000})
    embed = payload["embeds"][0]
    assert "7.776 MCM/day" in embed["fields"][1]["value"]
    assert len(embed["title"]) <= 256 and len(embed["description"]) <= 4096
    assert all(len(f["name"]) <= 256 and len(f["value"]) <= 1024 for f in embed["fields"])
    assert sum(len(f["name"]) + len(f["value"]) for f in embed["fields"]) + len(embed["footer"]["text"]) + len(embed["description"]) + len(embed["title"]) < 6000


def test_notifier_interface_and_existing_gmail_delegate(channel):
    calls = []
    gmail = discord.GmailAdapter(lambda snapshot: calls.append(snapshot))
    assert isinstance(gmail, discord.Notifier) and isinstance(channel, discord.Notifier)
    snapshot = state()
    gmail.start()
    gmail.notify(snapshot, {"incident_id": "ds-test"})
    gmail.stop()
    assert calls == [snapshot]


def test_thread_webhook_query_is_preserved(channel, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK + "?thread_id=12345")
    incident(channel)
    pump(channel)
    assert channel.calls[0].url.params["thread_id"] == "12345"
    assert channel.calls[0].url.params["wait"] == "true"


def test_chart_failure_does_not_crash_delivery_pump_or_expose_secrets(channel):
    channel._chart = lambda history: (_ for _ in ()).throw(RuntimeError(WEBHOOK))
    incident(channel)
    pump(channel)
    assert not channel.calls
    assert channel.settings_payload()["last_delivery"] == "FAILED"
    assert "private-webhook-token" not in json.dumps(channel.settings_payload())


def test_chart_uses_available_real_samples_and_history_is_bounded(channel):
    from PIL import Image
    for n in range(80): channel.notify(state("NORMAL", n), None)
    assert len(channel._history) == 60
    channel.notify(state("NORMAL", 79), None)
    assert len(channel._history) == 60
    for history in ([], [("step1", 80, 90)], list(channel._history)):
        image = Image.open(BytesIO(discord.downstream_chart(history)))
        assert image.format == "PNG" and image.width == 600 and image.height == 250


def test_reminders_use_backend_state_even_without_simulation_ticks_and_ack_stops_them(channel):
    incident(channel)
    pump(channel)
    channel.clock.advance(299)
    pump(channel)
    assert len(channel.calls) == 1
    channel.clock.advance(1)
    pump(channel)
    assert len(channel.calls) == 2
    assert "REMINDER" in multipart(channel.calls[-1])[0]["embeds"][0]["footer"]["text"]
    acknowledged = channel.acknowledge("ds-test")
    assert acknowledged["acknowledged"] and acknowledged["acknowledged_at"]
    assert channel.acknowledge("ds-test") == acknowledged
    channel.clock.advance(1000)
    pump(channel)
    assert len(channel.calls) == 2


def test_recovery_once_even_after_ack_and_new_incident_gets_new_alert(channel):
    incident(channel)
    pump(channel)
    channel.acknowledge("ds-test")
    channel.notify(state("NORMAL", 2), None)
    pump(channel)
    assert len(channel.calls) == 1  # recovery obeys the incident cooldown
    channel.clock.advance(30)
    pump(channel)
    payload, _ = multipart(channel.calls[-1])
    assert payload["embeds"][0]["color"] == discord.COLOURS["RECOVERY"]
    assert "RECOVERY" in payload["embeds"][0]["title"]
    assert not channel.active_incidents()
    channel.notify(state("NORMAL", 3), None)
    pump(channel)
    assert len(channel.calls) == 2
    with pytest.raises(ValueError): channel.acknowledge("ds-test")
    incident(channel, incident_id="ds-new")
    pump(channel)
    assert len(channel.calls) == 3


def test_unknown_telemetry_does_not_invent_recovery(channel):
    incident(channel)
    pump(channel)
    channel.notify(state("UNKNOWN", 2), None)
    channel.clock.advance(30)
    pump(channel)
    assert len(channel.calls) == 1
    incident(channel, incident_id="ds-new")
    pump(channel)
    assert len(channel.calls) == 2 and len(channel.active_incidents()) == 1


def test_escalation_obeys_cooldown_and_stays_in_same_incident(channel):
    incident(channel)
    pump(channel)
    channel.clock.advance(1)
    incident(channel, "CRITICAL", 2)
    pump(channel)
    assert len(channel.calls) == 1
    channel.clock.advance(29)
    pump(channel)
    assert len(channel.calls) == 2
    assert multipart(channel.calls[-1])[0]["embeds"][0]["color"] == discord.COLOURS["CRITICAL"]
    assert channel.active_incidents()[0]["incident_id"] == "ds-test"


def test_acknowledging_pending_alert_cancels_delivery(channel):
    incident(channel)
    channel.acknowledge("ds-test")
    pump(channel)
    assert not channel.calls
    with pytest.raises(KeyError): channel.acknowledge("unknown")


def test_toggle_cancels_pending_alert_and_reenable_does_not_reopen_incident(channel):
    incident(channel)
    channel.set_enabled(False)
    pump(channel)
    assert not channel.calls
    channel.set_enabled(True)
    pump(channel)
    assert len(channel.calls) == 1
    assert channel.active_incidents()[0]["incident_id"] == "ds-test"


def test_test_button_pending_dedupe_and_cooldown(channel):
    first = channel.send_test(state())
    assert first["status"] == "PENDING"
    assert channel.send_test(state()) == first
    pump(channel)
    assert len(channel.calls) == 1
    with pytest.raises(RuntimeError, match="cooldown"): channel.send_test()
    channel.clock.advance(30)
    channel.send_test()
    pump(channel)
    assert len(channel.calls) == 2


@pytest.mark.parametrize("body,headers,delay", [({"retry_after": 1.25}, {}, 1.25),
                                               ({}, {"Retry-After": "2.5"}, 2.5)])
def test_429_waits_provider_delay_and_retries_once(channel, body, headers, delay):
    def transport(request):
        channel.calls.append(request)
        return httpx.Response(429, json=body, headers=headers) if len(channel.calls) == 1 else httpx.Response(200, json={"id": "ok"})
    channel._transport = httpx.MockTransport(transport)
    incident(channel)
    pump(channel)
    assert len(channel.calls) == 2 and delay in channel.waits
    assert channel.settings_payload()["last_delivery"] == "SENT"


def test_persistent_429_is_capped_at_one_retry_and_blocks_later_delivery(channel):
    def transport(request):
        channel.calls.append(request)
        return httpx.Response(429, json={"retry_after": 40})
    channel._transport = httpx.MockTransport(transport)
    incident(channel)
    pump(channel)
    assert len(channel.calls) == 2 and channel.settings_payload()["last_delivery"] == "FAILED"
    channel.send_test()
    pump(channel)
    assert len(channel.calls) == 4 and channel.waits.count(40) == 3


def test_ack_during_429_wait_cancels_retry(channel):
    def transport(request):
        channel.calls.append(request)
        return httpx.Response(429, json={"retry_after": 1})
    channel._transport = httpx.MockTransport(transport)
    async def wait(seconds):
        if seconds: channel.acknowledge("ds-test")
        channel.clock.advance(seconds)
    channel._wait = wait
    incident(channel)
    pump(channel)
    assert len(channel.calls) == 1


def test_successful_exhausted_bucket_delays_next_request(channel):
    def transport(request):
        channel.calls.append(request)
        return httpx.Response(200, json={"id": "ok"}, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "4"})
    channel._transport = httpx.MockTransport(transport)
    incident(channel)
    channel.send_test()
    pump(channel)
    assert len(channel.calls) == 2 and 4 in channel.waits


@pytest.mark.parametrize("code", [400, 401, 403, 404, 500])
def test_provider_errors_do_not_retry_or_expose_response_secrets(channel, code, caplog):
    def transport(request):
        channel.calls.append(request)
        return httpx.Response(code, json={"message": WEBHOOK})
    channel._transport = httpx.MockTransport(transport)
    incident(channel)
    pump(channel)
    assert len(channel.calls) == 1
    assert channel.settings_payload()["last_delivery"] == "FAILED"
    assert "private-webhook-token" not in json.dumps(channel.settings_payload()) + caplog.text
    if code in (401, 403, 404):
        channel.clock.advance(300)
        pump(channel)
        assert len(channel.calls) == 1


@pytest.mark.parametrize("error", [httpx.ReadTimeout(WEBHOOK), RuntimeError(WEBHOOK)])
def test_timeouts_and_exceptions_are_sanitized(channel, error, caplog):
    channel._transport = httpx.MockTransport(lambda request: (_ for _ in ()).throw(error))
    incident(channel)
    pump(channel)
    assert channel.settings_payload()["last_delivery"] == "FAILED"
    assert "private-webhook-token" not in json.dumps(channel.settings_payload()) + caplog.text


@pytest.mark.parametrize("response", [httpx.Response(204), httpx.Response(200, json={}), httpx.Response(200, text="not-json")])
def test_missing_confirmation_never_reports_sent(channel, response):
    channel._transport = httpx.MockTransport(lambda request: response)
    incident(channel)
    pump(channel)
    assert channel.settings_payload()["last_delivery"] == "FAILED"


def test_queue_bounded_and_shutdown_marks_pending_work(channel):
    for index in range(180): incident(channel, day=index, incident_id=f"ds-{index}")
    assert channel._queue.qsize() == channel._queue.maxsize == 32
    assert len(channel._records) <= 128
    assert channel.settings_payload()["error"] == "Discord delivery queue full"
    channel.stop()
    assert channel._queue.empty()
    assert not any(r["_pending"] for r in channel._records.values())


@pytest.mark.parametrize("discord_failure", [True, False, "unconfigured"])
def test_gmail_delivery_unchanged_and_independent(channel, monkeypatch, discord_failure):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "operator@example.com"))
    emails = []
    manager = notifications.DownstreamNotificationManager(transport=lambda *args: emails.append(args))
    manager.discord = channel
    if discord_failure == "unconfigured": monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    elif discord_failure:
        channel._transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"error": WEBHOOK}))
    try:
        manager.observe(state())
        manager._future.result(timeout=2)
        incident_id = manager.current["incident_id"]
        pump(channel)
        manager.observe(state("CRITICAL", 2))
        assert manager.current["incident_id"] == incident_id
        assert len(emails) == 1 and manager.payload()["status"] == "SENT"
        assert manager.payload()["discord"]["last_delivery"] == (
            "NOT_CONFIGURED" if discord_failure == "unconfigured" else "FAILED" if discord_failure else "SENT")
        assert "Downstream flow: 80" in emails[0][2]
    finally:
        manager.executor.shutdown(wait=True)


def test_gmail_failure_does_not_prevent_discord(channel, monkeypatch):
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "operator@example.com"))
    manager = notifications.DownstreamNotificationManager(transport=lambda *args: (_ for _ in ()).throw(RuntimeError("mail secret")))
    manager.discord = channel
    try:
        manager.observe(state())
        manager._future.result(timeout=2)
        pump(channel)
        assert manager.payload()["status"] == "FAILED"
        assert manager.payload()["discord"]["last_delivery"] == "SENT"
    finally:
        manager.executor.shutdown(wait=True)


def test_worker_is_async_nonblocking_and_uses_short_timeouts(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)
    entered, release, finished = Event(), Event(), Event()
    requests = []
    async def transport(request):
        requests.append(request)
        entered.set()
        while not release.is_set(): await asyncio.sleep(.01)
        finished.set()
        return httpx.Response(200, json={"id": "ok"})
    adapter = discord.DiscordAdapter(transport=httpx.MockTransport(transport), chart=lambda history: PNG)
    adapter.start()
    thread = adapter._thread
    adapter.start()
    assert adapter._thread is thread
    try:
        incident(adapter)
        assert entered.wait(2)
        # Both dispatch and state access finish while the provider is blocked.
        adapter.notify(state("CRITICAL", 2), {"incident_id": "ds-test"})
        assert adapter.active_incidents()[0]["severity"] == "CRITICAL"
        assert not finished.is_set()
        assert requests[0].extensions["timeout"] == {"connect": 2, "read": 3, "write": 3, "pool": 3}
    finally:
        release.set()
        adapter.stop()
    assert not thread.is_alive()


def test_shutdown_interrupts_long_rate_limit_wait(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)
    entered = Event()
    def transport(request):
        entered.set()
        return httpx.Response(429, json={"retry_after": 3600})
    adapter = discord.DiscordAdapter(transport=httpx.MockTransport(transport), chart=lambda history: PNG)
    adapter.start()
    incident(adapter)
    assert entered.wait(2)
    end = time.monotonic()
    adapter.stop()
    assert time.monotonic() - end < 1 and not adapter._thread.is_alive()


def test_settings_test_and_ack_endpoints_have_no_physics_authority(channel, monkeypatch):
    from fastapi.testclient import TestClient
    from src.dashboard.api.app import app
    from src.dashboard.api.state_manager import sim_state

    manager = notifications.DownstreamNotificationManager()
    manager.discord = channel
    monkeypatch.setattr(sim_state, "notification_manager", manager)
    monkeypatch.setattr(sim_state, "get_adapted_state", lambda: state())
    monkeypatch.setattr(sim_state, "_notification_loop", None)
    before = (sim_state.sim_step_index, dict(sim_state.manual_gates), sim_state.bridge.cascade.network.timestep,
              {k: v.state.storage for k, v in sim_state.bridge.cascade.network.nodes.items()})
    client = TestClient(app)
    try:
        manager.observe(state())
        active = client.get("/api/notifications/settings").json()
        incident_id = active["active_incidents"][0]["incident_id"]
        assert active["discord"]["status"] == "Configured" and WEBHOOK not in json.dumps(active)
        assert client.post(f"/api/notifications/incidents/{incident_id}/acknowledge").json()["acknowledged"]
        assert client.post("/api/notifications/incidents/missing/acknowledge").status_code == 404
        assert client.put("/api/notifications/settings", json={"discord_enabled": False}).json()["discord"]["enabled"] is False
        assert client.post("/api/notifications/discord/test").json()["status"] == "PENDING"
        pump(channel)
        assert client.get("/api/notifications/settings").json()["discord"]["last_delivery"] == "SENT"
        assert client.post("/api/notifications/discord/test").status_code == 409
        channel.notify(state("NORMAL", 2), None)
        assert client.post(f"/api/notifications/incidents/{incident_id}/acknowledge").status_code == 409
        assert before == (sim_state.sim_step_index, dict(sim_state.manual_gates), sim_state.bridge.cascade.network.timestep,
                          {k: v.state.storage for k, v in sim_state.bridge.cascade.network.nodes.items()})
    finally:
        client.close()
        manager.executor.shutdown(wait=True)
