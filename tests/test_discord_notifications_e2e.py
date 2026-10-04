"""Real local backend + browser, with Discord and Gmail delivery mocked."""
from copy import deepcopy
import json
from pathlib import Path
import socket
from threading import Thread
import time

import httpx
import pytest
import uvicorn

from src.dashboard.api import notifications
from src.notifications import discord_alerts as discord


def test_notification_settings_browser_e2e(monkeypatch, tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    from src.dashboard.api.app import app
    from src.dashboard.api.state_manager import sim_state

    webhook = "https://discord.com/api/webhooks/123456789/e2e-private-token"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", webhook)
    monkeypatch.setattr(notifications, "_config", lambda *args: ("smtp", 587, "u", "p", "from", "operator@example.com"))
    sent, emails, errors = [], [], []
    def provider(request):
        sent.append(request)
        return httpx.Response(200, json={"id": str(len(sent))})

    manager = notifications.DownstreamNotificationManager(transport=lambda *args: emails.append(args),
                                                          on_update=sim_state._schedule_notification_broadcast)
    manager.discord = discord.DiscordAdapter(transport=httpx.MockTransport(provider), on_update=manager._changed)
    # The headless browser can take several minutes on software-rendered CI;
    # keep the E2E's expected send count independent of that wall-clock delay.
    manager.discord.repeat_seconds = 3600
    snapshot = deepcopy(sim_state.get_adapted_state())
    snapshot["downstream"].update(status="WARNING", flow_m3_s=80, capacity_m3_s=90)
    snapshot["control"].update(downstream_proposed_predicted_flow_mcm_day=55, downstream_capacity_mcm_day=50)
    manager.observe(snapshot)
    manager._future.result(timeout=2)

    def published_state():
        result = deepcopy(snapshot)
        result["notification"] = manager.payload()
        return result

    monkeypatch.setattr(sim_state, "notification_manager", manager)
    monkeypatch.setattr(sim_state, "get_adapted_state", published_state)
    monkeypatch.setattr(sim_state, "running", False)
    monkeypatch.setattr(sim_state, "_notification_loop", None)
    before = (sim_state.sim_step_index, dict(sim_state.manual_gates), sim_state.bridge.cascade.network.timestep,
              {k: v.state.storage for k, v in sim_state.bridge.cascade.network.nodes.items()})
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started, "Local test backend failed to start"
        with playwright_api.sync_playwright() as playwright:
            if not Path(playwright.chromium.executable_path).is_file():
                pytest.skip("Playwright Chromium is not installed")
            browser = playwright.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader"])
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + "/?settings=notifications", wait_until="domcontentloaded")
                card = page.locator("[data-discord-card]")
                playwright_api.expect(card).to_be_visible(timeout=30000)
                playwright_api.expect(card.locator("[data-discord-status]")).to_have_text("Configured")
                playwright_api.expect(card.locator("[data-discord-webhook]")).to_have_text(discord.MASKED_WEBHOOK)
                playwright_api.expect(card.locator("[data-discord-delivery]")).to_have_text("SENT", timeout=15000)
                gmail = page.locator(".settings-card").filter(has=page.get_by_role("heading", name="GMAIL ALERTS", exact=True))
                assert card.bounding_box()["x"] > gmail.bounding_box()["x"]
                assert abs(card.bounding_box()["y"] - gmail.bounding_box()["y"]) < 2
                assert "reminders every 5 minutes" in card.inner_text()
                assert webhook not in page.content() and "e2e-private-token" not in page.content()

                with page.expect_response(lambda r: r.url.endswith("/api/notifications/discord/test")) as response:
                    card.locator("[data-discord-test]").click()
                assert response.value.status == 200
                playwright_api.expect(card.locator("[data-discord-delivery]")).to_have_text("SENT", timeout=15000)
                assert len(sent) == 2
                toggle = card.locator("[data-discord-enabled]")
                with page.expect_response(lambda r: r.url.endswith("/api/notifications/settings") and r.request.method == "PUT"):
                    toggle.set_checked(False)
                playwright_api.expect(toggle).not_to_be_checked()
                assert not manager.discord.enabled
                with page.expect_response(lambda r: r.url.endswith("/api/notifications/settings") and r.request.method == "PUT"):
                    toggle.set_checked(True)
                playwright_api.expect(toggle).to_be_checked()
                assert manager.discord.enabled

                acknowledge = page.locator("[data-active-incidents] button[data-ack-incident]")
                with page.expect_response(lambda r: r.url.endswith("/acknowledge")) as response:
                    acknowledge.click()
                assert response.value.json()["acknowledged"]
                playwright_api.expect(acknowledge).to_have_text("Acknowledged")
                playwright_api.expect(acknowledge).to_be_disabled()
                assert manager.discord.active_incidents()[0]["acknowledged"]
                # Recovery uses the backend incident, even after acknowledgement.
                snapshot["downstream"].update(status="NORMAL", flow_m3_s=20)
                with manager.discord._lock:
                    manager.discord._current["_last_attempt"] -= manager.discord.cooldown_seconds
                manager.observe(snapshot)
                # Browser rendering can be slow during the full model/browser
                # suite. The authoritative incident closes synchronously here;
                # allow the scheduled websocket refresh to reach the page.
                playwright_api.expect(page.locator("[data-active-incidents]")).to_have_text("No active downstream incidents.", timeout=45000)
                assert len(sent) == 3 and len(emails) == 1
                assert b"attachment://chart.png" in sent[-1].content
                assert b"filename=\"chart.png\"" in sent[-1].content
                assert b"RECOVERY" in sent[-1].content
                assert not errors
                page.screenshot(path=str(tmp_path / "discord_settings.png"), full_page=True)
                assert before == (sim_state.sim_step_index, dict(sim_state.manual_gates), sim_state.bridge.cascade.network.timestep,
                                  {k: v.state.storage for k, v in sim_state.bridge.cascade.network.nodes.items()})
                (tmp_path / "notification_e2e.json").write_text(json.dumps({
                    "notification_e2e": "PASS", "providers": "MOCKED", "browser": "Chromium",
                    "discord_deliveries": len(sent), "gmail_deliveries": len(emails), "page_errors": errors,
                    "checks": ["side-by-side cards", "masked webhook", "test delivery", "toggle",
                               "dashboard acknowledgement", "recovery attachment", "physics unchanged"]}, indent=2), encoding="utf-8")
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        manager.discord.stop()
        manager.telegram.stop()
        manager.executor.shutdown(wait=True)
    assert not thread.is_alive()
