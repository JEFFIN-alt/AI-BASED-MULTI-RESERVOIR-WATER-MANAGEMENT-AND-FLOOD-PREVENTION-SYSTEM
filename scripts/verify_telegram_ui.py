"""Playwright UI verification against a real local AquaFlow backend.

No Telegram responses or browser requests are mocked. If credentials are
configured, the button sends exactly one controlled real test alert.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "telegram_verification"
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    configured = all(os.getenv(key, "").strip() for key in
                     ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USER_IDS"))
    errors = []
    log = (output / "backend.log").open("w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.dashboard.api.app:app",
                               "--host", "127.0.0.1", "--port", str(port)], cwd=root,
                              stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(100):
            try:
                with urlopen(base + "/api/notifications/settings", timeout=1) as response:
                    initial = json.load(response)
                break
            except OSError:
                if server.poll() is not None:
                    raise RuntimeError("Backend failed to start; inspect backend.log")
                time.sleep(.3)
        else:
            raise RuntimeError("Backend startup timed out")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto(base + "/?settings=notifications", wait_until="networkidle")
            card = page.locator("[data-telegram-card]")
            expect(card).to_be_visible(timeout=30000)
            expected = initial["telegram"]
            expect(card.locator("[data-telegram-status]")).to_have_text(expected["status"])
            expect(card.locator("[data-telegram-chat]")).to_have_text(expected["chat_id"] or "—")
            assert "One notification per incident" in card.inner_text()
            assert "It does not control the simulation." in card.inner_text()
            with page.expect_response(lambda response: response.url.endswith("/api/notifications/telegram/test")) as test_response:
                card.locator("[data-telegram-test]").click()
            result = test_response.value.json()
            assert result["status"] in {"NOT_CONFIGURED", "PENDING", "SENT", "FAILED"}
            terminal = "SENT" if expected["configured"] else "NOT_CONFIGURED"
            expect(card.locator("[data-telegram-delivery]")).to_have_text(terminal, timeout=30000)
            with urlopen(base + "/api/notifications/settings") as response:
                backend_state = json.load(response)
            assert backend_state["telegram"]["last_delivery"] == terminal
            if expected["configured"]:
                assert backend_state["telegram"]["last_delivery_timestamp"]
            # Exercise saving the additive channel preference, then restore it.
            toggle = card.locator("[data-telegram-enabled]")
            toggle.set_checked(not expected["enabled"])
            page.wait_for_timeout(500)
            with urlopen(base + "/api/notifications/settings") as response:
                assert json.load(response)["telegram"]["enabled"] is not expected["enabled"]
            toggle.set_checked(expected["enabled"])
            page.wait_for_timeout(500)
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            assert not token or token not in page.content()
            assert "TELEGRAM_BOT_TOKEN" not in page.content()
            chat = os.getenv("TELEGRAM_CHAT_ID", "")
            assert not chat or chat not in card.inner_text()
            page.screenshot(path=str(output / "settings_telegram.png"), full_page=True)
            assert not errors, f"Browser errors: {errors}"
            browser.close()
        report = {"playwright": "PASS", "credentials": "Configured" if configured else "Not configured",
                  "real_telegram_e2e": "PASS" if expected["configured"] else "PENDING — credentials not configured",
                  "delivery": terminal, "console_errors": errors}
        (output / "ui_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True))
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


if __name__ == "__main__":
    main()
