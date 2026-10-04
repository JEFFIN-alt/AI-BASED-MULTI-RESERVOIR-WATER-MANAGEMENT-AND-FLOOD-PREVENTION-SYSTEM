"""Notification providers must be explicitly mocked in automated tests."""
import pytest


@pytest.fixture(autouse=True)
def block_unmocked_telegram_requests(monkeypatch):
    from src.notifications import telegram_alerts

    def blocked(*args, **kwargs):
        raise AssertionError("Telegram requests must be explicitly mocked in tests")

    monkeypatch.setattr(telegram_alerts, "_request", blocked)


@pytest.fixture(autouse=True)
def block_unmocked_discord_requests(monkeypatch):
    import httpx

    original = httpx.AsyncHTTPTransport.handle_async_request

    async def guarded(self, request):
        if (request.url.host in {"discord.com", "discordapp.com"} or
                request.url.host.endswith((".discord.com", ".discordapp.com"))):
            raise AssertionError("Discord requests must use a mocked transport in tests")
        return await original(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded)
    # Explicitly empty environment overrides any real local .env credentials.
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
