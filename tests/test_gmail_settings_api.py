from fastapi.testclient import TestClient
from urllib.parse import parse_qs, urlparse

from src.dashboard.api import notifications
from src.dashboard.api.app import app
from src.dashboard.api.gmail_auth import gmail_oauth
from src.dashboard.api.state_manager import sim_state


def test_settings_oauth_configuration_and_test_email_are_safe(monkeypatch):
    for key in ("AQUAFLOW_GOOGLE_CLIENT_ID", "AQUAFLOW_GOOGLE_CLIENT_SECRET", "AQUAFLOW_GOOGLE_REDIRECT_URI"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(notifications, "_config", lambda *args: None)
    gmail_oauth.disconnect()
    old_preferences = dict(sim_state.notification_manager.preferences)
    old_gates = dict(sim_state.manual_gates)
    old_step = sim_state.sim_step_index
    try:
        with TestClient(app) as client:
            status = client.get("/api/notifications/settings").json()
            assert status["gmail"]["connected"] is False
            assert status["gmail"]["oauth_configured"] is False
            assert "Google OAuth is not configured" in status["configuration_error"]
            assert client.post("/api/notifications/gmail/connect").status_code == 503
            assert client.put("/api/notifications/settings", json={"recipient": "invalid"}).status_code == 422
            saved = client.put("/api/notifications/settings", json={
                "recipient": "operator@example.com", "high_enabled": True, "critical_enabled": True,
            })
            assert saved.status_code == 200
            assert saved.json()["recipient"] == "operator@example.com"
            assert client.post("/api/notifications/test").status_code == 409
            callback = client.get("/api/notifications/gmail/callback?code=abc&state=bad", follow_redirects=False)
            assert callback.status_code == 307
            assert "gmail_oauth=error" in callback.headers["location"]
            assert sim_state.manual_gates == old_gates
            assert sim_state.sim_step_index == old_step
    finally:
        sim_state.notification_manager.update_preferences(**old_preferences)


def test_oauth_authorization_callback_and_refresh(monkeypatch):
    from src.dashboard.api import gmail_auth
    from src.dashboard.api.gmail_auth import GmailOAuthManager

    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/notifications/gmail/callback")
    manager = GmailOAuthManager()
    parsed = parse_qs(urlparse(manager.authorization_url()).query)
    assert parsed["client_id"] == ["client-id"]
    assert parsed["redirect_uri"] == ["http://127.0.0.1:8000/api/notifications/gmail/callback"]
    assert "gmail.send" in parsed["scope"][0]
    assert "gmail.readonly" not in parsed["scope"][0]

    calls = []
    def post_form(url, fields):
        calls.append((url, fields))
        if fields.get("grant_type") == "authorization_code":
            return {"access_token": "access-1", "refresh_token": "refresh-1", "expires_in": 3600}
        return {"access_token": "access-2", "expires_in": 3600}
    monkeypatch.setattr(gmail_auth, "_post_form", post_form)
    monkeypatch.setattr(gmail_auth, "_get_json", lambda url, token: {"email": "operator@gmail.com"})
    state = parsed["state"][0]
    manager.complete("authorization-code", state)
    assert manager.status()["connected"] is True
    assert manager.status()["email"] == "operator@gmail.com"
    assert calls[0][1]["client_secret"] == "client-secret"
    manager._tokens["expires_at"] = 0
    assert manager.access_token() == "access-2"
    assert calls[1][1]["grant_type"] == "refresh_token"
    manager.disconnect()
    assert manager.status()["connected"] is False


def test_oauth_state_is_one_time_and_callback_route_exists(monkeypatch):
    from src.dashboard.api import gmail_auth
    from src.dashboard.api.gmail_auth import GmailOAuthManager
    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/notifications/gmail/callback")
    manager = GmailOAuthManager()
    state = parse_qs(urlparse(manager.authorization_url()).query)["state"][0]
    try:
        manager.complete("code", "incorrect")
        assert False, "invalid state must be rejected"
    except ValueError:
        pass
    monkeypatch.setattr(gmail_auth, "_post_form", lambda url, fields: {
        "access_token": "access", "refresh_token": "refresh", "expires_in": 3600
    })
    monkeypatch.setattr(gmail_auth, "_get_json", lambda url, token: {"email": "operator@gmail.com"})
    manager.complete("code", state)
    try:
        manager.complete("code", state)
        assert False, "state must be consumed after successful validation"
    except ValueError:
        pass


def test_failed_refresh_returns_authentication_error_without_secrets(monkeypatch):
    from src.dashboard.api import gmail_auth
    from src.dashboard.api.gmail_auth import GmailOAuthManager
    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AQUAFLOW_GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/notifications/gmail/callback")
    manager = GmailOAuthManager()
    manager._tokens = {"access_token": "stale", "refresh_token": "private-refresh-token", "expires_at": 0}
    manager.email = "operator@gmail.com"
    monkeypatch.setattr(gmail_auth, "_post_form", lambda *args: (_ for _ in ()).throw(RuntimeError("secret token")))
    try:
        manager.access_token()
        assert False, "refresh failure must be surfaced"
    except RuntimeError as exc:
        assert "authorization" in str(exc).lower()
        assert "secret" not in str(exc)
    assert manager.status()["connected"] is False
    assert manager.status()["auth_error"] is True
