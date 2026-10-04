"""Small in-memory Google OAuth manager for Gmail API delivery."""
from __future__ import annotations

import base64
from email.message import EmailMessage
import json
import os
import secrets
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class GmailOAuthManager:
    SCOPES = "https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/userinfo.email"

    def __init__(self):
        self._states = {}
        self._tokens = None
        self.email = None
        self._token_error = False

    def config(self):
        client_id = os.getenv("AQUAFLOW_GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.getenv("AQUAFLOW_GOOGLE_CLIENT_SECRET", "")
        redirect_uri = os.getenv("AQUAFLOW_GOOGLE_REDIRECT_URI", "").strip()
        if not all((client_id, client_secret, redirect_uri)):
            return None
        return client_id, client_secret, redirect_uri

    def status(self):
        return {"oauth_configured": bool(self.config()), "connected": bool(self._tokens and self.email),
                "email": self.email if self._tokens else None, "auth_error": self._token_error}

    def authorization_url(self):
        config = self.config()
        if not config:
            return None
        client_id, _, redirect_uri = config
        state = secrets.token_urlsafe(32)
        self._states[state] = time.time() + 600
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "client_id": client_id, "redirect_uri": redirect_uri,
            "response_type": "code", "scope": self.SCOPES,
            "access_type": "offline", "prompt": "consent",
            "include_granted_scopes": "true", "state": state,
        })

    def complete(self, code, state):
        config = self.config()
        if not config:
            raise RuntimeError("Google OAuth is not configured")
        expires = self._states.pop(state, None) if state else None
        if not expires or expires < time.time():
            raise ValueError("OAuth state verification failed")
        client_id, client_secret, redirect_uri = config
        token = _post_form("https://oauth2.googleapis.com/token", {
            "code": code, "client_id": client_id, "client_secret": client_secret,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code",
        })
        info = _get_json("https://www.googleapis.com/oauth2/v2/userinfo", token["access_token"])
        email = info.get("email")
        if not email:
            raise RuntimeError("Google did not return an account email")
        self._tokens = {"access_token": token["access_token"],
                        "refresh_token": token.get("refresh_token"),
                        "expires_at": time.time() + int(token.get("expires_in", 3600))}
        self.email = email
        self._token_error = False
        self._states.clear()

    def access_token(self):
        if not self._tokens:
            return None
        if self._tokens["expires_at"] - 60 > time.time():
            return self._tokens["access_token"]
        refresh = self._tokens.get("refresh_token")
        config = self.config()
        if not refresh or not config:
            self._tokens = None
            self.email = None
            self._token_error = True
            raise RuntimeError("Google authorization needs to be completed again")
        client_id, client_secret, _ = config
        try:
            token = _post_form("https://oauth2.googleapis.com/token", {
                "client_id": client_id, "client_secret": client_secret,
                "refresh_token": refresh, "grant_type": "refresh_token",
            })
        except Exception:
            self._tokens = None
            self.email = None
            self._token_error = True
            raise RuntimeError("Google authorization needs to be completed again") from None
        self._tokens["access_token"] = token["access_token"]
        self._tokens["expires_at"] = time.time() + int(token.get("expires_in", 3600))
        self._token_error = False
        return self._tokens["access_token"]

    def send(self, subject, body, recipient):
        token = self.access_token()
        if not token:
            raise RuntimeError("Gmail OAuth is not connected")
        msg = EmailMessage()
        msg["To"] = recipient
        msg["From"] = self.email
        msg["Subject"] = subject
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        _post_json("https://gmail.googleapis.com/gmail/v1/users/me/messages/send", token, {"raw": raw})

    def disconnect(self):
        token = (self._tokens or {}).get("refresh_token") or (self._tokens or {}).get("access_token")
        self._tokens = None
        self.email = None
        self._states.clear()
        self._token_error = False
        if token:
            try:
                _post_form("https://oauth2.googleapis.com/revoke", {"token": token})
            except Exception:
                # Local disconnection always completes; revoke failures never
                # restore tokens or leak response details to the API/UI.
                pass


def _post_form(url, fields):
    body = urlencode(fields).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _get_json(url, token):
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _post_json(url, token, payload):
    request = Request(url, data=json.dumps(payload).encode(), method="POST",
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read())


gmail_oauth = GmailOAuthManager()
