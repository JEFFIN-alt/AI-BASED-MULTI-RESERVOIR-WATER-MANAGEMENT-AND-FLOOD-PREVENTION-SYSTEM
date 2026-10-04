# Gmail alerts and Google OAuth

AquaFlow uses a server-side Google OAuth flow. The browser receives only the
authorization URL and connection status; the client secret and OAuth tokens
stay in the backend process. Gmail permissions are limited to `gmail.send`
and the read-only `userinfo.email` identity scope. AquaFlow never asks for a
Google password. Tokens are held in backend memory and cleared on restart or
disconnect; a restart requires connecting again.

## Google Cloud setup

1. Create or select a project in [Google Cloud Console](https://console.cloud.google.com/).
2. In **APIs & Services → Library**, enable **Gmail API**.
3. In **Google Auth Platform** (or **APIs & Services → OAuth consent screen**), configure the consent screen. Add the Gmail send and user info email scopes if the console requires manual scope selection. Add your Google account as a test user while the app is in Testing.
4. Under **Clients** (or **Credentials → Create credentials → OAuth client ID**), create a **Web application** OAuth client.
5. Add this exact authorized redirect URI:
   `http://127.0.0.1:8000/api/notifications/gmail/callback`
6. Copy `.env.example` to `.env`. Set `AQUAFLOW_GOOGLE_CLIENT_ID` and `AQUAFLOW_GOOGLE_CLIENT_SECRET` to the values from the client you created. Keep `AQUAFLOW_GOOGLE_REDIRECT_URI` set to the exact URI above. The `.env` file is ignored by Git; never commit credentials.
7. Restart AquaFlow so it reads the environment variables. The included `.env.example` is a template and is not loaded automatically by the launcher.
8. Open AquaFlow at `http://127.0.0.1:8000/` and open **Settings → Notifications**.
9. Click **Connect Gmail**.
10. Select the sending Google account on Google's authorization page and grant the requested Gmail send and email identity permissions.
11. Google redirects to AquaFlow's callback, which validates one-time OAuth state, exchanges the code on the server, looks up the account email, then returns to Settings.
12. Confirm that the connection shows **CONNECTED** and the authorized email is displayed.
13. Configure and save the alert recipient, then click **Send Test Email**. The test uses the connected OAuth Gmail account and reports **SENT** only after Gmail API accepts the message.

The Google Cloud project, Gmail API enablement, consent screen, test-user list,
client creation, and Google account consent must be completed by an administrator
in Google's interface. AquaFlow cannot automate those Google-side steps. OAuth
is unavailable until the three `AQUAFLOW_GOOGLE_*` environment variables are
set. The local redirect URI must match exactly, including host and port.

## Local environment

PowerShell:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2], 'Process')
  }
}
python app.py --no-browser
```

Bash:

```bash
set -a
. ./.env
set +a
python app.py --no-browser
```

The legacy SMTP environment options remain available for downstream incident
delivery. **Send Test Email** specifically requires an active Google OAuth
connection and will not claim success through SMTP. No OAuth access or refresh
tokens are returned by AquaFlow APIs or written to frontend storage.
