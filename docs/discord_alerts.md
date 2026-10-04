# Discord notifications

Set `DISCORD_WEBHOOK_URL` in the backend process environment to a Discord channel
webhook, or save it in the project-root `.env` file (ignored by Git). The Discord
adapter reads this local fallback once on backend startup. A process environment
value takes precedence; an explicitly empty value disables the local fallback.
`.env.example` contains a blank placeholder. Never put the webhook in frontend configuration.
Settings → Notifications displays only a fully masked webhook. Request logs and
delivery errors redact credentials.

Discord appears next to Gmail. Its enable toggle affects automatic Discord
notifications; **Send Test Discord Alert** remains available independently.
Gmail retains its existing one-email-per-contiguous-incident policy, OAuth/SMTP
delivery and test-email behaviour. Telegram remains unchanged.

## Incident policy

- Initial downstream WARNING/CRITICAL alert is queued immediately, independently
  of Gmail delivery. Repeated observations of the same incident are deduplicated.
- Unacknowledged incidents repeat every five minutes, including while paused.
  Escalation and test requests have a 30-second cooldown. Recovery also observes
  the incident cooldown. Provider rate-limit waits take precedence.
- Acknowledge in Settings → Notifications. The backend endpoint is
  `POST /api/notifications/incidents/{incident_id}/acknowledge`. Acknowledgement
  is idempotent and stops queued/future reminders; an already in-flight request
  may complete. It does not change reservoir gates or resolve the risk.
- Backend NORMAL telemetry closes the incident and sends one green recovery
  message if an alert was delivered, including after acknowledgement. UNKNOWN
  telemetry does not invent a recovery. A new incident gets a new identifier.
- Incident state and enable preferences are in memory, like existing Gmail
  state; they reset on backend restart. Run the existing single-process backend.

Each message has one bounded embed with available downstream flow/limit,
proposed MPC downstream prediction/capacity, applied gates and simulation day.
Recent real downstream samples are plotted in `chart.png`, uploaded as multipart
`files[0]`, and displayed through `attachment://chart.png`. A test before any
samples exist shows an explicitly empty chart. There are no Discord chat buttons,
upstream-driver lines or fabricated D forecasts.

Delivery uses an independent async httpx worker with a 32-job queue, a three-second
request timeout and a two-second connection timeout. HTTP 429 respects
`retry_after` (or `Retry-After`) and permits at most one retry per delivery. An
exhausted bucket also delays later requests. Invalid/unavailable webhooks stop
network retries until configuration changes. Queue overflow, HTTP errors and
timeouts cannot block the simulation or prevent Gmail delivery.

## Verification

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_discord_alerts.py tests/test_downstream_notifications.py tests/test_gmail_settings_api.py tests/test_telegram_alerts.py
python -B -m pytest -q -p no:cacheprovider tests/test_discord_notifications_e2e.py
```

The browser e2e uses a real local FastAPI server and Chromium with mocked Gmail
and Discord providers. It requires an installed Playwright Chromium browser.
A PASS for this check does not claim delivery to a real Discord channel.
