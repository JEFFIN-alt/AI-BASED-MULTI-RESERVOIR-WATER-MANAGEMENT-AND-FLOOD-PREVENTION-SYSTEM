# Telegram notifications

Set server environment variables `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and
`TELEGRAM_ALLOWED_USER_IDS` (comma-separated numeric user IDs). All three are
required. Keep the token in the server environment; never put it in frontend
configuration. Settings → Notifications shows only a masked chat ID.

Run one AquaFlow application process per bot. The managed Telegram worker uses
authenticated `getUpdates` polling; a bot with an existing webhook or another
polling consumer must have that integration removed before using AquaFlow.
Polling failures appear in the backend `telegram.callback_status` field.

Telegram consumes the existing Gmail manager's WARNING/CRITICAL incidents,
independently of Gmail delivery and preferences. Each incident is attempted
once, with no automatic retry or escalation resend. Recovery closes the
incident; a subsequent warning creates a new incident. ACKNOWLEDGE changes
only notification acknowledgement metadata. Unknown users, other chats,
closed incidents, and unsuccessful deliveries cannot be acknowledged.

The worker uses a bounded queue and eight-second HTTP timeouts. Shutdown stops
polling, waits for the active request, and marks queued deliveries as failed.
Telegram state is in memory, matching the existing incident manager; restarting
the application clears delivery and acknowledgement history.

Charts require three distinct backend state observations with numeric
downstream flow and safe limit. Unavailable metrics are omitted. Message and
chart generation, sending, and callback polling have no simulation control
authority. Bot API methods and provider confirmation follow the
[official Telegram Bot API](https://core.telegram.org/bots/api).

Verification:

```text
py -m pytest tests/test_telegram_alerts.py -q
py -m pytest tests/test_downstream_notifications.py tests/test_gmail_settings_api.py -q
py scripts/verify_telegram_ui.py
py -m pytest -q
```

Automated tests block unmocked Telegram requests. The Playwright script uses
the real local backend, checks the UI, and saves evidence under
`results/telegram_verification/`. When configured, its test button sends
exactly one real test alert. Otherwise real E2E remains pending.
