"""
Web Push notifications, so signals can alert you even when the dashboard
tab/app isn't open on screen (as long as your phone has it installed and
notifications are allowed).

Uses VAPID (no third-party push service account needed, works with Safari
on iOS 16.4+ for home-screen-installed web apps).

Generate your VAPID keypair once:
    pip install py-vapid
    vapid --gen
This creates private_key.pem / public_key.pem — read them into the env
vars below (see README).
"""
import os
import json
from pywebpush import webpush, WebPushException

VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "")
VAPID_CLAIMS_EMAIL = os.environ.get("VAPID_CLAIMS_EMAIL", "mailto:you@example.com")

# In-memory subscription store. For anything beyond solo/local use, swap
# this for a small database table (subscriptions rarely change).
SUBSCRIPTIONS: list[dict] = []


def add_subscription(subscription: dict):
    if subscription not in SUBSCRIPTIONS:
        SUBSCRIPTIONS.append(subscription)


def remove_subscription(endpoint: str):
    global SUBSCRIPTIONS
    SUBSCRIPTIONS = [s for s in SUBSCRIPTIONS if s.get("endpoint") != endpoint]


def notify_all(title: str, body: str, url: str = "/"):
    """Send a push notification to every subscribed device."""
    if not VAPID_PRIVATE_KEY:
        return  # push not configured — silently skip rather than crash the scan loop

    payload = json.dumps({"title": title, "body": body, "url": url})
    dead = []
    for sub in SUBSCRIPTIONS:
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            )
        except WebPushException:
            dead.append(sub)  # subscription expired or device unsubscribed
    for sub in dead:
        remove_subscription(sub.get("endpoint"))
