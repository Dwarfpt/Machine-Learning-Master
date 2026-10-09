"""RPA integration: pushes events to n8n webhooks. Without n8n configured, events go to data/outbox.jsonl
so the whole pipeline still runs (and can be replayed into n8n later)."""
from __future__ import annotations

import hashlib
import hmac
import json

import requests

from . import config, db


def approval_token(action_id: int) -> str:
    return hmac.new(config.APPROVAL_SECRET.encode(), str(action_id).encode(), hashlib.sha256).hexdigest()[:32]


def verify_token(action_id: int, token: str) -> bool:
    return hmac.compare_digest(approval_token(action_id), token or "")


def _post(url: str, event: str, payload: dict) -> dict:
    body = {"event": event, "sent_at": db.now(), **payload}
    if url:
        try:
            r = requests.post(url, json=body, timeout=10)
            return {"channel": "n8n", "url": url, "status_code": r.status_code, "ok": r.ok}
        except requests.RequestException as e:
            body["delivery_error"] = str(e)
    with open(config.OUTBOX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False, default=str) + "\n")
    return {"channel": "outbox", "path": str(config.OUTBOX_PATH), "ok": True,
            "note": body.get("delivery_error", "N8N_WEBHOOK_URL not set")}


def notify_action_created(action_id: int, extra: dict | None = None) -> dict:
    a = db.get_action(action_id)
    token = approval_token(action_id)
    base = f"{config.API_PUBLIC_URL}/actions/{action_id}/decision?token={token}"
    payload = {"action_id": action_id, "token": token, "action": a,
               "requires_approval": bool(a["requires_approval"]),
               "approve_url": f"{base}&decision=approved", "reject_url": f"{base}&decision=rejected",
               "execute_url": f"{config.API_PUBLIC_URL}/actions/{action_id}/execute?token={token}",
               **(extra or {})}
    return _post(config.N8N_WEBHOOK_URL, "action_created", payload)


def notify_decision(action_id: int) -> dict:
    a = db.get_action(action_id)
    token = approval_token(action_id)
    return _post(config.N8N_DECISION_WEBHOOK_URL or config.N8N_WEBHOOK_URL, "action_decided",
                 {"action_id": action_id, "token": token, "action": a, "execute_url": f"{config.API_PUBLIC_URL}/actions/{action_id}/execute?token={token}"})
