import os
import logging
from consistencyguard.models import ConsistencyViolation

logger = logging.getLogger(__name__)


def _serialize(v: ConsistencyViolation) -> dict:
    d = v.model_dump()
    d["severity"] = d["severity"].value if hasattr(d["severity"], "value") else d["severity"]
    d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else d["timestamp"]
    d["event"] = "consistency_violation"
    return d


def fire_webhook(v: ConsistencyViolation, url: str = None) -> None:
    """POST violation JSON to url. Best-effort — never raises to caller."""
    target = url or os.getenv("WEBHOOK_URL")
    if not target:
        return
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            client.post(target, json=_serialize(v))
    except Exception as exc:
        logger.warning("Webhook delivery failed: %s", exc)


async def afire_webhook(v: ConsistencyViolation, url: str = None) -> None:
    """Async version of fire_webhook."""
    target = url or os.getenv("WEBHOOK_URL")
    if not target:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(target, json=_serialize(v))
    except Exception as exc:
        logger.warning("Webhook delivery failed: %s", exc)
