"""
Core proxy. guarded_call / aguarded_call are the drop-in replacement
for direct LLM API calls. Both return (response_text, violations).
"""

import os
import asyncio
from datetime import datetime
from typing import Union, Optional

from consistencyguard.models import LLMCall, ConsistencyViolation
from consistencyguard.embedder import embed
from consistencyguard.store import init_db, save_call, save_violation
from consistencyguard.detector import check_consistency
from consistencyguard.providers import get_provider, AnthropicProvider, OpenAIProvider
from consistencyguard.webhooks import fire_webhook, afire_webhook

_db_initialized = False


def _ensure_db() -> None:
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True


ProviderArg = Optional[Union[str, AnthropicProvider, OpenAIProvider]]


def guarded_call(
    prompt: str,
    model: str = None,
    max_tokens: int = 1024,
    agent_id: str = "default",
    api_key: str = None,
    provider: ProviderArg = None,
) -> tuple[str, list[ConsistencyViolation]]:
    """
    Drop-in replacement for a direct LLM call.
    Returns (response_text, list_of_violations).

    Args:
        provider: "anthropic" | "openai" | an already-instantiated provider
                  object. Defaults to PROVIDER env var (default: anthropic).
    """
    _ensure_db()

    if model is None:
        model = os.getenv("MODEL", "claude-haiku-4-5-20251001")

    llm = provider if hasattr(provider, "complete") else get_provider(provider, api_key)

    prompt_embedding = embed(prompt)
    call = LLMCall(
        prompt=prompt,
        response="",
        model=model,
        agent_id=agent_id,
        timestamp=datetime.utcnow(),
        prompt_embedding=prompt_embedding,
    )

    response_text = llm.complete(prompt, model, max_tokens)
    call.response = response_text

    call_id = save_call(call)
    call.id = call_id

    violations = check_consistency(call)
    for v in violations:
        v.call_id_new = call_id
        save_violation(v)
        fire_webhook(v)

    return response_text, violations


async def aguarded_call(
    prompt: str,
    model: str = None,
    max_tokens: int = 1024,
    agent_id: str = "default",
    api_key: str = None,
    provider: ProviderArg = None,
) -> tuple[str, list[ConsistencyViolation]]:
    """Async version of guarded_call."""
    _ensure_db()

    if model is None:
        model = os.getenv("MODEL", "claude-haiku-4-5-20251001")

    llm = provider if hasattr(provider, "acomplete") else get_provider(provider, api_key)

    # sentence-transformers is CPU-bound / not async-native — run in thread pool
    loop = asyncio.get_event_loop()
    prompt_embedding = await loop.run_in_executor(None, embed, prompt)

    call = LLMCall(
        prompt=prompt,
        response="",
        model=model,
        agent_id=agent_id,
        timestamp=datetime.utcnow(),
        prompt_embedding=prompt_embedding,
    )

    response_text = await llm.acomplete(prompt, model, max_tokens)
    call.response = response_text

    call_id = save_call(call)
    call.id = call_id

    violations = await loop.run_in_executor(None, check_consistency, call)
    for v in violations:
        v.call_id_new = call_id
        save_violation(v)
        await afire_webhook(v)

    return response_text, violations
