"""
Tests for provider abstraction.
All tests are mocked — no real API calls, no API keys required.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from consistencyguard.providers import (
    AnthropicProvider,
    OpenAIProvider,
    get_provider,
)

# Patch targets — match the aliases used in providers.py
_ANTHROPIC = "consistencyguard.providers._anthropic_mod"
_OPENAI = "consistencyguard.providers._openai_mod"


# ── helpers ────────────────────────────────────────────────────────────────────

def _anthropic_response(text: str = "Test response") -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


def _openai_response(text: str = "Test response") -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=text))]
    return mock


# ── factory tests ──────────────────────────────────────────────────────────────

def test_get_provider_returns_anthropic_by_default(monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    with patch(_ANTHROPIC) as mock_mod:
        mock_mod.Anthropic.return_value = MagicMock()
        mock_mod.AsyncAnthropic.return_value = MagicMock()
        p = get_provider()
    assert isinstance(p, AnthropicProvider)


def test_get_provider_explicit_anthropic(monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    with patch(_ANTHROPIC) as mock_mod:
        mock_mod.Anthropic.return_value = MagicMock()
        mock_mod.AsyncAnthropic.return_value = MagicMock()
        p = get_provider("anthropic")
    assert isinstance(p, AnthropicProvider)


def test_get_provider_returns_openai_by_env(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    with patch(_OPENAI) as mock_mod, patch("consistencyguard.providers._OPENAI_AVAILABLE", True):
        mock_mod.OpenAI.return_value = MagicMock()
        mock_mod.AsyncOpenAI.return_value = MagicMock()
        p = get_provider()
    assert isinstance(p, OpenAIProvider)


def test_get_provider_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("groq")


# ── Anthropic sync ─────────────────────────────────────────────────────────────

def test_anthropic_complete_returns_text():
    resp = _anthropic_response("Hello from Anthropic")
    with patch(_ANTHROPIC) as mock_mod:
        mock_mod.Anthropic.return_value.messages.create.return_value = resp
        mock_mod.AsyncAnthropic.return_value = MagicMock()
        p = AnthropicProvider(api_key="fake-key")
        result = p.complete("hello", "claude-haiku-4-5-20251001", 100)
    assert result == "Hello from Anthropic"


def test_anthropic_complete_passes_correct_args():
    resp = _anthropic_response()
    with patch(_ANTHROPIC) as mock_mod:
        mock_client = mock_mod.Anthropic.return_value
        mock_client.messages.create.return_value = resp
        mock_mod.AsyncAnthropic.return_value = MagicMock()
        p = AnthropicProvider(api_key="fake-key")
        p.complete("my prompt", "claude-haiku-4-5-20251001", 512)
    mock_client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "my prompt"}],
    )


# ── OpenAI sync ────────────────────────────────────────────────────────────────

def test_openai_complete_returns_text():
    resp = _openai_response("Hello from OpenAI")
    with patch(_OPENAI) as mock_mod, patch("consistencyguard.providers._OPENAI_AVAILABLE", True):
        mock_mod.OpenAI.return_value.chat.completions.create.return_value = resp
        mock_mod.AsyncOpenAI.return_value = MagicMock()
        p = OpenAIProvider(api_key="fake-key")
        result = p.complete("hello", "gpt-4o-mini", 100)
    assert result == "Hello from OpenAI"


def test_openai_complete_passes_correct_args():
    resp = _openai_response()
    with patch(_OPENAI) as mock_mod, patch("consistencyguard.providers._OPENAI_AVAILABLE", True):
        mock_client = mock_mod.OpenAI.return_value
        mock_client.chat.completions.create.return_value = resp
        mock_mod.AsyncOpenAI.return_value = MagicMock()
        p = OpenAIProvider(api_key="fake-key")
        p.complete("my prompt", "gpt-4o-mini", 256)
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[{"role": "user", "content": "my prompt"}],
    )


# ── Anthropic async ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_acomplete_returns_text():
    resp = _anthropic_response("Async Anthropic response")
    with patch(_ANTHROPIC) as mock_mod:
        mock_mod.Anthropic.return_value = MagicMock()
        mock_async_client = mock_mod.AsyncAnthropic.return_value
        mock_async_client.messages.create = AsyncMock(return_value=resp)
        p = AnthropicProvider(api_key="fake-key")
        result = await p.acomplete("hello", "claude-haiku-4-5-20251001", 100)
    assert result == "Async Anthropic response"


# ── OpenAI async ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_acomplete_returns_text():
    resp = _openai_response("Async OpenAI response")
    with patch(_OPENAI) as mock_mod, patch("consistencyguard.providers._OPENAI_AVAILABLE", True):
        mock_mod.OpenAI.return_value = MagicMock()
        mock_async_client = mock_mod.AsyncOpenAI.return_value
        mock_async_client.chat.completions.create = AsyncMock(return_value=resp)
        p = OpenAIProvider(api_key="fake-key")
        result = await p.acomplete("hello", "gpt-4o-mini", 100)
    assert result == "Async OpenAI response"
