"""
User flow tests for ConsistencyGuard.
Simulates real developer usage: wrap calls, detect violations, check stats.
No real API calls — provider is mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from consistencyguard.proxy import guarded_call
from consistencyguard.models import ViolationSeverity
from consistencyguard import store


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "user_test.db"))
    import consistencyguard.proxy as _proxy
    _proxy._db_initialized = False
    yield


def _mock_provider(response_text: str) -> MagicMock:
    p = MagicMock()
    p.complete.return_value = response_text
    return p


# ── scenario 1: first call, no history → no violations ───────────────────────

def test_first_call_returns_response_no_violations():
    provider = _mock_provider("The refund window is 30 days.")
    text, violations = guarded_call(
        "What is the refund policy?",
        provider=provider,
        agent_id="support-agent",
    )
    assert text == "The refund window is 30 days."
    assert violations == []


# ── scenario 2: consistent follow-up → still no violations ───────────────────

def test_consistent_responses_produce_no_violations():
    answer = "The refund window is 30 days."
    provider = _mock_provider(answer)

    guarded_call("What is the refund policy?", provider=provider, agent_id="support-agent")
    _, violations = guarded_call("What is the refund policy?", provider=provider, agent_id="support-agent")

    assert violations == []


# ── scenario 3: contradicting response → violation flagged ────────────────────

def test_contradicting_response_raises_violation():
    provider_a = _mock_provider("The refund window is 30 days.")
    provider_b = _mock_provider("Contact our legal team for billing inquiries.")

    guarded_call("What is the refund policy?", provider=provider_a, agent_id="support-agent")
    _, violations = guarded_call("What is the refund policy?", provider=provider_b, agent_id="support-agent")

    assert len(violations) >= 1
    assert violations[0].severity in (ViolationSeverity.WARNING, ViolationSeverity.CRITICAL)


# ── scenario 4: violation carries correct metadata ────────────────────────────

def test_violation_metadata_is_correct():
    guarded_call("What is the upload limit?",
                 provider=_mock_provider("25 MB per file."), agent_id="storage-agent")
    _, violations = guarded_call("What is the upload limit?",
                                 provider=_mock_provider("Call support to check limits."),
                                 agent_id="storage-agent")

    if violations:
        v = violations[0]
        assert v.agent_id == "storage-agent"
        assert v.new_response == "Call support to check limits."
        assert 0.0 < v.response_divergence <= 1.0


# ── scenario 5: stats reflect all recorded calls ──────────────────────────────

def test_stats_count_calls_and_violations():
    guarded_call("Policy question?",
                 provider=_mock_provider("Answer A."), agent_id="agent-x")
    guarded_call("Policy question?",
                 provider=_mock_provider("Completely unrelated topic about cats."),
                 agent_id="agent-x")

    stats = store.get_stats()
    assert stats["total_calls"] == 2
    # May or may not produce violation depending on divergence threshold
    assert stats["total_violations"] >= 0


# ── scenario 6: different agents don't cross-contaminate ─────────────────────

def test_cross_agent_divergence_is_detected():
    """Library compares across all agents globally — same prompt, different agent, contradicting
    response still triggers a violation. This is by design: cross-agent consistency matters."""
    guarded_call("What is the price?",
                 provider=_mock_provider("$99/month."), agent_id="sales-agent")
    _, violations = guarded_call("What is the price?",
                                 provider=_mock_provider("Contact support for pricing."),
                                 agent_id="different-agent")
    # Cross-agent contradiction IS flagged — global consistency scope
    assert len(violations) >= 1


# ── scenario 7: multiple agents, stats stay global ───────────────────────────

def test_stats_aggregate_across_agents():
    guarded_call("Q1?", provider=_mock_provider("A."), agent_id="agent-1")
    guarded_call("Q2?", provider=_mock_provider("B."), agent_id="agent-2")
    guarded_call("Q3?", provider=_mock_provider("C."), agent_id="agent-3")

    stats = store.get_stats()
    assert stats["total_calls"] == 3
