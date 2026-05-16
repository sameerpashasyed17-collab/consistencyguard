"""
Shared pytest fixtures for ConsistencyGuard tests.

Key guarantee: every test gets an isolated SQLite DB via tmp_path so
no test can pollute another test's data or the production database.
"""

import os
import pytest
from datetime import datetime
from consistencyguard.models import LLMCall, ConsistencyViolation, ViolationSeverity
from consistencyguard import store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Automatically give every test its own SQLite file.
    Without this, all tests share 'consistencyguard.db' in CWD —
    causing cross-test pollution and accidentally overwriting production data.
    """
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    store.init_db()
    yield db_path


@pytest.fixture
def sample_call() -> LLMCall:
    from consistencyguard.embedder import embed
    return LLMCall(
        prompt="What is the refund policy?",
        response="Full refund within 30 days.",
        model="test-model",
        agent_id="test-agent",
        timestamp=datetime.utcnow(),
        prompt_embedding=embed("What is the refund policy?"),
    )


@pytest.fixture
def sample_violation(sample_call) -> ConsistencyViolation:
    call_id = store.save_call(sample_call)
    return ConsistencyViolation(
        call_id_new=call_id,
        call_id_ref=call_id - 1,
        prompt_similarity=0.97,
        response_divergence=0.45,
        severity=ViolationSeverity.CRITICAL,
        new_prompt=sample_call.prompt,
        new_response=sample_call.response,
        ref_response="No refunds under any circumstances.",
        explanation="Test violation",
        agent_id="test-agent",
        timestamp=datetime.utcnow(),
    )
