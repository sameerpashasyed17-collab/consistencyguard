"""
Tests for the SQLite store.
Each test uses a fresh temp DB via the tmp_path fixture.
"""

import os
import sqlite3
import pytest
from datetime import datetime
from consistencyguard.models import LLMCall, ConsistencyViolation, ViolationSeverity
from consistencyguard import store


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    yield db_path


def _make_call(**kwargs) -> LLMCall:
    defaults = dict(
        prompt="Test prompt",
        response="Test response",
        model="test-model",
        agent_id="test-agent",
        timestamp=datetime.utcnow(),
        prompt_embedding=[0.1, 0.2, 0.3],
    )
    defaults.update(kwargs)
    return LLMCall(**defaults)


def test_init_db_creates_llm_calls_table(use_temp_db):
    store.init_db()
    with sqlite3.connect(use_temp_db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    assert ("llm_calls",) in tables


def test_init_db_creates_violations_table(use_temp_db):
    store.init_db()
    with sqlite3.connect(use_temp_db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    assert ("violations",) in tables


def test_save_call_returns_integer_id(use_temp_db):
    store.init_db()
    call = _make_call()
    result = store.save_call(call)
    assert isinstance(result, int)
    assert result >= 1


def test_get_all_calls_returns_saved_call(use_temp_db):
    store.init_db()
    call = _make_call(prompt="Hello world", response="Hi there")
    call_id = store.save_call(call)

    calls = store.get_all_calls()
    assert len(calls) == 1
    assert calls[0].id == call_id
    assert calls[0].prompt == "Hello world"
    assert calls[0].response == "Hi there"


def test_save_violation_does_not_raise(use_temp_db):
    store.init_db()
    call = _make_call()
    call_id = store.save_call(call)

    v = ConsistencyViolation(
        call_id_new=call_id,
        call_id_ref=call_id - 1,
        prompt_similarity=0.95,
        response_divergence=0.45,
        severity=ViolationSeverity.CRITICAL,
        new_prompt="Test prompt",
        new_response="New response",
        ref_response="Old response",
        explanation="Test explanation",
        agent_id="test-agent",
        timestamp=datetime.utcnow(),
    )
    store.save_violation(v)  # should not raise


def test_get_stats_returns_correct_counts(use_temp_db):
    store.init_db()

    for _ in range(3):
        call = _make_call()
        call_id = store.save_call(call)

    v = ConsistencyViolation(
        call_id_new=call_id,
        call_id_ref=call_id - 1,
        prompt_similarity=0.95,
        response_divergence=0.45,
        severity=ViolationSeverity.CRITICAL,
        new_prompt="Test prompt",
        new_response="New response",
        ref_response="Old response",
        explanation="Test explanation",
        agent_id="test-agent",
        timestamp=datetime.utcnow(),
    )
    store.save_violation(v)

    stats = store.get_stats()
    assert stats["total_calls"] == 3
    assert stats["total_violations"] == 1
    assert stats["critical"] == 1
    assert stats["warning"] == 0
    assert stats["info"] == 0
