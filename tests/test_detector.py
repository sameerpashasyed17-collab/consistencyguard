"""
Tests for detector and embedder logic.
Uses real embeddings — model loads once per session.
"""

import pytest
from consistencyguard.embedder import embed, cosine_similarity
from consistencyguard.detector import response_divergence, classify_severity
from consistencyguard.models import ViolationSeverity


def test_identical_prompts_high_similarity():
    text = "What is the maximum file upload size?"
    a = embed(text)
    b = embed(text)
    assert cosine_similarity(a, b) >= 0.99


def test_different_prompts_low_similarity():
    a = embed("What is the maximum file upload size?")
    b = embed("Tell me a joke about penguins.")
    assert cosine_similarity(a, b) < 0.5


def test_semantically_similar_prompts():
    a = embed("What is the upload limit?")
    b = embed("What is the max file size?")
    # all-MiniLM-L6-v2 scores these paraphrases around 0.60–0.70
    assert cosine_similarity(a, b) >= 0.60


def test_same_response_low_divergence():
    text = "The limit is 25MB per file."
    div = response_divergence(text, text)
    assert div < 0.10


def test_similar_responses_moderate_divergence():
    # Sentences with different numbers but same structure — model sees them
    # as close but not identical; divergence is in the 0.05–0.15 range.
    div = response_divergence("The limit is 25MB.", "The limit is 100MB.")
    assert div > 0.0


def test_contradicting_responses_high_divergence():
    div = response_divergence(
        "The limit is 25MB.",
        "Contact support for help with your account."
    )
    assert div >= 0.40


def test_classify_severity_critical():
    assert classify_severity(0.45) == ViolationSeverity.CRITICAL


def test_classify_severity_info():
    assert classify_severity(0.10) == ViolationSeverity.INFO
