"""Unit tests for Aura ConversationProcessor — US-211.

Tests cover:
  1. Stress keyword detection
  2. Positive keyword detection
  3. Word-boundary stressor matching (US-204 regression)
  4. Fatigue detection
  5. Override mention detection
  6. Sentiment score bounds
  7. Emotional state classification
  8. Confidence trend tracking across messages
  9. Topic extraction
  10. Session reset
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aura.core.conversation_processor import (
    ConversationProcessor,
    STRESSOR_KEYWORDS,
    STRESS_KEYWORDS,
    POSITIVE_KEYWORDS,
)


@pytest.fixture
def processor():
    return ConversationProcessor()


# ── 1. Stress keyword detection ──────────────────────────────────────────

def test_stress_keywords_detected(processor):
    signals = processor.process_message("I'm so stressed and overwhelmed today")
    assert "stressed" in signals.stress_keywords_found
    assert "overwhelmed" in signals.stress_keywords_found


def test_no_false_positive_stress(processor):
    """Neutral message should have no stress keywords."""
    signals = processor.process_message("The weather is nice today")
    assert len(signals.stress_keywords_found) == 0


# ── 2. Positive keyword detection ────────────────────────────────────────

def test_positive_keywords_detected(processor):
    signals = processor.process_message("I feel amazing and confident about today")
    assert "amazing" in signals.positive_keywords_found
    assert "confident" in signals.positive_keywords_found


# ── 3. Word-boundary stressor matching (US-204) ─────────────────────────

def test_stressor_no_false_positive_apartment(processor):
    """'parent' should NOT match inside 'apartment'."""
    signals = processor.process_message("I moved to a new apartment last week")
    stressor_labels = signals.detected_stressors
    assert "family responsibility" not in stressor_labels


def test_stressor_no_false_positive_quite(processor):
    """'quit' should NOT match inside 'quite'."""
    signals = processor.process_message("I'm quite happy with everything")
    stressor_labels = signals.detected_stressors
    assert "career change" not in stressor_labels


def test_stressor_job_standalone(processor):
    """'job' as a standalone word should match."""
    signals = processor.process_message("I'm worried about my job security")
    stressor_labels = signals.detected_stressors
    assert "career change" in stressor_labels


def test_stressor_no_false_positive_enjoyable(processor):
    """'job' should NOT match inside 'enjoyable' — no word boundary there."""
    signals = processor.process_message("That was an enjoyable experience")
    stressor_labels = signals.detected_stressors
    assert "career change" not in stressor_labels


def test_stressor_exact_word_match(processor):
    """Exact 'parent' as standalone word should match."""
    signals = processor.process_message("I'm worried about my parent being ill")
    assert "family responsibility" in signals.detected_stressors


def test_stressor_multi_word_still_works(processor):
    """Multi-word stressor keywords should still match via substring."""
    # The STRESSOR_KEYWORDS dict uses single words, but STRESS_KEYWORDS uses
    # multi-word phrases like "can't sleep" which use substring matching.
    signals = processor.process_message("I can't sleep at night anymore")
    assert "insomnia" in signals.stress_keywords_found or "can't sleep" in signals.stress_keywords_found


# ── 4. Fatigue detection ─────────────────────────────────────────────────

def test_fatigue_detected(processor):
    signals = processor.process_message("I'm completely exhausted, didn't sleep well")
    assert signals.fatigue_detected is True


def test_no_fatigue_on_neutral(processor):
    signals = processor.process_message("Had a productive day at work")
    assert signals.fatigue_detected is False


# ── 5. Override mention detection ────────────────────────────────────────

def test_override_detected(processor):
    signals = processor.process_message("I ignored buddy and took the trade anyway")
    assert signals.override_mentioned is True


def test_no_override_on_normal(processor):
    signals = processor.process_message("Buddy suggested a good trade today")
    assert signals.override_mentioned is False


# ── 6. Sentiment score bounds ────────────────────────────────────────────

def test_sentiment_always_bounded(processor):
    """Sentiment must always be in [0, 1], even with extreme input."""
    # Very negative
    signals = processor.process_message(
        "stressed exhausted overwhelmed angry frustrated anxious "
        "burnout insomnia losing money can't sleep"
    )
    assert 0.0 <= signals.sentiment_score <= 1.0

    # Very positive
    processor2 = ConversationProcessor()
    signals2 = processor2.process_message(
        "amazing wonderful happy excited energized confident "
        "productive inspired motivated grateful optimistic winning"
    )
    assert 0.0 <= signals2.sentiment_score <= 1.0


# ── 7. Emotional state classification ────────────────────────────────────

def test_emotional_state_fatigued(processor):
    signals = processor.process_message("I'm so tired and drained")
    assert signals.emotional_state == "fatigued"


def test_emotional_state_stressed(processor):
    signals = processor.process_message("stressed overwhelmed angry frustrated losing money")
    assert signals.emotional_state == "stressed"


def test_emotional_state_energized(processor):
    signals = processor.process_message("amazing wonderful excited inspired motivated")
    assert signals.emotional_state == "energized"


def test_emotional_state_neutral(processor):
    signals = processor.process_message("The market opened at 9am")
    assert signals.emotional_state == "neutral"


# ── 8. Confidence trend tracking ─────────────────────────────────────────

def test_confidence_trend_falling(processor):
    """Positive first → negative second should show 'falling'."""
    processor.process_message("I feel great and confident today")
    signals = processor.process_message("Now I'm stressed and angry about my losses")
    assert signals.confidence_trend == "falling"


def test_confidence_trend_rising(processor):
    """Negative first → positive second should show 'rising'."""
    processor.process_message("I'm stressed and overwhelmed")
    signals = processor.process_message("Actually I feel amazing and motivated now")
    assert signals.confidence_trend == "rising"


# ── 9. Topic extraction ─────────────────────────────────────────────────

def test_topics_trading(processor):
    signals = processor.process_message("How's Buddy doing on the forex market today?")
    assert "trading" in signals.topics


def test_topics_career(processor):
    signals = processor.process_message("I'm thinking about my career and job options")
    assert "career" in signals.topics


# ── 10. Session reset ───────────────────────────────────────────────────

def test_session_reset(processor):
    processor.process_message("I'm stressed")
    processor.process_message("I'm overwhelmed")
    summary = processor.get_session_summary()
    assert summary["message_count"] == 2
    assert summary["cumulative_stress"] > 0

    processor.reset_session()
    summary = processor.get_session_summary()
    assert summary["message_count"] == 0
    assert summary["cumulative_stress"] == 0.0


# ── 11. Assistant messages are not analyzed ──────────────────────────────

def test_assistant_message_not_analyzed(processor):
    """Only user messages should update emotional signals."""
    signals = processor.process_message("I'm so stressed!", role="assistant")
    assert signals.emotional_state == "neutral"  # default, not analyzed
    assert len(signals.stress_keywords_found) == 0
