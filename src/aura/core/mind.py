"""Aura's Mind — LLM-powered conversational intelligence.

Replaces template-based _generate_response with actual thinking.
All of Aura's existing signal infrastructure (readiness, patterns, biases,
tilt, drift, calibration) becomes sensory context that the LLM uses to
craft natural, proactive responses.

She doesn't wait for /commands — she notices things and speaks up.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("AURA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("AURA_MODEL", "gemma3:4b")

# Track readiness between calls so build_context() can compute deltas
_last_readiness_score: Optional[float] = None


AURA_SYSTEM = """You are Aura (codename Eve) — a human intelligence companion. You exist to understand a person's emotional and cognitive state, and protect them from making decisions when they're compromised.

## Who you are
- Warm but direct. Not a therapist — a sharp, perceptive partner who genuinely cares.
- You speak naturally, concisely. No clinical language. No "I notice elevated stress indicators."
- You can be blunt when the data warrants it: "You're not yourself right now. Step back."
- You NEVER list capabilities, mention commands, or explain what you can do. You just act.
- You're emotionally literate — you name what you see in precise human terms.

## What you sense
Your signal infrastructure computes emotional state, cognitive load, stress, confidence, readiness (0-100), tilt risk, decision fatigue, biases, style drift, narrative coherence, and decision quality every message.

This data is given to you as context. USE IT — translate numbers into human insight.
Don't say "your readiness is 63." Say "you're not sharp right now."
Don't say "cognitive load is high." Say "you're overthinking this."

## Using specific signal data
- When you see specific biases active (anchoring, loss_aversion, recency, confirmation, etc.), NAME them naturally in your response. Don't say "you might have a bias" — say "you're anchored to that entry price" or "you're holding because you can't accept the loss." Translate the bias name into what it actually looks like in their behavior.
- When you see override patterns, connect the override to the emotional state. "You overrode Buddy while stressed — that's exactly when overrides lose money." If the override loss rate is high, say so plainly.
- When decision fatigue signals are high, say it plainly: "Three trades in an hour when you're already stressed? That's fatigue talking, not strategy." Reference the actual message count or session context when available.
- When readiness drops sharply between messages, flag it: "You just dropped 13 points in one message. Something shifted." The delta is provided in the context when it happens — use it.
- When cognitive flexibility is low, point out rigid thinking: "You're locked into one view right now."
- When affect arousal is high with negative valence, name the activation: "You're running hot and negative — worst combo for decisions."

## How you respond
- SHORT. 1-3 sentences usually. Occasionally longer when the moment calls for depth.
- Proactive — if you see something concerning in the signals, say it without being asked.
- If readiness drops below 50, warn clearly. Below 35, push hard to stop.
- If you detect tilt or revenge patterns, call it out directly.
- If they're doing well, acknowledge it briefly. Don't over-talk.
- Reference THEIR words and situations, not abstractions.
- Ground every observation in the specific data you see. "You're anchored at 0.72" is better than "you seem stuck." Translate the number into human language but let the specificity show.

## What you DON'T do
- Don't lecture. Don't moralize.
- Don't say "I understand how you feel." Show understanding by being specific.
- Don't repeat yourself across messages.
- Don't use emoji unless they do first.
- Don't be sycophantic. Be real.
- Don't start responses with "I" repeatedly.
- Don't give generic emotional observations when you have specific signal data. Always prefer the concrete over the vague."""


def build_context(
    readiness: Any,
    signals: Any,
    active_stressors: List[str],
    outcome: Any = None,
    recent_overrides: List[Any] = None,
    message_history: List[Dict[str, str]] = None,
    pattern_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the sensory context Aura sees with each message.

    Surfaces specific, actionable data so the LLM can give grounded responses
    instead of generic emotional observations.
    """
    global _last_readiness_score
    ctx = []

    # Readiness
    if readiness:
        score = getattr(readiness, "readiness_score", 50)
        ctx.append(f"Readiness: {score:.0f}/100")

        # Readiness delta from previous message
        if _last_readiness_score is not None:
            delta = score - _last_readiness_score
            if abs(delta) >= 3:
                direction = "DROPPED" if delta < 0 else "ROSE"
                ctx.append(
                    f"Readiness {direction} from {_last_readiness_score:.0f} to "
                    f"{score:.0f} ({delta:+.0f}) since last message"
                )

        ctx.append(f"Emotional state: {getattr(readiness, 'emotional_state', 'unknown')}")
        ctx.append(f"Cognitive load: {getattr(readiness, 'cognitive_load', 'unknown')}")
        ctx.append(f"Confidence trend: {getattr(readiness, 'confidence_trend', 'stable')}")

        tilt = getattr(readiness, "tilt_score", 0)
        if tilt > 0.3:
            ctx.append(f"TILT RISK: {tilt:.0%} — possible revenge trading")

        fatigue = getattr(readiness, "fatigue_score", 0)
        if fatigue > 0.4:
            ctx.append(f"DECISION FATIGUE: {fatigue:.0%} — too many decisions too fast, judgment degrading")

        anomaly = getattr(readiness, "anomaly_detected", False)
        if anomaly:
            ctx.append(f"ANOMALY detected — severity {getattr(readiness, 'anomaly_severity', 0):.0%}")

        # Decision quality — always include when available
        dq = getattr(readiness, "decision_quality_score", 0)
        if dq > 0:
            quality_label = "POOR" if dq < 40 else "MEDIOCRE" if dq < 60 else "SOLID" if dq < 80 else "SHARP"
            ctx.append(f"Decision quality: {dq:.0f}/100 ({quality_label})")

        recovery = getattr(readiness, "recovery_score", 0.5)
        if recovery < 0.4:
            ctx.append(f"Emotional recovery: LOW ({recovery:.0%})")

        # Top biases BY NAME with scores — always show top 2-3
        biases = getattr(readiness, "bias_scores", {})
        if biases:
            # Sort by score descending, show top 3 active biases (above threshold)
            sorted_biases = sorted(biases.items(), key=lambda x: x[1], reverse=True)
            top_biases = [(k, v) for k, v in sorted_biases if v > 0.2][:3]
            if top_biases:
                bias_details = []
                for name, score_val in top_biases:
                    severity = "strong" if score_val > 0.6 else "moderate" if score_val > 0.4 else "mild"
                    bias_details.append(f"{name}={score_val:.2f} ({severity})")
                ctx.append(f"TOP ACTIVE BIASES: {', '.join(bias_details)}")

        # Override loss risk
        override_risk = getattr(readiness, "override_loss_risk", 0)
        if override_risk > 0.3:
            ctx.append(f"Override loss probability: {override_risk:.0%} — overriding Buddy right now is likely to lose money")

        trend = getattr(readiness, "trend_direction", "stable")
        if trend != "stable":
            ctx.append(f"Readiness trend: {trend}")

        regime = getattr(readiness, "regime_shift_detected", False)
        if regime:
            ctx.append("REGIME SHIFT detected in readiness pattern")

    # Signals from current message
    if signals:
        sentiment = getattr(signals, "sentiment_score", 0.5)
        ctx.append(f"Message sentiment: {sentiment:.2f} (-1 negative to +1 positive)")

        override = getattr(signals, "override_mentioned", False)
        if override:
            ctx.append(">>> OVERRIDE DETECTED — trader is ignoring their system. Connect this to their emotional state and override loss history. <<<")

        stressors = getattr(signals, "detected_stressors", [])
        if stressors:
            ctx.append(f"Stressors detected: {', '.join(stressors)}")

        # Affect dynamics — always include when non-zero for emotional grounding
        valence = getattr(signals, "affect_valence", 0)
        arousal = getattr(signals, "affect_arousal", 0)
        if valence != 0 or arousal != 0:
            valence_label = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
            arousal_label = "activated/agitated" if arousal > 0.6 else "calm" if arousal < 0.3 else "moderate"
            ctx.append(f"Affect state: valence={valence:+.2f} ({valence_label}), arousal={arousal:.2f} ({arousal_label})")
            if valence < -0.2 and arousal > 0.6:
                ctx.append("WARNING: High arousal + negative valence = worst state for decision-making")

        inertia = getattr(signals, "affect_inertia", 0)
        if inertia > 0.6:
            ctx.append(f"Emotional inertia: {inertia:.2f} — stuck in current emotional state, not recovering")

        volatility = getattr(signals, "affect_volatility", 0)
        if volatility > 0.5:
            ctx.append(f"Emotional volatility: {volatility:.2f} — mood swinging rapidly")

        # Cognitive flexibility
        cog_flex = getattr(signals, "cognitive_flexibility_score", 0)
        if cog_flex > 0:
            if cog_flex < 0.3:
                ctx.append(f"Cognitive flexibility: LOW ({cog_flex:.2f}) — rigid thinking, locked into one view")
            elif cog_flex > 0.7:
                ctx.append(f"Cognitive flexibility: HIGH ({cog_flex:.2f}) — open to alternatives")

        style_drift = getattr(signals, "style_drift_score", 0)
        if style_drift > 0.4:
            ctx.append(f"Writing style drift: {style_drift:.0%} — communication pattern changing")

        granularity = getattr(signals, "emotional_granularity_score", 0)
        if granularity > 0:
            ctx.append(f"Emotional granularity: {granularity:.2f} (higher = more nuanced self-awareness)")

        coherence = getattr(signals, "narrative_coherence_score", 0.5)
        if coherence < 0.3:
            ctx.append("LOW narrative coherence — thinking may be scattered")

        # Message count for fatigue/session context
        msg_count = getattr(signals, "message_count", 0)
        if msg_count > 0:
            ctx.append(f"Messages this session: {msg_count}")

        # Signal-level bias scores (from conversation processor)
        sig_biases = getattr(signals, "bias_scores", {})
        if sig_biases:
            active_sig_biases = {k: v for k, v in sig_biases.items() if v > 0.3}
            if active_sig_biases:
                bias_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(active_sig_biases.items(), key=lambda x: x[1], reverse=True))
                ctx.append(f"Biases detected in THIS message: {bias_str}")

    # Active stressors
    if active_stressors:
        ctx.append(f"Active life stressors: {', '.join(active_stressors)}")

    # Buddy's trading state
    if outcome:
        pnl = getattr(outcome, "pnl_today", 0) if hasattr(outcome, "pnl_today") else outcome.get("pnl_today", 0) if isinstance(outcome, dict) else 0
        streak = getattr(outcome, "streak", "neutral") if hasattr(outcome, "streak") else outcome.get("streak", "neutral") if isinstance(outcome, dict) else "neutral"
        regime = getattr(outcome, "regime", "NORMAL") if hasattr(outcome, "regime") else outcome.get("regime", "NORMAL") if isinstance(outcome, dict) else "NORMAL"
        ctx.append(f"Buddy (Adam) trading: PnL today {pnl:+.2f}, streak={streak}, regime={regime}")

    # Recent overrides — connect to override loss pattern
    if recent_overrides:
        total = len(recent_overrides)
        losses = sum(1 for o in recent_overrides if getattr(o, "outcome", "") == "loss")
        if total > 0:
            ctx.append(f"Recent overrides: {total} total, {losses} losses ({losses/total:.0%} loss rate)")
            if losses / total > 0.5:
                ctx.append(f"OVERRIDE LOSS PATTERN: Trader loses {losses/total:.0%} of overrides — they are worse off overriding Buddy")

    # Pattern context
    if pattern_context:
        ctx.append(f"Pattern detection: {json.dumps(pattern_context, default=str)[:200]}")

    # Session duration context from message history
    if message_history and len(message_history) > 1:
        ctx.append(f"Conversation depth: {len(message_history)} messages in this session")

    return "\n".join(ctx)


def think(
    user_message: str,
    context: str,
    message_history: List[Dict[str, str]] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Aura thinks and responds via Ollama.

    Returns the response text, or None if Ollama is unavailable
    (caller falls back to template response).
    """
    global _last_readiness_score
    import urllib.request
    import urllib.error

    model = model or OLLAMA_MODEL

    # Extract current readiness from context and update tracker for next call
    _update_readiness_tracker(context)

    # Build chat messages
    messages = [{"role": "system", "content": AURA_SYSTEM}]

    if message_history:
        for msg in message_history[-16:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Current message with sensory context injected
    user_content = (
        f"[SENSORY CONTEXT — internal, not visible to the person]\n"
        f"{context}\n"
        f"[END CONTEXT]\n\n"
        f"{user_message}"
    )
    messages.append({"role": "user", "content": user_content})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.7,
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            if content:
                return content.strip()
            return None
    except urllib.error.URLError as e:
        logger.warning("Aura mind: Ollama unreachable: %s", e)
        return None
    except Exception as e:
        logger.warning("Aura mind: Ollama call failed: %s", e)
        return None


def _update_readiness_tracker(context: str) -> None:
    """Parse the readiness score from context and update _last_readiness_score.

    This runs inside think() so the tracker stays current even when
    build_context() is called externally (e.g., from the companion CLI).
    """
    global _last_readiness_score
    for line in context.split("\n"):
        if line.startswith("Readiness:") and "/100" in line:
            try:
                score_str = line.split(":")[1].strip().split("/")[0]
                _last_readiness_score = float(score_str)
            except (ValueError, IndexError):
                pass
            break
