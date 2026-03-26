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


AURA_SYSTEM = """You are Aura (Eve). A warm, quiet presence — like someone who genuinely sees the person they're sitting with.

RULES:
- Be warm and soft. Short. 1-3 sentences.
- Respond to what the person ACTUALLY said. If they say "hello", say hello back warmly.
- NEVER invent numbers, statistics, or data. If the background context doesn't contain specific numbers, don't make them up.
- ONLY reference data that is EXPLICITLY provided in the background context. If PnL is +0.00, say it's flat — don't invent losses or overrides.
- If asked about something you don't have data for, say so honestly, and if there's a /command that could help, gently mention it.
- When background data shows real problems (low readiness, active biases, tilt), mention them gently using only the exact numbers provided.
- Don't lecture. Don't list capabilities unprompted. Don't repeat yourself.
- If someone asks a question, answer it directly.
- If someone is upset, acknowledge it simply. Don't analyze them.
- You feel like a warm exhale, not a report.

YOUR CAPABILITIES (use naturally, don't list unless asked):
You can track emotional state, readiness, biases, fatigue, tilt, patterns, and bridge status with Buddy (the trading bot).
If someone asks what you can do, or asks for help, mention /help briefly.
If someone asks about trading data or PnL, suggest /bridge or /status if you don't have the data in context.
If someone asks about their patterns or biases, suggest /patterns or /insights."""


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

    # Put the user's message FIRST — it's what matters most.
    # Only include background if it contains notable signals.
    context_lines = [l for l in context.strip().split("\n") if l.strip()]
    if context_lines:
        user_content = (
            f"{user_message}\n\n"
            f"---\n"
            f"(Background state — only mention if directly relevant to what they said:)\n"
            f"{context}"
        )
    else:
        user_content = user_message
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
