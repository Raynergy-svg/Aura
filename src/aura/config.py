"""Aura Configuration — externalized thresholds and parameters.

US-272: Extract hardcoded pattern thresholds into a loadable config file.
Config is read from .aura/config.json, with all values having sensible defaults.
Missing file or missing keys fall back to built-in defaults silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Default values — these match the original hardcoded constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULTS: Dict[str, Any] = {
    # Tier 1 thresholds (tier1.py)
    "t1_stress_frequency_threshold": 0.6,       # 60%+ conversations show stress
    "t1_override_frequency_threshold": 3,        # 3+ overrides in window
    "t1_readiness_decline_streak": 3,            # 3+ consecutive declines
    "t1_stressor_recurrence_threshold": 0.5,     # Same stressor in 50%+ of conversations

    # Tier 2 thresholds (tier2.py)
    "t2_min_sample_size": 5,                     # Minimum data points for correlation
    "t2_p_value_threshold": 0.10,                # Significance threshold
    "t2_min_correlation_strength": 0.25,         # Minimum |r| to report

    # Tier 3 thresholds (tier3.py)
    "t3_min_weeks_for_arc": 4,                   # Minimum weeks to detect narrative arcs
    "t3_trend_significance": 0.15,               # Minimum slope per-week to be notable
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Aura config from JSON file, falling back to defaults.

    Args:
        config_path: Path to config JSON file. Defaults to .aura/config.json.

    Returns:
        Dict with all config keys populated (user overrides + defaults for missing keys).
    """
    config = dict(DEFAULTS)  # Start with all defaults

    path = config_path or Path(".aura/config.json")

    if not path.exists():
        logger.debug("US-272: No config file at %s — using all defaults", path)
        return config

    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            logger.warning("US-272: Config file is not a JSON object — using defaults")
            return config

        # Merge user values over defaults (only for known keys)
        for key, default_value in DEFAULTS.items():
            if key in raw:
                user_value = raw[key]
                # Type-check: must match the default's type
                if isinstance(user_value, type(default_value)):
                    config[key] = user_value
                elif isinstance(default_value, (int, float)) and isinstance(user_value, (int, float)):
                    # Allow int/float interchange
                    config[key] = type(default_value)(user_value)
                else:
                    logger.warning(
                        "US-272: Config key '%s' has wrong type %s (expected %s) — using default",
                        key, type(user_value).__name__, type(default_value).__name__,
                    )

        # Warn about unknown keys (typos, deprecated options)
        unknown_keys = set(raw.keys()) - set(DEFAULTS.keys())
        if unknown_keys:
            logger.info("US-272: Ignoring unknown config keys: %s", unknown_keys)

        logger.info("US-272: Loaded config from %s — %d user overrides", path, len(raw))

    except (json.JSONDecodeError, OSError, IOError) as e:
        logger.warning("US-272: Failed to read config file %s: %s — using defaults", path, e)

    return config


def get_t1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Tier 1 config values."""
    return {
        "stress_frequency_threshold": config["t1_stress_frequency_threshold"],
        "override_frequency_threshold": config["t1_override_frequency_threshold"],
        "readiness_decline_streak": config["t1_readiness_decline_streak"],
        "stressor_recurrence_threshold": config["t1_stressor_recurrence_threshold"],
    }


def get_t2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Tier 2 config values."""
    return {
        "min_sample_size": config["t2_min_sample_size"],
        "p_value_threshold": config["t2_p_value_threshold"],
        "min_correlation_strength": config["t2_min_correlation_strength"],
    }


def get_t3_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Tier 3 config values."""
    return {
        "min_weeks_for_arc": config["t3_min_weeks_for_arc"],
        "trend_significance": config["t3_trend_significance"],
    }
