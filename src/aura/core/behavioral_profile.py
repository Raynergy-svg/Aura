"""Behavioral Profile Consolidation.

US-360: Persists and classifies behavioral patterns into strengths, vulnerabilities,
and signatures using SQLite or in-memory storage.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """A single behavioral pattern entry."""
    domain: str             # e.g., "trading", "emotional", "cognitive"
    pattern_name: str       # e.g., "loss_aversion", "overconfidence"
    frequency: float        # 0.0 to 1.0
    strength: float         # 0.0 to 1.0
    category: str           # "strength" | "vulnerability" | "signature"


class BehavioralProfile:
    """Consolidates and persists behavioral patterns across domains.

    Uses SQLite for persistence if db_path is provided, otherwise in-memory dict.
    Classifies patterns as:
    - strength: strength > 0.7
    - vulnerability: strength < 0.3
    - signature: everything else (0.3 <= strength <= 0.7)

    profile_score = min(len(strengths)*10, 40) + max(0, 40 - len(vulnerabilities)*10) + 20
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path
        self._memory_store: Dict[str, ProfileEntry] = {}
        self._db_conn: Optional[sqlite3.Connection] = None

        if db_path is not None:
            self._init_db(db_path)

    def _init_db(self, db_path: Path) -> None:
        """Initialize SQLite database for persistence."""
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db_conn.execute(
                """CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    key TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    frequency REAL NOT NULL,
                    strength REAL NOT NULL,
                    category TEXT NOT NULL
                )"""
            )
            self._db_conn.commit()
            # Load existing data into memory
            cursor = self._db_conn.execute(
                "SELECT key, domain, pattern_name, frequency, strength, category FROM behavioral_patterns"
            )
            for row in cursor.fetchall():
                key, domain, pattern_name, frequency, strength, category = row
                self._memory_store[key] = ProfileEntry(
                    domain=domain,
                    pattern_name=pattern_name,
                    frequency=frequency,
                    strength=strength,
                    category=category,
                )
        except Exception as e:
            logger.warning("US-360: Failed to init SQLite, using in-memory: %s", e)
            self._db_conn = None

    def _make_key(self, domain: str, pattern_name: str) -> str:
        return f"{domain}:{pattern_name}"

    def update_pattern(self, domain: str, pattern_name: str, frequency: float, strength: float) -> None:
        """Upsert a behavioral pattern with classification.

        Args:
            domain: Pattern domain (e.g., "trading")
            pattern_name: Name of the pattern
            frequency: How often the pattern appears (0-1)
            strength: How strong the pattern is (0-1)
        """
        category = self.classify_pattern(frequency, strength)
        key = self._make_key(domain, pattern_name)
        entry = ProfileEntry(
            domain=domain,
            pattern_name=pattern_name,
            frequency=frequency,
            strength=strength,
            category=category,
        )
        self._memory_store[key] = entry

        if self._db_conn is not None:
            try:
                self._db_conn.execute(
                    """INSERT OR REPLACE INTO behavioral_patterns
                       (key, domain, pattern_name, frequency, strength, category)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (key, domain, pattern_name, frequency, strength, category),
                )
                self._db_conn.commit()
            except Exception as e:
                logger.warning("US-360: Failed to persist pattern: %s", e)

    def classify_pattern(self, frequency: float, strength: float) -> str:
        """Classify pattern as strength, vulnerability, or signature.

        Args:
            frequency: Pattern frequency (0-1)
            strength: Pattern strength (0-1)

        Returns:
            "strength" if strength > 0.7
            "vulnerability" if strength < 0.3
            "signature" otherwise
        """
        if strength > 0.7:
            return "strength"
        elif strength < 0.3:
            return "vulnerability"
        else:
            return "signature"

    def get_profile_summary(self) -> Dict[str, Any]:
        """Compute profile summary with categorized patterns and score.

        Returns:
            Dict with "strengths", "vulnerabilities", "signatures" lists and "profile_score"
        """
        strengths: List[Dict[str, Any]] = []
        vulnerabilities: List[Dict[str, Any]] = []
        signatures: List[Dict[str, Any]] = []

        for entry in self._memory_store.values():
            entry_dict = {
                "domain": entry.domain,
                "pattern_name": entry.pattern_name,
                "frequency": entry.frequency,
                "strength": entry.strength,
                "category": entry.category,
            }
            if entry.category == "strength":
                strengths.append(entry_dict)
            elif entry.category == "vulnerability":
                vulnerabilities.append(entry_dict)
            else:
                signatures.append(entry_dict)

        # profile_score: max 40 from strengths, max 40 from lack of vulnerabilities, 20 base
        strength_contribution = min(len(strengths) * 10, 40)
        vulnerability_contribution = max(0, 40 - len(vulnerabilities) * 10)
        profile_score = strength_contribution + vulnerability_contribution + 20

        return {
            "strengths": strengths,
            "vulnerabilities": vulnerabilities,
            "signatures": signatures,
            "profile_score": profile_score,
        }

    def save_state(self, path: Path) -> None:
        """Save profile to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: {
                    "domain": e.domain,
                    "pattern_name": e.pattern_name,
                    "frequency": e.frequency,
                    "strength": e.strength,
                    "category": e.category,
                }
                for key, e in self._memory_store.items()
            }
            path.write_text(json.dumps(data, indent=2))
            logger.debug("US-360: Saved %d patterns to %s", len(data), path)
        except Exception as e:
            logger.warning("US-360: Failed to save state: %s", e)

    def load_state(self, path: Path) -> None:
        """Load profile from JSON file."""
        try:
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._memory_store.clear()
            for key, entry_data in data.items():
                self._memory_store[key] = ProfileEntry(
                    domain=entry_data["domain"],
                    pattern_name=entry_data["pattern_name"],
                    frequency=entry_data["frequency"],
                    strength=entry_data["strength"],
                    category=entry_data["category"],
                )
            logger.debug("US-360: Loaded %d patterns from %s", len(self._memory_store), path)
        except Exception as e:
            logger.warning("US-360: Failed to load state: %s", e)
