"""Feedback Service
-------------------
A lightweight service to persist and retrieve user feedback for the Conversational
Query Engine. Feedback is appended to newline-delimited JSON (jsonl) files on
disk so that it can later be used for evaluation, fine-tuning, or RLHF.

The design mirrors the simple file-based cache service already present in the
codebase, keeping the footprint minimal and dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default location inside repo root (relative to LLM Query Engine package)
DEFAULT_FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "feedback"

class FeedbackService:  # singleton-ish – stateless but exposed via get_feedback_service()
    """Persist feedback to JSONL files and load it back when needed."""

    def __init__(self, storage_dir: Optional[os.PathLike] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_FEEDBACK_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Feedback storage directory: %s", self.storage_dir)

    # -----------------------------------------------------
    # Public helpers
    # -----------------------------------------------------
    def record(self, feedback: Dict[str, Any]) -> None:
        """Append a feedback dict to a per-client JSONL file.

        Args:
            feedback: A dictionary that **must** include `client_id`.
        """
        client_id = feedback.get("client_id", "default").lower()
        file_path = self._file_for_client(client_id)
        feedback["timestamp"] = feedback.get("timestamp", datetime.utcnow().isoformat())
        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
            logger.debug("Recorded feedback for client '%s'", client_id)
        except Exception as e:
            logger.exception("Failed to record feedback: %s", e)

    def list(self, client_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return feedback entries.

        Args:
            client_id: If provided, filter to that client only.
            limit: Max number of rows (from most recent backwards).
        """
        entries: List[Dict[str, Any]] = []
        paths = [self._file_for_client(client_id)] if client_id else self.storage_dir.glob("*.jsonl")
        for p in paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Corrupt feedback line ignored in %s", p)
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries[:limit] if limit else entries

    # -----------------------------------------------------------------
    # Learning-time helpers
    # -----------------------------------------------------------------
    def get_feedback_entry(self, client_id: str, prompt: str, similarity_threshold: float = 0.9) -> Optional[Dict[str, Any]]:
        """Return the feedback entry (any) best matching a prompt.
        Includes entries without corrected_sql (free-form feedback).
        """
        from difflib import SequenceMatcher
        normalized_query = prompt.strip().lower()
        feedback_entries = self.list(client_id=client_id)
        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry in feedback_entries:
            entry_prompt = entry.get("prompt", "").strip().lower()
            if not entry_prompt:
                continue
            if entry_prompt == normalized_query:
                return entry
            score = SequenceMatcher(None, normalized_query, entry_prompt).ratio()
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_entry = entry
        return best_entry

    def get_corrected_sql(self, client_id: str, prompt: str, similarity_threshold: float = 0.9) -> Optional[Dict[str, Any]]:
        """Return feedback entry containing `corrected_sql` that best matches the prompt.

        Exact match on normalized prompt is preferred; otherwise fall back to a
        simple fuzzy match using difflib.SequenceMatcher.
        """
        from difflib import SequenceMatcher

        normalized_query = prompt.strip().lower()
        feedback_entries = self.list(client_id=client_id)

        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for entry in feedback_entries:
            corrected_sql = entry.get("corrected_sql")
            if not corrected_sql:
                continue  # Nothing to learn from

            entry_prompt = entry.get("prompt", "").strip().lower()
            if not entry_prompt:
                continue

            if entry_prompt == normalized_query:
                return entry  # perfect match

            # Fuzzy similarity
            score = SequenceMatcher(None, normalized_query, entry_prompt).ratio()
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_entry = entry

        return best_entry

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------
    def _file_for_client(self, client_id: str) -> Path:
        safe_id = client_id.replace("/", "_")
        return self.storage_dir / f"{safe_id}.jsonl"

# ---------------------------------------------------------------------------
# Helper to obtain a singleton instance lazily (mirrors CacheService pattern)
# ---------------------------------------------------------------------------
_feedback_service: Optional[FeedbackService] = None

def get_feedback_service() -> FeedbackService:
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
