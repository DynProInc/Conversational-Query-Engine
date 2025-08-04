from __future__ import annotations
import uuid, difflib, datetime as dt
from pathlib import Path
import pandas as pd
from utils.file_csv_logger import CSVLogger, DATA_DIR

exec_logger = CSVLogger(
    "executions.csv",
    ["execution_id","timestamp","client_id","prompt","model",
     "generated_query","success","error_message"]
)
feedback_logger = CSVLogger(
    "feedback.csv",
    ["feedback_id","execution_id","timestamp","type",
     "text","corrected_query","user_id"]
)

class FeedbackManager:
    """Write executions / feedback; fetch latest feedback snippet."""

    # ---------- write-side ---------- #
    @staticmethod
    def log_execution(*, exec_id: str, client_id: str, prompt: str,
                       model: str, generated_query: str,
                       success: bool, error_message: str="") -> None:
        exec_logger.append([
            exec_id,
            dt.datetime.now().isoformat(timespec="seconds"),
            client_id, prompt, model,
            generated_query, success, error_message
        ])

    @staticmethod
    def log_feedback(*, fb_id: str, exec_id: str, fb_type: str,
                      text: str, corrected_query: str, user_id: str) -> None:
        feedback_logger.append([
            fb_id, exec_id,
            dt.datetime.now().isoformat(timespec="seconds"),
            fb_type, text, corrected_query, user_id
        ])

    # ---------- read-side ---------- #
    @staticmethod
    def latest_feedback_for(prompt: str, client_id: str = None, exact_client_match: bool = False,
                          min_occurrences: int = 1, max_age_hours: int = None,
                          exact_match: bool = False) -> tuple[str, str, str] | None:
        """Return newest feedback text and corrected query whose source prompt ~ matches input.
        
        Args:
            prompt: The prompt to find feedback for
            client_id: Optional client ID to filter feedback by
            exact_client_match: If True, only return feedback from the exact same client
            min_occurrences: Minimum number of times feedback must appear to be returned
            max_age_hours: Only return feedback from within this time window
            exact_match: If True, only return feedback for exactly matching prompts
            
        Returns:
            Tuple of (feedback_text, corrected_query, feedback_type) or None if no feedback found
        """
        fpath = DATA_DIR / "feedback.csv"
        epath = DATA_DIR / "executions.csv"
        if not (fpath.exists() and epath.exists()):
            return None

        try:
            fdf = pd.read_csv(fpath)
            edf = pd.read_csv(epath, usecols=["execution_id", "prompt", "client_id"])
            df = fdf.merge(edf, on="execution_id", how="left")
            
            # Apply filters based on parameters
            
            # Client filtering
            if exact_client_match and client_id:
                df = df[df["client_id"] == client_id]
            
            # Time window filtering
            if max_age_hours is not None:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                cutoff_time = dt.datetime.now() - dt.timedelta(hours=max_age_hours)
                df = df[df["timestamp"] >= cutoff_time]
            
            # Prompt similarity filtering
            if exact_match:
                mask = df["prompt"] == prompt
            else:
                # Use the original fuzzy matching with ratio threshold 0.85
                mask = df["prompt"].apply(lambda p: difflib.SequenceMatcher(None, str(p).lower(), prompt.lower()).ratio() > 0.85)
            
            filtered_df = df.loc[mask]
            
            # Occurrence count filtering
            if min_occurrences > 1:
                # Group by feedback text and count occurrences
                text_counts = filtered_df.groupby("text").size()
                # Filter to texts that appear at least min_occurrences times
                frequent_texts = text_counts[text_counts >= min_occurrences].index
                if len(frequent_texts) == 0:
                    return None
                # Filter to only those texts
                filtered_df = filtered_df[filtered_df["text"].isin(frequent_texts)]
            
            # Sort by timestamp and get most recent
            recent = filtered_df.sort_values("timestamp", ascending=False)
            if recent.empty:
                return None
                
            # Get the most recent feedback row
            feedback_row = recent.iloc[0]
            feedback_text = str(feedback_row["text"]).strip() or ""
            corrected_query = str(feedback_row["corrected_query"]).strip() or ""
            feedback_type = str(feedback_row["type"]).strip() or ""
            
            # Return all feedback information
            return feedback_text, corrected_query, feedback_type
        except Exception as e:
            print(f"Error retrieving feedback: {e}")
            return None
            
    @staticmethod
    def get_all_recent_feedback(prompt: str, client_id: str = None, exact_client_match: bool = False,
                               max_age_minutes: int = 20, exact_match: bool = False, max_entries: int = None,
                               min_confidence: float = 0.85) -> list[dict]:
        """Return recent feedback for a prompt within the specified time window and constraints.
        
        Args:
            prompt: The prompt to find feedback for
            client_id: Optional client ID to filter feedback by (e.g., 'mts' for specific client)
            exact_client_match: If True, only return feedback from the exact same client
            max_age_minutes: Only return feedback from within this time window (in minutes)
            exact_match: If True, only return feedback for exactly matching prompts
            max_entries: Maximum number of recent feedback entries to return (None = all entries)
            min_confidence: Minimum similarity threshold for fuzzy matching (0.0-1.0, default: 0.85)
            
        Returns:
            List of feedback dictionaries with text, corrected_query, type, and timestamp,
            limited to max_entries if specified, sorted by most recent first
        """
        fpath = DATA_DIR / "feedback.csv"
        epath = DATA_DIR / "executions.csv"
        if not (fpath.exists() and epath.exists()):
            return []

        try:
            fdf = pd.read_csv(fpath)
            edf = pd.read_csv(epath, usecols=["execution_id", "prompt", "client_id"])
            df = fdf.merge(edf, on="execution_id", how="left")
            
            # Apply filters based on parameters
            
            # Client filtering
            if exact_client_match and client_id:
                df = df[df["client_id"] == client_id]
            
            # Time window filtering
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff_time = dt.datetime.now() - dt.timedelta(minutes=max_age_minutes)
            df = df[df["timestamp"] >= cutoff_time]
            
            # Prompt similarity filtering
            if exact_match:
                mask = df["prompt"] == prompt
            else:
                # Use the original fuzzy matching with ratio threshold 0.85
                mask = df["prompt"].apply(lambda p: difflib.SequenceMatcher(None, str(p).lower(), prompt.lower()).ratio() > 0.85)
            
            filtered_df = df.loc[mask]
            
            # Sort by timestamp (newest first)
            recent = filtered_df.sort_values("timestamp", ascending=False)
            if recent.empty:
                return []
            
            # Apply max_entries limit if specified
            if max_entries is not None and max_entries > 0:
                recent = recent.head(max_entries)
                print(f"Limiting feedback to {max_entries} most recent entries")
            
            # Convert to list of dictionaries
            feedback_list = []
            for _, row in recent.iterrows():
                feedback_list.append({
                    "text": str(row["text"]).strip() or "",
                    "corrected_query": str(row["corrected_query"]).strip() or "",
                    "type": str(row["type"]).strip() or "",
                    "timestamp": row["timestamp"],
                    "client_id": str(row["client_id"]) if "client_id" in row and pd.notna(row["client_id"]) else ""
                })
            
            print(f"Returning {len(feedback_list)} feedback entries")
            return feedback_list
        except Exception as e:
            print(f"Error retrieving all feedback: {e}")
            return []
