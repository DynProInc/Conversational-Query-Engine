from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from fastapi import Depends
from services.feedback_manager import FeedbackManager

# called inside /query/unified before hitting LLM
def build_final_prompt(orig_prompt: str, feedback_mode: str = "never", client_id: str = None, max_feedback_entries: int = None, confidence_threshold: float = None, feedback_time_window_minutes: int = None) -> tuple[str, bool, Dict[str, Any]]:
    """
    Build the final prompt by combining the original prompt with accumulated feedback.
    
    Args:
        orig_prompt: The original user prompt
        feedback_mode: The feedback mode to use (never, client_scoped, high_confidence, time_bounded, explicit, client_exact)
        client_id: The client ID for filtering feedback
        max_feedback_entries: Maximum number of recent feedback entries to include (applies to all modes)
        confidence_threshold: Minimum similarity threshold for fuzzy matching (0.0-1.0, default: 0.85)
        feedback_time_window_minutes: Time window for feedback in minutes (1-∞, default: 20 minutes)
        
    Returns:
        Tuple of (enhanced_prompt, has_feedback, feedback_info)
        where feedback_info contains all feedback details for display
    """
    # Initialize feedback variables
    all_feedback = []
    feedback_info = {
        "original_prompt": orig_prompt,
        "feedback_used": False,
        "feedback_entries": [],
        "enhanced_prompt": orig_prompt
    }
    
    # Only retrieve feedback if mode is not "never"
    if feedback_mode != "never":
        # Set time window if provided, otherwise use default (20 minutes)
        time_window = feedback_time_window_minutes if feedback_time_window_minutes is not None else 20
        
        # Get all recent feedback 
        if feedback_mode == "client_scoped":
            # Only use feedback from the same client
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, client_id=client_id, exact_client_match=True, 
                max_entries=max_feedback_entries, max_age_minutes=time_window)
        elif feedback_mode == "high_confidence":
            # Only use feedback with high confidence using configurable threshold
            threshold = confidence_threshold if confidence_threshold is not None else 0.8
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, min_confidence=threshold, 
                max_entries=max_feedback_entries, max_age_minutes=time_window)
        elif feedback_mode == "time_bounded":
            # Only use recent feedback (this mode specifically focuses on time window)
            # For this mode, we'll use a shorter default time window if not specified
            specific_time_window = feedback_time_window_minutes if feedback_time_window_minutes is not None else 10
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, max_age_minutes=specific_time_window, max_entries=max_feedback_entries)
        elif feedback_mode == "explicit":
            # Only use feedback with exact match
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, exact_match=True, 
                max_entries=max_feedback_entries, max_age_minutes=time_window)
        elif feedback_mode == "client_exact":
            # Only use feedback with exact match AND from the same client
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, client_id=client_id, exact_client_match=True, exact_match=True, 
                max_entries=max_feedback_entries, max_age_minutes=time_window)
        else:
            # Default behavior
            all_feedback = FeedbackManager.get_all_recent_feedback(
                orig_prompt, max_entries=max_feedback_entries, max_age_minutes=time_window)
    
    # If we have feedback, build the enhanced prompt
    if all_feedback:
        feedback_info["feedback_used"] = True
        feedback_info["feedback_entries"] = all_feedback
        
        # Build accumulated feedback text
        feedback_parts = []
        
        for fb in all_feedback:
            fb_text = fb["text"]
            corrected_query = fb["corrected_query"]
            fb_type = fb["type"]
            
            # Format feedback based on type
            if fb_type == "correction" and fb_text and corrected_query:
                # For corrections, include both explanation and corrected SQL
                feedback_parts.append(f"{fb_text}. Here's a corrected SQL query for reference: {corrected_query}")
            elif fb_type == "thumbs_up" and fb_text:
                # For thumbs_up with text, include the positive feedback
                feedback_parts.append(f"Previous feedback: {fb_text}")
            elif fb_text:
                # For other types with text
                feedback_parts.append(fb_text)
        
        # Combine all feedback parts into a single string
        if feedback_parts:
            all_feedback_text = "; ".join(feedback_parts)
            enhanced_prompt = f"{orig_prompt}, {all_feedback_text}"
            feedback_info["enhanced_prompt"] = enhanced_prompt
        else:
            enhanced_prompt = orig_prompt
    else:
        enhanced_prompt = orig_prompt
    
    # Return the enhanced prompt, whether feedback was used, and all feedback info
    return enhanced_prompt, bool(all_feedback), feedback_info
