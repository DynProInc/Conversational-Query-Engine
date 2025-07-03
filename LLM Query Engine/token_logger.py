"""
Token Logger - Tracks token usage and costs for LLM API calls
"""
import os
import csv
import datetime
from typing import Dict, Any, Optional

class TokenLogger:
    """
    Logger to track token usage and costs for OpenAI API calls
    """
    # Pricing per 1000 tokens (as of July 2025)
    # Update these values if pricing changes
    PRICING = {
        # OpenAI models
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-32k': {'input': 0.06, 'output': 0.12},
        'gpt-4o': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
    }
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the token logger
        
        Args:
            log_file: Path to log file (defaults to token_usage.csv in current directory)
        """
        self.log_file = log_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_usage.csv')
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Create log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'model', 'query', 'prompt_tokens', 
                    'completion_tokens', 'total_tokens', 'input_cost', 
                    'output_cost', 'total_cost'
                ])
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """
        Calculate costs based on token usage and model
        
        Args:
            model: OpenAI model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Dictionary with input_cost, output_cost, and total_cost
        """
        # Find the closest matching model for pricing
        pricing_model = model
        if model not in self.PRICING:
            # Try to find a matching model family
            model_family = model.split('-')[0]  # e.g., extract 'gpt-4' from 'gpt-4-1106-preview'
            if f"{model_family}" in self.PRICING:
                pricing_model = f"{model_family}"
            else:
                # Default to gpt-4 pricing if no match is found
                pricing_model = 'gpt-4'
        
        # Calculate costs
        pricing = self.PRICING[pricing_model]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6)
        }
    
    def log_usage(self, model: str, query: str, usage: Dict[str, int]) -> Dict[str, Any]:
        """
        Log token usage and costs to CSV file
        
        Args:
            model: OpenAI model name
            query: User query
            usage: Dictionary with prompt_tokens, completion_tokens, total_tokens
            
        Returns:
            Dictionary with usage and cost information
        """
        # Extract token usage
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Calculate costs
        costs = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model,
            'query': query.replace('\n', ' ')[:100],  # Truncate long queries
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            **costs  # Include all cost fields
        }
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                log_entry['timestamp'],
                log_entry['model'],
                log_entry['query'],
                log_entry['prompt_tokens'],
                log_entry['completion_tokens'],
                log_entry['total_tokens'],
                log_entry['input_cost'],
                log_entry['output_cost'],
                log_entry['total_cost']
            ])
        
        return log_entry
