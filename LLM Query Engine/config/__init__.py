"""
Configuration package for multi-client support in the Conversational Query Engine
"""

from .client_manager import client_manager
from .client_integration import with_client_context, get_client_data_dictionary_path

__all__ = ['client_manager', 'with_client_context', 'get_client_data_dictionary_path']
