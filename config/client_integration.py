"""
Client Integration Module
- Provides integration hooks for the existing codebase
- Extends functionality without changing API structure
"""

import os
from functools import wraps
from typing import Dict, Optional, Any, Callable
from .client_manager import client_manager

# Store the original environment variables
ORIGINAL_ENV = {
    'SNOWFLAKE_USER': os.environ.get('SNOWFLAKE_USER', ''),
    'SNOWFLAKE_PASSWORD': os.environ.get('SNOWFLAKE_PASSWORD', ''),
    'SNOWFLAKE_ACCOUNT': os.environ.get('SNOWFLAKE_ACCOUNT', ''),
    'SNOWFLAKE_WAREHOUSE': os.environ.get('SNOWFLAKE_WAREHOUSE', ''),
    'SNOWFLAKE_DATABASE': os.environ.get('SNOWFLAKE_DATABASE', ''),
    'SNOWFLAKE_SCHEMA': os.environ.get('SNOWFLAKE_SCHEMA', ''),
    'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
    'OPENAI_MODEL': os.environ.get('OPENAI_MODEL', ''),
    'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', ''),
    'ANTHROPIC_MODEL': os.environ.get('ANTHROPIC_MODEL', ''),
    'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', ''),
    'GEMINI_MODEL': os.environ.get('GEMINI_MODEL', '')
}

def restore_original_env():
    """Restore original environment variables"""
    for key, value in ORIGINAL_ENV.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

def set_client_context(client_id: str, model_provider: str = None) -> bool:
    """
    Set the client context by updating environment variables
    
    Args:
        client_id: The client identifier
        model_provider: The model provider to use (openai, anthropic, gemini)
        
    Returns:
        True if client context was set, False otherwise
    """
    if not client_id:
        # Use default configuration
        return False
    
    try:
        # Set Snowflake connection parameters
        sf_params = client_manager.get_snowflake_connection_params(client_id)
        os.environ['SNOWFLAKE_USER'] = sf_params['user']
        os.environ['SNOWFLAKE_PASSWORD'] = sf_params['password']
        os.environ['SNOWFLAKE_ACCOUNT'] = sf_params['account']
        os.environ['SNOWFLAKE_WAREHOUSE'] = sf_params['warehouse']
        
        if sf_params['database']:
            os.environ['SNOWFLAKE_DATABASE'] = sf_params['database']
        if sf_params['schema']:
            os.environ['SNOWFLAKE_SCHEMA'] = sf_params['schema']
        
        # Determine which providers to load based on model_provider parameter
        providers_to_load = []
        
        if model_provider:
            # Map model name to provider
            if model_provider.lower() in ['openai', 'gpt', 'gpt-4', 'gpt-3.5', 'gpt-4o'] or 'gpt' in model_provider.lower():
                providers_to_load = ['openai']
            elif model_provider.lower() in ['claude', 'anthropic', 'claude-3'] or 'claude' in model_provider.lower():
                providers_to_load = ['anthropic']
            elif model_provider.lower() in ['gemini', 'google', 'palm'] or 'gemini' in model_provider.lower():
                providers_to_load = ['gemini']
            elif model_provider.lower() in ['all', 'compare']:
                # For comparison, try to load all available providers but don't fail if some are missing
                providers_to_load = ['openai', 'anthropic', 'gemini']
            else:
                # Default to OpenAI if provider not recognized
                providers_to_load = ['openai']
        else:
            # If no provider specified, try to load OpenAI as default
            providers_to_load = ['openai']
        
        # Set LLM API keys and models for requested providers
        for provider in providers_to_load:
            try:
                llm_config = client_manager.get_llm_config(client_id, provider)
                if provider == 'openai':
                    os.environ['OPENAI_API_KEY'] = llm_config['api_key']
                    os.environ['OPENAI_MODEL'] = llm_config['model']
                elif provider == 'anthropic':
                    os.environ['ANTHROPIC_API_KEY'] = llm_config['api_key']
                    os.environ['ANTHROPIC_MODEL'] = llm_config['model']
                elif provider == 'gemini':
                    os.environ['GEMINI_API_KEY'] = llm_config['api_key']
                    os.environ['GEMINI_MODEL'] = llm_config['model']
            except Exception as provider_err:
                # Only raise exception if this is the primary requested provider
                if len(providers_to_load) == 1 or (model_provider and model_provider.lower() == provider):
                    print(f"Error setting {provider} API key: {str(provider_err)}")
                    raise
                else:
                    # For comparison endpoints, just log the error and continue
                    print(f"Note: {provider} API key not available for client '{client_id}'")
        
        return True
    except Exception as e:
        print(f"Error setting client context: {str(e)}")
        restore_original_env()
        return False

def get_client_data_dictionary_path(client_id: Optional[str], default_path: Optional[str] = None) -> Optional[str]:
    """
    Get the data dictionary path for a client or return the default
    
    Args:
        client_id: The client identifier
        default_path: Default path to use if client_id is None
        
    Returns:
        Path to the data dictionary
    """
    if client_id:
        try:
            client_dict_path = client_manager.get_data_dictionary_path(client_id)
            # Verify the dictionary file exists
            if client_dict_path and os.path.exists(client_dict_path):
                print(f"✅ Using client-specific dictionary for '{client_id}': {client_dict_path}")
                return client_dict_path
            else:
                print(f"⚠️ Client '{client_id}' dictionary path not found: {client_dict_path}")
        except ValueError as e:
            print(f"⚠️ Error getting dictionary path for client '{client_id}': {str(e)}")
    
    if default_path:
        print(f"ℹ️ Using default dictionary path: {default_path}")
    return default_path

def with_client_context(func):
    """
    Decorator to set client context for a function
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with client context
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract client_id from kwargs if present (don't pop it as it may be needed)
        client_id = kwargs.get('client_id', None)
        
        # Get the requested model (to only validate required API keys)
        requested_model = None
        if 'request' in kwargs:
            request = kwargs['request']
            if hasattr(request, 'model') and request.model:
                requested_model = request.model.lower()
                
        # If client_id in request body, extract it
        if not client_id and 'request' in kwargs:
            request = kwargs['request']
            if hasattr(request, 'client_id'):
                client_id = request.client_id
                print(f"Found client_id in request: {client_id}")
        
        original_data_dict_path = kwargs.get('data_dictionary_path')
        
        # Set client context if client_id is provided
        context_set = False
        try:
            if client_id:
                print(f"Setting client context for: {client_id}")
                # STRUCTURED ERROR HANDLING WITH CLEAR PRIORITY
                # Priority 1: Check if client exists
                # PRIORITY 1: CHECK CLIENT EXISTS
                client_info = client_manager.get_client_info(client_id)
                if client_info is None:
                    error_msg = f"Client '{client_id}' not found: Client ID not registered"
                    print(f"❌ Error: {error_msg}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=404, detail=error_msg)
                
                # PRIORITY 2: CHECK API KEYS
                if requested_model:
                    # Map model name to provider
                    provider = None
                    if requested_model in ['openai', 'gpt', 'gpt-4', 'gpt-3.5', 'gpt-4o'] or 'gpt' in requested_model:
                        provider = 'openai'
                    elif requested_model in ['claude', 'anthropic', 'claude-3'] or 'claude' in requested_model:
                        provider = 'anthropic'
                    elif requested_model in ['gemini', 'google', 'palm'] or 'gemini' in requested_model:
                        provider = 'gemini'
                    
                    # Check specific provider API key
                    if provider:
                        try:
                            client_manager.get_llm_config(client_id, provider)
                        except Exception as api_err:
                            error_msg = f"Client '{client_id}' {provider} API key not configured"
                            print(f"❌ Error: {error_msg}")
                            from fastapi import HTTPException
                            raise HTTPException(status_code=400, detail=error_msg)
                    
                    # Special handling for 'all' or 'compare' model request
                    elif requested_model == 'all' or requested_model == 'compare':
                        # Check for at least one configured provider
                        available_providers = []
                        for provider in ['openai', 'anthropic', 'gemini']:
                            try:
                                client_manager.get_llm_config(client_id, provider)
                                available_providers.append(provider)
                            except Exception:
                                pass  # Skip unavailable providers
                        
                        if not available_providers:
                            error_msg = f"Client '{client_id}' has no configured LLM providers"
                            print(f"❌ Error: {error_msg}")
                            from fastapi import HTTPException
                            raise HTTPException(status_code=400, detail=error_msg)
                
                # PRIORITY 3: CHECK SNOWFLAKE CREDENTIALS
                try:
                    client_manager.get_snowflake_connection_params(client_id)
                except Exception as sf_err:
                    error_msg = f"Client '{client_id}' Snowflake credentials error: {str(sf_err)}"
                    print(f"❌ Error: {error_msg}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # PRIORITY 4: CHECK DATA DICTIONARY
                try:
                    dict_path = client_manager.get_data_dictionary_path(client_id)
                    if not dict_path or not os.path.exists(dict_path):
                        error_msg = f"Client '{client_id}' data dictionary not found at {dict_path}"
                        print(f"❌ Error: {error_msg}")
                        from fastapi import HTTPException
                        raise HTTPException(status_code=400, detail=error_msg)
                except Exception as dict_err:
                    error_msg = f"Client '{client_id}' data dictionary error: {str(dict_err)}"
                    print(f"❌ Error: {error_msg}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # PRIORITY 5: SET CLIENT CONTEXT
                # Extract model from request if available
                model_to_use = requested_model if requested_model else None
                
                # Set client context with the appropriate model provider
                context_set = set_client_context(client_id, model_to_use)
                if not context_set:
                    error_msg = f"Client '{client_id}' context could not be set"
                    print(f"❌ Error: {error_msg}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=500, detail=error_msg)
                
                # Override data_dictionary_path if exists for the client
                try:
                    client_dict_path = get_client_data_dictionary_path(client_id, original_data_dict_path)
                    if client_dict_path:
                        print(f"Using data dictionary: {client_dict_path}")
                        # Only set data_dictionary_path if the function can accept it
                        # Check if the parameter exists in the function signature
                        import inspect
                        sig = inspect.signature(func)
                        if 'data_dictionary_path' in sig.parameters:
                            kwargs['data_dictionary_path'] = client_dict_path
                        else:
                            print(f"Function {func.__name__} does not accept data_dictionary_path parameter")
                except Exception as e:
                    print(f"❌ Error setting data dictionary path: {str(e)}")
                    # Continue with default path
                    pass
            
            # Call the original function
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"❌ Error in client context wrapper: {str(e)}")
            raise
        finally:
            # Restore original environment
            if context_set:
                print(f"Restoring original environment settings")
                restore_original_env()
                
                # Restore original data_dictionary_path if needed
                import inspect
                sig = inspect.signature(func)
                if original_data_dict_path and 'data_dictionary_path' in sig.parameters:
                    kwargs['data_dictionary_path'] = original_data_dict_path
    
    return wrapper
