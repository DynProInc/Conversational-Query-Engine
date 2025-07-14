"""
Example of integrating the multi-client system with your existing API routes
This file shows how to enhance your API with client support without changing the core structure
"""

from fastapi import FastAPI, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Import your existing request and response models (just examples)
# from api_server import UnifiedQueryRequest, UnifiedQueryResponse

# Import the client manager
from .client_manager import client_manager
from .client_integration import with_client_context


# Example of extending your existing request model
class ClientAwareRequest(BaseModel):
    """Extend your existing request model with client_id"""
    client_id: Optional[str] = Field(None, description="Client identifier")
    # All your existing fields would be here


# Example of how to wrap existing route handlers with client context
@with_client_context
async def enhanced_query_handler(request, client_id: Optional[str] = None):
    """
    Example of enhancing an existing query handler with client context
    
    The @with_client_context decorator will:
    1. Extract client_id from request or parameters
    2. Set up the environment variables for the client
    3. Update the data_dictionary_path if needed
    4. Execute your existing handler logic
    5. Restore the original environment
    
    Your original handler code would run unchanged!
    """
    # Your existing handler code here
    pass


# Example of how to integrate with your FastAPI app
def add_client_routes(app: FastAPI):
    """
    Add client-related routes to your existing FastAPI app
    
    This keeps your existing routes unchanged while adding new functionality
    """
    
    @app.get("/clients", tags=["Clients"])
    async def list_clients():
        """List all available clients"""
        return {"clients": client_manager.list_active_clients()}
    
    @app.get("/clients/{client_id}", tags=["Clients"])
    async def get_client_info(client_id: str):
        """Get information about a specific client"""
        client_info = client_manager.get_client_info(client_id)
        if not client_info:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Remove sensitive information
        if "snowflake" in client_info:
            client_info["snowflake"] = {
                key: value if key not in ["password", "user"] else "***" 
                for key, value in client_info["snowflake"].items()
            }
        
        return {"client": client_info}


"""
To integrate this with your existing API server (api_server.py), you would:

1. Import the client integration module:
   from config.client_integration import with_client_context, get_client_data_dictionary_path
   from config.client_manager import client_manager

2. Update your request models to optionally include client_id:
   client_id: Optional[str] = Field(None, description="Client identifier")

3. Wrap your route handlers with the @with_client_context decorator:
   @app.post("/query/unified")
   @with_client_context
   async def unified_query_route(request: UnifiedQueryRequest):
       # Your existing code remains unchanged
       ...

4. Add client routes to your app:
   from config.api_integration_example import add_client_routes
   add_client_routes(app)

This approach keeps your existing code intact while enhancing it with multi-client support.
"""
