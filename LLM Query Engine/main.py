"""
Main entry point for the Conversational Query Engine API when deployed to Render.
This file imports the FastAPI app from api_server.py and configures it to run with Gunicorn.
"""
import os
from api_server import app

# Load environment variables from .env file if running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except Exception as e:
    print(f"Note: Running without .env file, expecting environment variables to be set in the deployment platform: {str(e)}")

# Get the port from the environment variable
# Render sets the PORT environment variable for web services
port = int(os.environ.get("PORT", 10000))

# Print a message about environment variables
print("Starting server with environment variables from Render dashboard or local .env file")
print(f"Server will run on port {port}")
print("IMPORTANT: Using the PORT environment variable provided by Render")

# Make sure the app is accessible at the module level for Render
# This is critical for the deployment to work

# Add a simple health check endpoint for Render
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to bind to all interfaces
    # Explicitly log the binding information
    print(f"Starting server on 0.0.0.0:{port} with app from main.py")
    uvicorn.run(app, host="0.0.0.0", port=port)
