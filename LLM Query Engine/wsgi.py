"""
WSGI entry point for the Conversational Query Engine API when deployed to Render.
This file imports the FastAPI app from api_server.py and makes it available for WSGI servers.
"""
from api_server import app

# This is the WSGI application object that Render will use
application = app
