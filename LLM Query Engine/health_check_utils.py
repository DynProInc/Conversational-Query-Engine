import os
import openai
import snowflake.connector
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_openai_health():
    """Check if OpenAI API is reachable by listing models (quick, low-cost call)."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OpenAI API key not configured")
            return False, "OpenAI API connection error: API key not configured"
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True, "Connected to OpenAI API"
    except Exception as e:
        print(f"[ERROR] OpenAI API error: {str(e)}")
        return False, "OpenAI API connection error"

def check_claude_health():
    """Check if Anthropic Claude API is reachable."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[ERROR] Claude API key not configured")
            return False, "Claude API connection error: API key not configured"
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return True, "Connected to Claude API"
    except Exception as e:
        print(f"[ERROR] Claude API error: {str(e)}")
        return False, "Claude API connection error"

def check_gemini_health():
    """Check if Google Gemini API is reachable."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] Gemini API key not configured")
            return False, "Gemini API connection error: API key not configured"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        response = model.generate_content("Hello")
        return True, "Connected to Gemini API"
    except Exception as e:
        print(f"[ERROR] Gemini API error: {str(e)}")
        return False, "Gemini API connection error"

def check_snowflake_health():
    """Check if Snowflake is reachable by connecting and running a simple query."""
    try:
        user = os.environ.get('SNOWFLAKE_USER')
        password = os.environ.get('SNOWFLAKE_PASSWORD')
        account = os.environ.get('SNOWFLAKE_ACCOUNT')
        warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
        database = os.environ.get('SNOWFLAKE_DATABASE')
        schema = os.environ.get('SNOWFLAKE_SCHEMA')
        if not all([user, password, account, warehouse, database, schema]):
            print("[ERROR] One or more Snowflake configuration variables missing")
            return False, "Snowflake connection error: Configuration missing"
        ctx = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        cs = ctx.cursor()
        cs.execute("SELECT 1")
        cs.fetchone()
        cs.close()
        ctx.close()
        return True, "Connected to Snowflake"
    except Exception as e:
        print(f"[ERROR] Snowflake error: {str(e)}")
        return False, "Snowflake connection error"
