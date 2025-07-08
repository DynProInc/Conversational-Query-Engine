# Conversational Query Engine

A FastAPI backend application that converts natural language questions to SQL using various LLM providers (OpenAI, Claude, Gemini), executes the SQL against a Snowflake database, and returns the results.

## Deployment to Render

### Prerequisites

1. A [Render](https://render.com/) account
2. Access to the following API keys:
   - OpenAI API key
   - Anthropic API key (for Claude)
   - Google API key (for Gemini)
3. Snowflake database credentials

### Deployment Steps

1. **Fork or clone this repository**

2. **Create a new Web Service on Render**
   - Go to the Render dashboard and click "New" > "Web Service"
   - Connect your GitHub repository
   - Use the following settings:
     - **Name**: conversational-query-engine (or your preferred name)
     - **Environment**: Python 3.11.0
     - **Region**: Choose the region closest to your users
     - **Branch**: main (or your default branch)
     - **Build Command**: `pip install -r "LLM Query Engine/requirements.txt"`
     - **Start Command**: `cd "LLM Query Engine" && python main.py`

3. **Configure Environment Variables (IMPORTANT)**
   - In the Render dashboard, go to your web service
   - Click on "Environment" tab
   - Add all the environment variables listed in `.env.example`:
     - `SNOWFLAKE_USER`
     - `SNOWFLAKE_PASSWORD`
     - `SNOWFLAKE_ACCOUNT`
     - `SNOWFLAKE_WAREHOUSE`
     - `SNOWFLAKE_DATABASE`
     - `SNOWFLAKE_SCHEMA`
     - `OPENAI_API_KEY`
     - `OPENAI_MODEL`
     - `ANTHROPIC_API_KEY`
     - `CLAUDE_MODEL`
     - `GOOGLE_API_KEY`
     - `GEMINI_MODEL`
   - **NEVER hardcode these values in your source code**

4. **Configure Resource Allocation**
   - Based on previous memory usage patterns, this application may require more than the default memory allocation
   - In the Render dashboard, under your web service settings:
     - Set the instance type to at least 1GB RAM
     - This will avoid memory-related issues that were previously encountered

5. **Deploy the Service**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

6. **Access Your API**
   - Once deployed, you can access your API at the URL provided by Render
   - The API documentation will be available at `https://your-render-url.onrender.com/docs`
   - Monitor the logs in the Render dashboard to ensure the application is running correctly

### API Endpoints

- `/query` - OpenAI-based SQL generation and execution
- `/query/claude` - Claude-based SQL generation and execution
- `/query/gemini` - Gemini-based SQL generation and execution
- `/query/compare` - Compares results from all three models
- `/health` - Health check endpoint
- `/models` - Lists available models
- `/query/unified` - Unified endpoint that routes to the appropriate model

## Local Development

1. Clone the repository
2. Create a `.env` file based on `.env.example` and fill in your credentials
3. Install dependencies: `pip install -r "LLM Query Engine/requirements.txt"`
4. Run the server: `cd "LLM Query Engine" && python api_server.py`
5. Access the API at `http://localhost:8000`
