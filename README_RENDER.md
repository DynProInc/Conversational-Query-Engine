# Conversational Query Engine - Render Deployment Guide

This guide will help you deploy the Conversational Query Engine to Render.

## Prerequisites

1. A Render account (free tier available)
2. Your project code pushed to a Git repository (GitHub, GitLab, etc.)
3. API keys for the LLM services you want to use (OpenAI, Claude, Gemini)
4. Snowflake database credentials

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository contains all the necessary files:
- `api_server.py` (main FastAPI application)
- `requirements.txt` (Python dependencies)
- `render.yaml` (Render configuration)
- `runtime.txt` (Python version specification)
- `Procfile` (start command)

### 2. Set Up Environment Variables in Render

After connecting your repository to Render, you'll need to configure environment variables in the Render dashboard:

#### Required Environment Variables:

**Snowflake Configuration:**
```
CLIENT_MTS_SNOWFLAKE_USER=your_snowflake_user
CLIENT_MTS_SNOWFLAKE_PASSWORD=your_snowflake_password
CLIENT_MTS_SNOWFLAKE_ACCOUNT=your_snowflake_account
CLIENT_MTS_SNOWFLAKE_WAREHOUSE=your_warehouse
CLIENT_MTS_SNOWFLAKE_ROLE=your_role
CLIENT_MTS_SNOWFLAKE_DATABASE=your_database
CLIENT_MTS_SNOWFLAKE_SCHEMA=your_schema
```

**LLM API Keys:**
```
CLIENT_MTS_OPENAI_API_KEY=your_openai_api_key
CLIENT_MTS_ANTHROPIC_API_KEY=your_claude_api_key
CLIENT_MTS_GEMINI_API_KEY=your_gemini_api_key
```

**Optional Model Configuration:**
```
CLIENT_MTS_OPENAI_MODEL=gpt-4o
CLIENT_MTS_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
CLIENT_MTS_GEMINI_MODEL=models/gemini-2.5-flash
```

### 3. Deploy to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `conversational-query-engine` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Choose based on your needs (Starter is good for testing)

5. Add all the environment variables listed above
6. Click "Create Web Service"

### 4. Verify Deployment

Once deployed, your API will be available at:
`https://your-service-name.onrender.com`

Test the health endpoint:
```
GET https://your-service-name.onrender.com/health
```

## API Endpoints

Your deployed API will have the following main endpoints:

- `POST /query` - Generate SQL using OpenAI
- `POST /query/claude` - Generate SQL using Claude
- `POST /query/gemini` - Generate SQL using Gemini
- `POST /query/compare` - Compare all models
- `POST /query/unified` - Unified endpoint (defaults to Claude)
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /clients` - List available clients
- `GET /charts` - Chart viewer interface

## Security Considerations

1. **Never commit API keys to your repository**
2. Use Render's environment variable system for sensitive data
3. Consider using Render's private networking if connecting to private databases
4. Regularly rotate your API keys

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check that all dependencies are in `requirements.txt`
2. **Import Errors**: Ensure all Python files are in the correct directory structure
3. **Database Connection Issues**: Verify Snowflake credentials and network access
4. **API Key Errors**: Check that all required environment variables are set

### Logs:

View deployment logs in the Render dashboard under your service's "Logs" tab.

## Cost Optimization

- Use the free tier for development and testing
- Monitor usage to avoid unexpected charges
- Consider using smaller models for non-critical queries
- Implement proper error handling to avoid unnecessary API calls

## Support

If you encounter issues:
1. Check the Render documentation
2. Review the application logs
3. Verify all environment variables are correctly set
4. Test locally before deploying

## Local Development

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (create a .env file)
cp config/clients/env/mts.env .env

# Run the application
python api_server.py
```

The application will be available at `http://localhost:8000` 