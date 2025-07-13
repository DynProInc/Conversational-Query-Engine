FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY LLM\ Query\ Engine/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY LLM\ Query\ Engine/ .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
