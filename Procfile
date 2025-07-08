web: cd "LLM Query Engine" && gunicorn -w 4 -k uvicorn.workers.UvicornWorker wsgi:application --bind 0.0.0.0:$PORT
