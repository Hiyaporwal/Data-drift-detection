# ---------------------------------------------------------------------------
# Dockerfile for the Income Drift Detection API (app/main.py)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /code

# Install dependencies first (separate layer -> faster rebuilds when only
# your code changes, since Docker caches this step unless requirements.txt
# itself changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything the API needs at runtime:
#   app/    -> the FastAPI service itself
#   src/    -> drift_detectors.py, which app/main.py imports directly
#   models/ -> your trained .pkl pipelines (random_forest.pkl, logistic_regression.pkl)
#   data/   -> window_0.csv, used as the reference window for /check-drift
COPY app ./app
COPY src ./src
COPY models ./models
COPY data ./data

# Railway assigns a dynamic port via the $PORT environment variable -- we
# use shell form here so that variable actually gets substituted at
# container start (exec-form CMD would NOT expand env vars).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]