FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000 8501
CMD ["sh", "-c", "test -f artifacts/model.joblib || python -m nrg.train; uvicorn api.main:app --host 0.0.0.0 --port 8000"]
