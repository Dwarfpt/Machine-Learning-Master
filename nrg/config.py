"""Central project configuration. Values can be overridden with environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_CSV = Path(os.getenv("NRG_RAW_CSV", DATA_DIR / "raw" / "Video_Games_Data.csv"))
DB_PATH = Path(os.getenv("NRG_DB_PATH", DATA_DIR / "games.db"))
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
OUTBOX_PATH = Path(os.getenv("NRG_OUTBOX_PATH", DATA_DIR / "outbox.jsonl"))  # fallback when n8n is not configured

# Target: a game "gets a negative review" if its critic score is below this threshold.
NEGATIVE_THRESHOLD = int(os.getenv("NRG_NEGATIVE_THRESHOLD", 60))
RANDOM_STATE = 42
TEST_YEAR = 2008  # time-based split: train on 2004-2007, test on 2008 releases

# Business assumptions (used by the decision module and the agent)
AVG_UNIT_PRICE_USD = float(os.getenv("NRG_AVG_UNIT_PRICE_USD", 40))
APPROVAL_COST_THRESHOLD_K = float(os.getenv("NRG_APPROVAL_COST_THRESHOLD_K", 50))  # k$; above -> human approval (QA 60k and up)

# AI agent: local LLM via llama-cpp-python (GGUF). Two modes:
#  * in-process: LLM_BASE_URL empty -> the model file is loaded with llama_cpp.Llama
#  * server:     LLM_BASE_URL=http://llm:8080 -> llama_cpp.server (OpenAI-compatible), used in Docker
LLM_MODEL_PATH = Path(os.getenv("LLM_MODEL_PATH", ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 8192))
LLM_THREADS = int(os.getenv("LLM_THREADS", 0)) or None  # None -> llama.cpp decides
LLM_NAME = os.getenv("LLM_NAME", LLM_MODEL_PATH.stem)

# Automation (n8n)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")  # e.g. http://localhost:5678/webhook/nrg-action
N8N_DECISION_WEBHOOK_URL = os.getenv("N8N_DECISION_WEBHOOK_URL", "")
API_PUBLIC_URL = os.getenv("NRG_API_PUBLIC_URL", "http://localhost:8000")
APPROVAL_SECRET = os.getenv("NRG_APPROVAL_SECRET", "change-me")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

for _d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
