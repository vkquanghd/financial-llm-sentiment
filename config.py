import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Environment Detection

IS_COLAB = "COLAB_GPU" in os.environ


# API Keys

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HF_TOKEN         = os.environ.get("HF_TOKEN")

# Paths


DRIVE_DIR  = "/content/drive/MyDrive/financial-llm"
OUTPUT_DIR = os.path.join(DRIVE_DIR, "saved_model") if IS_COLAB else "./saved_model"
LOG_DIR    = os.path.join(DRIVE_DIR, "logs")        if IS_COLAB else "./logs"
CKPT_DIR   = os.path.join(DRIVE_DIR, "checkpoints") if IS_COLAB else "./checkpoints"

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
NUM_LABELS = 3
MAX_LENGTH = 256

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {0: "negative", 1: "neutral", 2: "positive"}

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────

LORA_R          = 8
LORA_ALPHA      = 16           # scaling = alpha / r = 2
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj"]

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

LEARNING_RATE      = 2e-4
NUM_EPOCHS         = 3
BATCH_SIZE         = 16        # A100 40GB — lower to 4-8 if using T4
GRAD_ACCUMULATION  = 2         # effective batch = BATCH_SIZE * GRAD_ACCUMULATION = 32
WARMUP_RATIO       = 0.1
WEIGHT_DECAY       = 0.01
SEED               = 42

USE_BF16 = IS_COLAB and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16 and torch.cuda.is_available()

# ─────────────────────────────────────────────
# Pinecone
# ─────────────────────────────────────────────

INDEX_NAME  = "financial-sentiment"
DIMENSION   = 384              # matches all-MiniLM-L6-v2 output
METRIC      = "cosine"
CLOUD       = "aws"
REGION      = "us-east-1"
TOP_K       = 3

# ─────────────────────────────────────────────
# Embedding Model (for RAG)
# ─────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# HuggingFace Hub (optional — push model after training)
# ─────────────────────────────────────────────

HF_REPO_ID = "your-username/financial-llm-lora"   # change before pushing