import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import (
    BASE_MODEL,
    HF_TOKEN,
    MAX_LENGTH,
    ID2LABEL,
    OUTPUT_DIR,
    USE_BF16,
    USE_FP16,
)
from rag.retriever import get_rag_prompt, build_plain_prompt


# ─────────────────────────────────────────────
# 1. Load trained model
# ─────────────────────────────────────────────

def load_trained_model(model_path: str = OUTPUT_DIR):
    print(f"[Inference] Loading model from: {model_path}")

    # 4-bit config — same as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model with quantization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Merge LoRA adapters on top of base model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    print("[Inference] Model ready.")
    return model, tokenizer


# ─────────────────────────────────────────────
# 2. Predict (no RAG)
# ─────────────────────────────────────────────

def predict_sentiment(text: str, model, tokenizer) -> dict:
    prompt = build_plain_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits → probabilities
    probs = F.softmax(outputs.logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()

    return {
        "label":  ID2LABEL[pred_idx],
        "scores": {
            ID2LABEL[i]: round(probs[i].item(), 4)
            for i in range(len(ID2LABEL))
        },
    }


# ─────────────────────────────────────────────
# 3. Predict with RAG
# ─────────────────────────────────────────────

def predict_with_rag(text: str, model, tokenizer, index, embed_model) -> dict:
    # Retrieve context + build augmented prompt
    prompt, context_list = get_rag_prompt(text, index, embed_model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()

    return {
        "label":             ID2LABEL[pred_idx],
        "scores":            {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))},
        "retrieved_context": context_list,
    }