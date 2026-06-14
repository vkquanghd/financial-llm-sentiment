import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from config import (
    BASE_MODEL,
    HF_TOKEN,
    NUM_LABELS,
    LABEL2ID,
    ID2LABEL,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRAD_ACCUMULATION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    OUTPUT_DIR,
    CKPT_DIR,
    USE_BF16,
    USE_FP16,
    HF_REPO_ID,
)
from data.prepare_data import prepare_dataset


# ─────────────────────────────────────────────
# 1. QLoRA quantization config
# ─────────────────────────────────────────────

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )


# ─────────────────────────────────────────────
# 2. Load base model
# ─────────────────────────────────────────────

def load_base_model(tokenizer, bnb_config: BitsAndBytesConfig):
    print(f"[Model] Loading {BASE_MODEL} with 4-bit quantization...")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    print(f"[Model] Base model loaded. Total parameters: {model.num_parameters():,}")

    return model


# ─────────────────────────────────────────────
# 3. Apply LoRA adapters
# ─────────────────────────────────────────────

def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[LoRA] Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    return model


# ─────────────────────────────────────────────
# 4. Metrics function
# ─────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1       = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": round(accuracy, 4),
        "f1":       round(f1, 4),
    }


# ─────────────────────────────────────────────
# 5. Build Trainer
# ─────────────────────────────────────────────

def build_trainer(model, tokenized_dataset, tokenizer) -> Trainer:
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,

        # Epochs & batch
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,

        # Optimizer
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        # Precision
        bf16=USE_BF16,
        fp16=USE_FP16,

        # Memory optimization
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        # Evaluation & saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


# ─────────────────────────────────────────────
# 6. Save model
# ─────────────────────────────────────────────

def save_model(model, tokenizer):
    print(f"[Save] Saving model to: {OUTPUT_DIR}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[Save] Model saved successfully.")

    if HF_REPO_ID and "your-username" not in HF_REPO_ID:
        print(f"[Save] Pushing to HuggingFace Hub: {HF_REPO_ID}")
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[Save] Pushed to Hub successfully.")


# ─────────────────────────────────────────────
# 7. Main training pipeline
# ─────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Financial Sentiment — QLoRA Fine-Tuning")
    print("=" * 60)

    print("\n[Step 1/5] Preparing dataset...")
    tokenized_dataset, tokenizer, rag_sentences = prepare_dataset()

    print("\n[Step 2/5] Loading base model...")
    bnb_config = build_bnb_config()
    model      = load_base_model(tokenizer, bnb_config)

    print("\n[Step 3/5] Applying LoRA adapters...")
    model = apply_lora(model)

    print("\n[Step 4/5] Starting training...")
    trainer = build_trainer(model, tokenized_dataset, tokenizer)
    trainer.train()

    print("\n[Step 5/5] Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(f"\n[Results] Test Accuracy : {test_results['eval_accuracy']:.4f}")
    print(f"[Results] Test F1 Macro : {test_results['eval_f1']:.4f}")
    print(f"[Results] Test Loss     : {test_results['eval_loss']:.4f}")

    save_model(model, tokenizer)

    print("\n[Done] Training complete.")
    return trainer, tokenizer, rag_sentences