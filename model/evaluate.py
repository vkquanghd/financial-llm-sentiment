import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from config import ID2LABEL, BATCH_SIZE, OUTPUT_DIR


# ─────────────────────────────────────────────
# 1. Full evaluation on test set
# ─────────────────────────────────────────────

def evaluate_on_testset(model, tokenizer, test_dataset) -> dict:
    model.eval()

    collator    = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collator)

    all_preds  = []
    all_labels = []

    for batch in dataloader:
        labels = batch.pop("labels").numpy()
        inputs = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro":  round(f1_score(all_labels, all_preds, average="macro"), 4),
        "precision": round(precision_score(all_labels, all_preds, average="macro"), 4),
        "recall":    round(recall_score(all_labels, all_preds, average="macro"), 4),
    }

    print("\n[Evaluation] Test Set Results:")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n[Evaluation] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(ID2LABEL.values())))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

    return results


# ─────────────────────────────────────────────
# 2. Confusion matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path: str = f"{OUTPUT_DIR}/confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    class_names = list(ID2LABEL.values())

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"[Evaluate] Confusion matrix saved to: {save_path}")


# ─────────────────────────────────────────────
# 3. Attention heatmap
# ─────────────────────────────────────────────

def get_token_scores(model, tokenizer, sentence: str):
    # 4-bit quantized + LoRA models do not support output_attentions.
    # Use gradient-based token importance instead:
    # score(token_i) = ||∂L/∂embedding_i|| (gradient norm of each token embedding)
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        padding=False,
    ).to(model.device)

    model.eval()

    # Get embeddings with gradient tracking
    embed_layer = model.base_model.model.model.embed_tokens
    input_ids   = inputs["input_ids"]
    embeds      = embed_layer(input_ids)          # [1, seq, hidden]
    embeds.retain_grad()

    outputs = model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"])
    logits  = outputs.logits                      # [1, num_labels]

    # Backprop through the predicted class score
    pred_class = logits.argmax(dim=-1)
    logits[0, pred_class].backward()

    # Gradient norm per token → importance score
    grad        = embeds.grad[0]                  # [seq, hidden]
    scores      = grad.norm(dim=-1).cpu().detach().numpy()   # [seq]
    scores      = scores / (scores.max() + 1e-8)  # normalize to [0, 1]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return scores, tokens


def plot_attention_heatmap(
    sentence: str,
    model,
    tokenizer,
    save_path: str = f"{OUTPUT_DIR}/attention_heatmap.png",
):
    # Note: renamed internally to token importance (4-bit models don't support attention output)
    scores, tokens = get_token_scores(model, tokenizer, sentence)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), 2.5))

    # Bar chart: one bar per token, height = importance score
    bars = ax.bar(range(len(tokens)), scores, color="steelblue")

    # Color the most important token
    max_idx = scores.argmax()
    bars[max_idx].set_color("tomato")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Gradient-based Importance")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Token Importance\n\"{sentence[:70]}\"")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"[Evaluate] Token importance chart saved to: {save_path}")


# ─────────────────────────────────────────────
# 4. Benchmark table
# ─────────────────────────────────────────────

def build_benchmark_table(results: dict, save_path: str = f"{OUTPUT_DIR}/benchmark.csv"):
    # results format:
    # {
    #   "BERT-base":   {"params_M": 110, "time_min": 15, "memory_gb": 2.1, "f1": 0.87, "accuracy": 0.88},
    #   "LLaMA-LoRA":  {"params_M": 7000, "time_min": 60, "memory_gb": 8.0, "f1": 0.91, "accuracy": 0.92},
    #   "LLaMA-QLoRA": {"params_M": 7000, "time_min": 45, "memory_gb": 5.2, "f1": 0.90, "accuracy": 0.91},
    # }

    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "Params (M)", "Train Time (min)", "Memory (GB)", "F1 Macro", "Accuracy"]

    print("\n[Benchmark] Model Comparison:")
    print(df.to_string(index=False))

    df.to_csv(save_path, index=False)
    print(f"[Benchmark] Table saved to: {save_path}")

    return df