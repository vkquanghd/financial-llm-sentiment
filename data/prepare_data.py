from collections import Counter
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

from config import (
    BASE_MODEL,
    HF_TOKEN,
    MAX_LENGTH,
    LABEL2ID,
    ID2LABEL,
    SEED,
)

# ─────────────────────────────────────────────
# Dataset info
# ─────────────────────────────────────────────
# zeroshot/twitter-financial-news-sentiment
# Columns: "text", "label"
# Labels:  0 = Bearish (negative)
#          1 = Bullish (positive)
#          2 = Neutral
# Size:    ~11,000 samples
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1. Load raw dataset
# ─────────────────────────────────────────────

def load_raw_data() -> Dataset:
    print("[Data] Loading twitter-financial-news-sentiment...")

    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    data    = dataset["train"]

    print(f"[Data] Total samples: {len(data)}")
    print(f"[Data] Columns: {data.column_names}")
    print(f"[Data] Label distribution:")

    label_counts = Counter(data["label"])
    for label_id, count in sorted(label_counts.items()):
        label_name = ID2LABEL[label_id]
        pct = count / len(data) * 100
        print(f"         {label_name:10s} (label={label_id}): {count} samples ({pct:.1f}%)")

    return data


# ─────────────────────────────────────────────
# 2. Split into train / val / test
# ─────────────────────────────────────────────

def split_data(data: Dataset) -> DatasetDict:
    print("[Data] Splitting: 80% train / 10% val / 10% test...")

    split_1 = data.train_test_split(
        test_size=0.2,
        seed=SEED,
    )
    train = split_1["train"]
    temp  = split_1["test"]

    split_2 = temp.train_test_split(
        test_size=0.5,
        seed=SEED,
    )
    val  = split_2["train"]
    test = split_2["test"]

    dataset_dict = DatasetDict({"train": train, "val": val, "test": test})

    print(f"[Data] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    return dataset_dict


# ─────────────────────────────────────────────
# 3. Build tokenizer
# ─────────────────────────────────────────────

def build_tokenizer() -> AutoTokenizer:
    print(f"[Tokenizer] Loading from: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
    )

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"[Tokenizer] Vocab size:   {tokenizer.vocab_size}")
    print(f"[Tokenizer] pad_token:    {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    return tokenizer


# ─────────────────────────────────────────────
# 4. Tokenize function
# ─────────────────────────────────────────────

def make_tokenize_fn(tokenizer: AutoTokenizer):
    def tokenize_fn(examples: dict) -> dict:
        tokenized = tokenizer(
            examples["text"],           # ← "text" column (not "sentence")
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    return tokenize_fn


# ─────────────────────────────────────────────
# 5. Apply tokenization
# ─────────────────────────────────────────────

def tokenize_dataset(dataset_dict: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    print("[Data] Tokenizing dataset...")

    tokenize_fn = make_tokenize_fn(tokenizer)

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],        # ← remove "text" column
        desc="Tokenizing",
    )

    tokenized.set_format("torch")

    print(f"[Data] Done. Sample keys: {list(tokenized['train'][0].keys())}")

    return tokenized


# ─────────────────────────────────────────────
# 6. Get raw sentences for RAG indexing
# ─────────────────────────────────────────────

def get_sentences_for_rag(data: Dataset) -> list[str]:
    sentences = data["text"]            # ← "text" column
    print(f"[RAG] Extracted {len(sentences)} sentences for Pinecone indexing.")
    return sentences


# ─────────────────────────────────────────────
# 7. Main entry point
# ─────────────────────────────────────────────

def prepare_dataset():
    raw_data          = load_raw_data()
    dataset_dict      = split_data(raw_data)
    tokenizer         = build_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer)
    rag_sentences     = get_sentences_for_rag(raw_data)

    return tokenized_dataset, tokenizer, rag_sentences