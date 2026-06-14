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
# 1. Load raw dataset
# ─────────────────────────────────────────────

def load_raw_data() -> Dataset:
    """
    Load the Financial Phrasebank dataset from HuggingFace.

    Subset: sentences_allagree
      → Only sentences where ALL annotators agreed on the label
      → Highest quality labels (~4840 samples)

    Labels:
      0 = negative
      1 = neutral
      2 = positive

    Returns:
        Dataset with columns: ["sentence", "label"]
    """
    print("[Data] Loading Financial Phrasebank (sentences_allagree)...")

    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True,
    )

    # Dataset only has a "train" split — we split manually later
    data = dataset["train"]

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
    """
    Split the dataset into train (80%), val (10%), test (10%).

    Strategy:
      Step 1: Split into train 80% and temp 20%
      Step 2: Split temp into val 50% and test 50%
      → Final: train 80% / val 10% / test 10%

    Uses stratified split (stratify_by_column="label") to ensure
    each split has the same class distribution as the full dataset.

    Args:
        data: raw Dataset from load_raw_data()

    Returns:
        DatasetDict with keys: "train", "val", "test"
    """
    print("[Data] Splitting dataset: 80% train / 10% val / 10% test...")

    # Step 1: 80% train, 20% temp
    split_1 = data.train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label",
    )
    train = split_1["train"]
    temp  = split_1["test"]

    # Step 2: temp → 50% val, 50% test
    split_2 = temp.train_test_split(
        test_size=0.5,
        seed=SEED,
        stratify_by_column="label",
    )
    val  = split_2["train"]
    test = split_2["test"]

    dataset_dict = DatasetDict({
        "train": train,
        "val":   val,
        "test":  test,
    })

    print(f"[Data] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    return dataset_dict


# ─────────────────────────────────────────────
# 3. Build tokenizer
# ─────────────────────────────────────────────

def build_tokenizer() -> AutoTokenizer:
    """
    Load the LLaMA tokenizer and apply required fixes.

    Fixes applied:
      1. pad_token = eos_token
         LLaMA has no padding token by default → causes error during batching
         Setting pad_token to eos_token is the standard workaround

      2. padding_side = "right"
         Pad on the right so the real tokens come first
         Required for sequence classification (reads last non-pad token)

    Returns:
        AutoTokenizer configured for LLaMA classification
    """
    print(f"[Tokenizer] Loading from: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
    )

    # Fix 1: LLaMA has no pad token
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Fix 2: Pad on the right for classification
    tokenizer.padding_side = "right"

    print(f"[Tokenizer] Vocab size:     {tokenizer.vocab_size}")
    print(f"[Tokenizer] pad_token:      {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"[Tokenizer] padding_side:   {tokenizer.padding_side}")

    return tokenizer


# ─────────────────────────────────────────────
# 4. Tokenize function (applied per batch)
# ─────────────────────────────────────────────

def make_tokenize_fn(tokenizer: AutoTokenizer):

    def tokenize_fn(examples: dict) -> dict:
        tokenized = tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        # Keep the original integer label as classification target
        tokenized["labels"] = examples["label"]
        return tokenized

    return tokenize_fn


# ─────────────────────────────────────────────
# 5. Apply tokenization to full dataset
# ─────────────────────────────────────────────

def tokenize_dataset(dataset_dict: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Apply tokenization to all splits in parallel using dataset.map().

    Steps:
      1. Apply tokenize_fn to each split with batched=True (faster)
      2. Remove the "sentence" column (model does not need raw text)
      3. Set format to "torch" so __getitem__ returns tensors directly

    Args:
        dataset_dict: DatasetDict from split_data()
        tokenizer:    configured AutoTokenizer

    Returns:
        Tokenized DatasetDict ready for Trainer
    """
    print("[Data] Tokenizing dataset...")

    tokenize_fn = make_tokenize_fn(tokenizer)

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sentence"],   # raw text no longer needed
        desc="Tokenizing",
    )

    # Return tensors instead of lists when indexing
    tokenized.set_format("torch")

    print(f"[Data] Tokenization complete.")
    print(f"[Data] Sample keys: {list(tokenized['train'][0].keys())}")
    print(f"[Data] input_ids shape: {tokenized['train'][0]['input_ids'].shape}")

    return tokenized


# ─────────────────────────────────────────────
# 6. Get raw sentences for RAG indexing
# ─────────────────────────────────────────────

def get_sentences_for_rag(data: Dataset) -> list[str]:
    """
    Extract the raw sentence strings from the full dataset.
    These are passed to pinecone_utils.index_documents() to build
    the RAG knowledge base.

    We use ALL sentences (not just train) because the RAG knowledge
    base is separate from the training process — it is a retrieval
    corpus, not a training set.

    Args:
        data: raw Dataset from load_raw_data()

    Returns:
        list of sentence strings
    """
    sentences = data["sentence"]
    print(f"[RAG] Extracted {len(sentences)} sentences for Pinecone indexing.")
    return sentences


# ─────────────────────────────────────────────
# 7. Main entry point
# ─────────────────────────────────────────────

def prepare_dataset():
    """
    Full data preparation pipeline in one call.

    Steps:
      1. Load raw Financial Phrasebank dataset
      2. Split into train / val / test
      3. Build tokenizer
      4. Tokenize all splits
      5. Extract raw sentences for RAG

    Returns:
        tokenized_dataset : DatasetDict  → pass to model/train.py
        tokenizer         : AutoTokenizer → pass to model/train.py
        rag_sentences     : list[str]    → pass to rag/pinecone_utils.py
    """
    raw_data          = load_raw_data()
    dataset_dict      = split_data(raw_data)
    tokenizer         = build_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer)
    rag_sentences     = get_sentences_for_rag(raw_data)

    return tokenized_dataset, tokenizer, rag_sentences