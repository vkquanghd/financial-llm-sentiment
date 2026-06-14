import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from config import OUTPUT_DIR
from model.inference import load_trained_model, predict_sentiment, predict_with_rag
from rag.pinecone_utils import initialize_rag
from data.prepare_data import load_raw_data, get_sentences_for_rag

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# Load model & RAG (cached — only runs once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    model, tokenizer = load_trained_model(OUTPUT_DIR)

    raw_data      = load_raw_data()
    rag_sentences = get_sentences_for_rag(raw_data)
    index, embed_model = initialize_rag(rag_sentences)

    return model, tokenizer, index, embed_model


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("📈 Financial Sentiment Analyzer")
st.caption("LLaMA-2-7B fine-tuned with QLoRA · Pinecone RAG")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Enter financial text",
        placeholder="e.g. The company reported record profits this quarter.",
        height=120,
    )
    use_rag = st.toggle("Use Pinecone RAG", value=True,
                        help="Retrieve similar financial sentences to improve prediction")
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

LABEL_EMOJI = {"bearish": "🔴", "bullish": "🟢", "neutral": "🟡"}
LABEL_COLOR = {"bearish": "red", "bullish": "green", "neutral": "orange"}

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        model, tokenizer, index, embed_model = load_resources()

        with st.spinner("Analyzing..."):
            if use_rag:
                result = predict_with_rag(text_input, model, tokenizer, index, embed_model)
            else:
                result = predict_sentiment(text_input, model, tokenizer)

        label  = result["label"]
        scores = result["scores"]

        # ── Result ──
        with col2:
            st.subheader("Result")
            emoji = LABEL_EMOJI[label]
            color = LABEL_COLOR[label]
            st.markdown(
                f"<h2 style='color:{color};'>{emoji} {label.upper()}</h2>",
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {max(scores.values()):.1%}")

        # ── Score bars ──
        st.subheader("Confidence Scores")
        bar_cols = st.columns(3)
        for i, (lbl, score) in enumerate(scores.items()):
            with bar_cols[i]:
                st.metric(
                    label=f"{LABEL_EMOJI[lbl]} {lbl.capitalize()}",
                    value=f"{score:.1%}",
                )
                st.progress(score)

        # ── RAG context ──
        if use_rag and "retrieved_context" in result:
            st.subheader("📚 Retrieved Context (Pinecone RAG)")
            st.caption("Top similar financial sentences used as context:")
            for i, ctx in enumerate(result["retrieved_context"], 1):
                st.info(f"**[{i}]** {ctx}")

        st.divider()

# ─────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────

st.subheader("📋 Try these examples")
examples = [
    ("The company reported record profits this quarter.", True),
    ("Markets remained flat amid economic uncertainty.", True),
    ("Stock prices collapsed after the earnings miss.", True),
    ("Apple unveiled a new product line today.", False),
    ("The merger deal fell through at the last minute.", False),
]

for sentence, rag in examples:
    if st.button(f"{'🔍 ' if rag else ''}{sentence}", key=sentence):
        st.session_state["example_text"] = sentence
        st.session_state["example_rag"]  = rag
        st.rerun()

# Pre-fill from example click
if "example_text" in st.session_state:
    st.info(f"Selected: **{st.session_state['example_text']}**  |  RAG: {st.session_state['example_rag']}")