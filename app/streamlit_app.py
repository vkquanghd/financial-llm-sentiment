import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from config import OUTPUT_DIR
from model.inference import load_trained_model, predict_sentiment, predict_with_rag
from rag.pinecone_utils import initialize_rag
from data.prepare_data import load_raw_data, get_sentences_for_rag

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------
# Load model & RAG once (cached across reruns)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading model and RAG index...")
def load_resources():
    model, tokenizer   = load_trained_model(OUTPUT_DIR)
    raw_data           = load_raw_data()
    rag_sentences      = get_sentences_for_rag(raw_data)
    index, embed_model = initialize_rag(rag_sentences)
    return model, tokenizer, index, embed_model

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📈 Financial Sentiment Analyzer")
st.caption("LLaMA-2-7B fine-tuned with QLoRA + Pinecone RAG")
st.divider()

# --------------------------------------------------
# Input form
# --------------------------------------------------

EXAMPLES = [
    "Apple reported record-breaking quarterly earnings, beating analyst expectations by 15%.",
    "Silicon Valley Bank collapsed after a massive bank run, wiping out billions in deposits.",
    "The company will hold its annual shareholder meeting on Friday.",
    "The merger was completed ahead of schedule.",
    "Markets remained flat amid economic uncertainty.",
]

text_input = st.text_area(
    "Enter financial text",
    placeholder="e.g. The company reported record profits this quarter.",
    height=130,
)

col_rag, col_btn = st.columns([3, 1])
with col_rag:
    use_rag = st.checkbox("Use Pinecone RAG", value=True,
                          help="Retrieve similar financial sentences from Pinecone to improve prediction accuracy.")
with col_btn:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# Quick-fill examples
st.caption("Quick examples:")
ex_cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
    with col:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True, help=ex):
            text_input = ex

# --------------------------------------------------
# Inference
# --------------------------------------------------

LABEL_EMOJI = {"bearish": "🔴", "bullish": "🟢", "neutral": "🟡"}
LABEL_COLOR = {"bearish": "#e74c3c", "bullish": "#27ae60", "neutral": "#f39c12"}

if analyze_btn and text_input.strip():
    model, tokenizer, index, embed_model = load_resources()

    with st.spinner("Analyzing..."):
        if use_rag:
            result = predict_with_rag(text_input, model, tokenizer, index, embed_model)
        else:
            result = predict_sentiment(text_input, model, tokenizer)

    label  = result["label"]
    scores = result["scores"]
    emoji  = LABEL_EMOJI[label]
    color  = LABEL_COLOR[label]

    st.divider()

    # Result banner
    st.markdown(
        f"<div style='text-align:center; padding:20px; border-radius:10px; "
        f"background-color:{color}22; border:2px solid {color};'>"
        f"<h1 style='color:{color}; margin:0;'>{emoji} {label.upper()}</h1>"
        f"<p style='color:{color}; margin:5px 0 0;'>Confidence: {max(scores.values()):.1%}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    # Score bars
    bar_cols = st.columns(3)
    for i, (lbl, score) in enumerate(scores.items()):
        with bar_cols[i]:
            st.metric(label=f"{LABEL_EMOJI[lbl]} {lbl.capitalize()}", value=f"{score:.1%}")
            st.progress(score)

    # RAG context
    if use_rag and result.get("retrieved_context"):
        st.divider()
        st.subheader("📚 Retrieved Context (Pinecone RAG)")
        st.caption("Top-3 similar financial sentences retrieved as context:")
        for i, ctx in enumerate(result["retrieved_context"], 1):
            st.info(f"**[{i}]** {ctx}")

elif analyze_btn:
    st.warning("Please enter some text before clicking Analyze.")