import gradio as gr

from config import OUTPUT_DIR
from model.inference import load_trained_model, predict_sentiment, predict_with_rag
from rag.pinecone_utils import initialize_rag
from data.prepare_data import load_raw_data, get_sentences_for_rag


# ─────────────────────────────────────────────
# Global state — loaded once at startup
# ─────────────────────────────────────────────

model       = None
tokenizer   = None
index       = None
embed_model = None


def startup():
    global model, tokenizer, index, embed_model

    print("[App] Loading model...")
    model, tokenizer = load_trained_model(OUTPUT_DIR)

    print("[App] Initializing Pinecone RAG...")
    raw_data      = load_raw_data()
    rag_sentences = get_sentences_for_rag(raw_data)
    index, embed_model = initialize_rag(rag_sentences)

    print("[App] Ready.")


# ─────────────────────────────────────────────
# Prediction function (called by Gradio)
# ─────────────────────────────────────────────

def predict_ui(text: str, use_rag: bool):
    if not text.strip():
        return {}, "Please enter a sentence."

    if use_rag:
        result = predict_with_rag(text, model, tokenizer, index, embed_model)
        context_display = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(result["retrieved_context"])
        )
    else:
        result = predict_sentiment(text, model, tokenizer)
        context_display = "RAG is disabled. No context retrieved."

    # gr.Label expects {"label": confidence} format
    label_scores = {
        f"{k.upper()}": v
        for k, v in result["scores"].items()
    }

    return label_scores, context_display


# ─────────────────────────────────────────────
# Gradio interface
# ─────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Financial Sentiment Analyzer") as app:

        gr.Markdown(
            """
            # Financial Sentiment Analyzer
            Fine-tuned LLaMA-2-7B with QLoRA + Pinecone RAG
            Classify financial text as **Positive**, **Neutral**, or **Negative**.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Financial Text",
                    placeholder="e.g. The company reported record profits this quarter.",
                    lines=3,
                )
                use_rag = gr.Checkbox(
                    label="Use RAG (retrieve relevant context from Pinecone)",
                    value=True,
                )
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Sentiment",
                    num_top_classes=3,
                )

        context_output = gr.Textbox(
            label="Retrieved Context (RAG)",
            lines=5,
            interactive=False,
        )

        # Examples
        gr.Examples(
            examples=[
                ["The company reported record profits this quarter.", True],
                ["Markets remained flat amid economic uncertainty.", True],
                ["Stock prices collapsed after the earnings miss.", True],
                ["Apple unveiled a new product line today.", False],
                ["The merger deal fell through at the last minute.", False],
            ],
            inputs=[text_input, use_rag],
        )

        submit_btn.click(
            fn=predict_ui,
            inputs=[text_input, use_rag],
            outputs=[label_output, context_output],
        )

        # Also trigger on Enter key
        text_input.submit(
            fn=predict_ui,
            inputs=[text_input, use_rag],
            outputs=[label_output, context_output],
        )

    return app


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    startup()
    app = build_app()
    app.launch(share=True)      # share=True → public link for Colab