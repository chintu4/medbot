import os

from dotenv import load_dotenv
import gradio as gr

load_dotenv()
print("GEMINI_API_KEY loaded:", bool(os.getenv("GEMINI_API_KEY")))

from src.rag_pipeline import RagPipeline

PDF_URLS = [
    "https://bookchapter.org/kitaplar/Medical%20Diagnosis%20and%20Treatment%20Methods%20in%20Basic%20Medical%20Sciences.pdf",
    "https://applications.emro.who.int/dsaf/dsa236.pdf",
    "https://www.dhhs.nh.gov/sites/g/files/ehbemt476/files/documents/2021-11/disease-handbook-complete.pdf",
    "https://www.dhhs.nh.gov/sites/g/files/ehbemt476/files/documents/2021-11/disease-handbook-complete.pdf",
    "https://aiimsrishikesh.edu.in/documents/standard-treatment-guidelines.pdf",
    "https://www.fao.org/4/y1238e/y1238e06.pdf",
]

PDF_PATHS = [
    # Add local PDF paths here if needed. Example:
    # "./docs/sample.pdf"
]

pipeline = RagPipeline(pdf_paths=PDF_PATHS, pdf_urls=PDF_URLS)


def format_sources(sources):
    if not sources:
        return "No sources retrieved yet."

    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)

    items = "".join(f"<li>{source}</li>" for source in unique_sources)
    return f"<ul class='source-list'>{items}</ul>"


def ask_medbot(question: str, model_name: str, top_k: int):
    if not question:
        return "Please enter a question.", "", ""

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not gemini_api_key:
        return (
            "Server-side Gemini API key is not configured. Set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY.",
            "",
            "",
        )

    result = pipeline.answer_question(
        question=question,
        api_key=gemini_api_key,
        model=model_name,
        top_k=top_k,
    )

    answer = result.get("answer", "No answer returned.")
    context = result.get("context", "No context available.")
    sources = result.get("sources", [])
    source_markup = format_sources(sources)
    return answer, context, source_markup


def clear_outputs():
    return "", "", ""


def build_demo():
    css = """
    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.16), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    }
    .hero {
        padding: 2rem 1.5rem 1rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 64, 175, 0.9));
        color: white;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.2rem);
        margin: 0 0 0.35rem;
        line-height: 1.05;
    }
    .hero p {
        margin: 0;
        max-width: 70ch;
        color: rgba(255, 255, 255, 0.84);
        font-size: 1rem;
    }
    .pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.16);
        font-size: 0.92rem;
        backdrop-filter: blur(14px);
    }
    .card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(18px);
    }
    .section-label {
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.55rem;
    }
    .source-list {
        margin: 0;
        padding-left: 1.2rem;
        line-height: 1.65;
    }
    .hint {
        color: #475569;
        font-size: 0.94rem;
    }
    """

    with gr.Blocks(title="MedBot RAG with Gemini") as demo:
        gr.HTML(
            """
            <section class="hero">
              <h1>MedBot RAG</h1>
              <p>
                A focused medical research assistant that retrieves passages from curated PDF sources
                and answers in a clean, guided workflow powered by Gemini.
              </p>
              <div class="pill-row">
                <span class="pill">PDF retrieval</span>
                <span class="pill">Gemini backend</span>
                <span class="pill">Source-aware answers</span>
              </div>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.HTML("<div class='card'><div class='section-label'>Ask</div></div>")

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask something about dosage, diagnosis, treatment guidance, or a specific topic in the loaded PDFs...",
                    lines=4,
                    show_label=True,
                )

                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=["gemini-2.5-flash", "gemini-2.5-lite", "gemini-3.0-flash"],
                        value="gemini-2.5-flash",
                        label="Gemini model",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=3,
                        label="Retrieved passages",
                    )

                with gr.Row():
                    submit_btn = gr.Button("Generate Answer", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")


                
            with gr.Column(scale=4):
                gr.HTML("<div class='card'><div class='section-label'>Answer</div></div>")
                answer_output = gr.Markdown(value="Your answer will appear here.")

                with gr.Accordion("Retrieved context", open=False):
                    context_output = gr.Textbox(
                        label="Context",
                        lines=12,
                        interactive=False,
                    )

                with gr.Accordion("Sources used", open=False):
                    sources_output = gr.HTML(value="No sources retrieved yet.")

        submit_btn.click(
            ask_medbot,
            inputs=[question_input, model_choice, top_k],
            outputs=[answer_output, context_output, sources_output],
        )
        clear_btn.click(
            clear_outputs,
            inputs=None,
            outputs=[answer_output, context_output, sources_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(theme=gr.themes.Soft(), css=css)
