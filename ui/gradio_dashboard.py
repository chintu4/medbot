import os

import gradio as gr

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


def ask_medbot(question: str, gemini_api_key: str, model_name: str, top_k: int):
    if not question:
        return "Please enter a question.", ""
    if not gemini_api_key:
        return "Gemini API key is required.", ""

    result = pipeline.answer_question(
        question=question,
        api_key=gemini_api_key,
        model=model_name,
        top_k=top_k,
    )

    answer = result.get("answer", "No answer returned.")
    context = result.get("context", "No context available.")
    return answer, context


def build_demo():
    with gr.Blocks(title="MedBot RAG with Gemini") as demo:
        gr.Markdown(
            """
            # MedBot RAG Demo
            Ask questions against a set of PDF documents and remote PDF links.

            - Enter a question.
            - Provide your Gemini API key.
            - The system retrieves relevant PDF passages and sends them to Gemini for completion.
            """
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the loaded PDFs...",
                lines=2,
            )
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Gemini API key here",
                type="password",
            )

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
            label="Retrieval top-k passages",
        )

        answer_output = gr.Textbox(label="Gemini Answer", lines=8)
        context_output = gr.Textbox(label="Retrieved Context", lines=12)

        submit_btn = gr.Button("Ask MedBot")
        submit_btn.click(
            ask_medbot,
            inputs=[question_input, api_key_input, model_choice, top_k],
            outputs=[answer_output, context_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
