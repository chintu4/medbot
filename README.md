---
title: MedBot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.15.2"
python_version: "3.11"
app_file: app.py
pinned: false
---

# medbot

A retrieval-augmented generation (RAG) demo project designed for a Hugging Face Spaces-friendly Gradio deployment.

## Hugging Face Spaces Ready
This repository is structured for a Space that uses a Gradio app. To make the Space reproducible, the project includes:

- `requirements.txt` with a pinned Gradio version.
- `README.md` with installation, usage, and deployment notes.
- `ui/gradio_dashboard.py` as the intended Gradio app entrypoint.

## Requirements
Recommended environment:

- Python 3.10 or 3.11
- `gradio==6.15.2`
- `numpy<2`
- `requests`
- `PyPDF2==3.0.1`
- `sentence-transformers==5.5.1`
- `faiss-cpu==1.8.0`
- `torch>=2.0.0`
- `python-dotenv` for loading local `.env` files when present

Install dependencies:

```bash
pip install -r requirements.txt
```

> If `faiss` is not available in your local environment, the pipeline will still work with a fallback search method, but installing `faiss-cpu` gives better retrieval performance.

## Local Run
If you add your Gradio interface to `ui/gradio_dashboard.py`, run locally with:

```bash
python ui/gradio_dashboard.py
```

For a basic Gradio app, this file should create a `gr.Interface` or `gr.Blocks` instance and call `launch()`.

## Standalone Retrieval Script
A simple root-level script is available for running retrieval directly from a pickle store:

```bash
python main.py --query "what is diabetes" --top-k 3
```

By default, the script loads `rag_store.pkl` from the repository root. To specify a different store file:

```bash
python main.py --path path/to/rag_store.pkl --query "what is diabetes"
```

## Deployment Notes for Hugging Face Spaces
To deploy on Hugging Face Spaces:

1. Push this repo to a Hugging Face Space.
2. Ensure `requirements.txt` is present.
3. Confirm `app.py` or your Gradio entrypoint is referenced in the Space settings if needed.
4. Use a compatible runtime: CPU is fine for light demos, GPU may be required for larger models.

### Recommended Space settings

- Runtime: `cpu` for simple demos
- Python version: `3.10` or `3.11`
- Install command: `pip install -r requirements.txt`

## What to Add
This repo currently has a working app shell at `ui/gradio_dashboard.py`.

The app is already configured to load these medical PDF links:

- `https://bookchapter.org/kitaplar/Medical%20Diagnosis%20and%20Treatment%20Methods%20in%20Basic%20Medical%20Sciences.pdf`
- `https://applications.emro.who.int/dsaf/dsa236.pdf`
- `https://www.dhhs.nh.gov/sites/g/files/ehbemt476/files/documents/2021-11/disease-handbook-complete.pdf`
- `https://aiimsrishikesh.edu.in/documents/standard-treatment-guidelines.pdf`
- `https://www.fao.org/4/y1238e/y1238e06.pdf`

To use the demo, configure the Gemini API key on the backend, then run the app locally:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
python ui/gradio_dashboard.py
```

Supported backend key names: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `GOOGLE_GENAI_API_KEY`.

Then open the Gradio link in your browser, ask a question, and the backend will retrieve PDF passages before sending them to Gemini.

## License and Attribution
Add a license file and any model attribution required by your dataset or API provider.

## Notes
If you need a direct model or API integration, add the dependency to `requirements.txt` and document it here.
