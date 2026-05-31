import os

from dotenv import load_dotenv
from ui.gradio_dashboard import build_demo

load_dotenv()

if __name__ == "__main__":
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("GEMINI_API_KEY is set:", gemini_key[:10] + "..." if len(gemini_key) > 10 else gemini_key)
    else:
        print("GEMINI_API_KEY is not set. Please add it to your .env or environment.")

    demo = build_demo()
    demo.launch()
