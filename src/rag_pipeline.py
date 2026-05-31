import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import requests
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RagPipeline:
    def __init__(
        self,
        pdf_paths: Optional[List[str]] = None,
        pdf_urls: Optional[List[str]] = None,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
    ):
        self.pdf_paths = pdf_paths or []
        self.pdf_urls = pdf_urls or []
        self.embed_model_name = embed_model_name
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="medbot_pdf_"))

        self.documents = self.load_documents(self.pdf_paths, self.pdf_urls)
        self.embed_model = self._load_embedding_model(self.embed_model_name)
        self.index, self.embeddings = self.build_faiss_index(self.documents)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        words = text.replace("\n", " ").split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text_chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
        return "\n\n".join(text_chunks)

    @staticmethod
    def download_pdf(url: str, dest_dir: str) -> str:
        dest_path = Path(dest_dir) / Path(url).name
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return str(dest_path)

    def load_documents(
        self,
        pdf_paths: List[str],
        pdf_urls: List[str],
    ) -> List[Dict[str, str]]:
        documents: List[Dict[str, str]] = []

        for path in pdf_paths:
            if not os.path.exists(path):
                continue
            text = self.extract_text_from_pdf(path)
            chunks = self.chunk_text(text)
            for idx, chunk in enumerate(chunks):
                documents.append(
                    {
                        "id": f"local-{Path(path).stem}-{idx}",
                        "source": path,
                        "text": chunk,
                    }
                )

        for url in pdf_urls:
            try:
                local_path = self.download_pdf(url, str(self.tmp_dir))
                text = self.extract_text_from_pdf(local_path)
                chunks = self.chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        {
                            "id": f"remote-{Path(local_path).stem}-{idx}",
                            "source": url,
                            "text": chunk,
                        }
                    )
            except Exception as exc:
                print(f"Warning: failed to load PDF from {url}: {exc}")

        return documents

    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name, device=device)

    def build_faiss_index(
        self, documents: List[Dict[str, str]]
    ) -> Tuple[faiss.IndexFlatIP, torch.Tensor]:
        texts = [doc["text"] for doc in documents]
        if not texts:
            index = faiss.IndexFlatIP(1)
            return index, torch.zeros((0, 1))

        embeddings = self.embed_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.embed_model.device,
        )
        embeddings = embeddings.detach()
        embeddings = embeddings.cpu()
        faiss.normalize_L2(embeddings.numpy())
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.numpy())
        return index, embeddings

    def query(self, question: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self.documents:
            return []
        query_embedding = self.embed_model.encode(
            question,
            convert_to_tensor=True,
            device=self.embed_model.device,
        )
        query_embedding = query_embedding.detach().cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.documents):
                continue
            results.append(
                {
                    "score": float(score),
                    "source": self.documents[index]["source"],
                    "text": self.documents[index]["text"],
                }
            )
        return results

    @staticmethod
    def build_context(results: List[Dict[str, str]]) -> str:
        context_parts = []
        for item in results:
            context_parts.append(f"Source: {item['source']}\n{item['text']}\n")
        return "\n---\n".join(context_parts)

    @staticmethod
    def generate_response_gemini(
        question: str,
        context: str,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> str:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass api_key.")

        endpoint = f"https://gemini.googleapis.com/v1beta2/models/{model}:generateText"
        prompt = (
            "You are an expert medical assistant. Use the context from the PDFs to answer the question precisely. "
            "If the answer is not contained in the context, say you cannot find enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        payload = {
            "prompt": {"text": prompt},
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "candidateCount": 1,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        text = ""
        if "candidates" in data and data["candidates"]:
            first = data["candidates"][0]
            text = first.get("output") or first.get("content", [{}])[0].get("text", "")
        return text.strip()

    def answer_question(
        self,
        question: str,
        api_key: Optional[str] = None,
        top_k: int = 3,
        model: str = "gemini-2.5-flash",
    ) -> Dict[str, str]:
        results = self.query(question, top_k=top_k)
        context = self.build_context(results)
        answer = self.generate_response_gemini(
            question=question,
            context=context,
            api_key=api_key,
            model=model,
        )
        return {
            "answer": answer,
            "context": context,
            "sources": [item["source"] for item in results],
        }
