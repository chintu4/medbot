import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    _HAS_FAISS = False


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GEMINI_TOKEN",
)


def _load_environment() -> None:
    if load_dotenv is None:
        return
    load_dotenv(override=False)


def _get_gemini_api_key(api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key

    _load_environment()
    for env_key in GEMINI_ENV_KEYS:
        value = os.getenv(env_key)
        if value:
            return value
    return None


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
    ) -> Tuple[Optional["faiss.IndexFlatIP"], torch.Tensor]:
        texts = [doc["text"] for doc in documents]
        if not texts:
            if _HAS_FAISS:
                index = faiss.IndexFlatIP(1)
            else:
                index = None
            return index, torch.zeros((0, 1))

        embeddings = self.embed_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.embed_model.device,
        )
        embeddings = embeddings.detach().cpu()

        if _HAS_FAISS:
            faiss.normalize_L2(embeddings.numpy())
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.numpy())
            return index, embeddings

        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return None, embeddings

    def query(self, question: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self.documents:
            return []
        query_embedding = self.embed_model.encode(
            question,
            convert_to_tensor=True,
            device=self.embed_model.device,
        )
        query_embedding = query_embedding.detach().cpu()

        if _HAS_FAISS and self.index is not None:
            query_array = query_embedding.numpy().reshape(1, -1)
            faiss.normalize_L2(query_array)
            distances, indices = self.index.search(query_array, top_k)
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

        query_embedding = torch.nn.functional.normalize(query_embedding, dim=1)
        scores = torch.matmul(self.embeddings, query_embedding.T).squeeze(1)
        top_indices = torch.topk(scores, min(top_k, len(scores))).indices.tolist()
        results = []
        for idx in top_indices:
            results.append(
                {
                    "score": float(scores[idx].item()),
                    "source": self.documents[idx]["source"],
                    "text": self.documents[idx]["text"],
                }
            )
        return results
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
    def _build_gemini_headers(api_key: str) -> Dict[str, str]:
        if api_key.startswith("AIza"):
            return {"Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_gemini_endpoint(api_key: str, model: str) -> str:
        if api_key.startswith("AIza"):
            return f"https://generativelanguage.googleapis.com/v1/models/{model}:generateText?key={api_key}"
        return f"https://generativelanguage.googleapis.com/v1/models/{model}:generateText"

    @staticmethod
    def _parse_gemini_response(data: Dict) -> str:
        if not isinstance(data, dict):
            return ""

        # Modern Gemini text generation response structure.
        if "candidates" in data and data["candidates"]:
            first = data["candidates"][0]
            if isinstance(first, dict):
                if "output" in first and isinstance(first["output"], str):
                    return first["output"].strip()
                if "content" in first:
                    content = first["content"]
                    if isinstance(content, dict) and "parts" in content:
                        return "\n".join(
                            part.get("text", "") for part in content.get("parts", [])
                        ).strip()
                    if isinstance(content, list):
                        return "\n".join(str(item) for item in content).strip()

        if "output" in data:
            if isinstance(data["output"], str):
                return data["output"].strip()
            if isinstance(data["output"], dict) and "content" in data["output"]:
                return str(data["output"]["content"]).strip()

        if "response" in data:
            response_content = data["response"]
            if isinstance(response_content, dict):
                if "output" in response_content:
                    return str(response_content["output"]).strip()
                if "content" in response_content:
                    return str(response_content["content"]).strip()

        return ""

    @staticmethod
    def generate_response_gemini(
        question: str,
        context: str,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> str:
        api_key = _get_gemini_api_key(api_key)
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY, GOOGLE_API_KEY, "
                "GOOGLE_GENAI_API_KEY, or add it to a local .env file."
            )

        endpoint = RagPipeline._build_gemini_endpoint(api_key, model)
        prompt_text = (
            "You are an expert medical assistant. Use the context from the PDFs to answer the question precisely. "
            "If the answer is not contained in the context, say you cannot find enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        payload = {
            "prompt": {"text": prompt_text},
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }

        headers = RagPipeline._build_gemini_headers(api_key)
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return RagPipeline._parse_gemini_response(data)

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
