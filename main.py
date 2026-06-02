import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

DEFAULT_PICKLE = Path(__file__).parent / "rag_store.pkl"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_store(path):
    try:
        return torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load store.\n"
            f"This usually means rag_store.pkl was saved with CUDA tensors.\n"
            f"Regenerate the store with embeddings.cpu() before saving.\n\n"
            f"Original error:\n{e}"
        )


def ensure_tensor(embeddings):
    if torch.is_tensor(embeddings):
        return embeddings.float().cpu()

    return torch.tensor(
        embeddings,
        dtype=torch.float32,
    )


def load_model(model_name, device):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    return SentenceTransformer(
        model_name,
        device=device,
    )


def encode_query(model, query):
    query = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    return model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def retrieve_top_k(
    query,
    embeddings,
    texts,
    model,
    top_k=5,
    threshold=0.35,
):
    query_embedding = encode_query(
        model,
        query,
    )

    embeddings = ensure_tensor(
        embeddings
    )

    scores = util.cos_sim(
        query_embedding,
        embeddings,
    )[0]

    top_k = min(
        top_k,
        len(texts),
    )

    top_results = torch.topk(
        scores,
        k=top_k,
    )

    results = []

    for idx, score in zip(
        top_results.indices,
        top_results.values,
    ):
        score = float(score)

        if score >= threshold:
            results.append(
                (
                    score,
                    texts[idx.item()],
                )
            )

    return results


def print_environment():
    print("\n=== Environment ===")
    print("PyTorch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("===================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=str(DEFAULT_PICKLE),
    )

    parser.add_argument(
        "--query",
        default="what is diabetes",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    args = parser.parse_args()

    print_environment()

    store_path = Path(args.path)

    if not store_path.exists():
        raise FileNotFoundError(
            f"Store not found: {store_path}"
        )

    print("Loading vector store...")

    store = load_store(store_path)

    embeddings = ensure_tensor(
        store["embeddings"]
    )

    texts = store["texts"]

    device = get_device()

    model = load_model(
        args.model,
        device,
    )

    results = retrieve_top_k(
        query=args.query,
        embeddings=embeddings,
        texts=texts,
        model=model,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print("\n=== Results ===\n")

    if not results:
        print(
            "No documents found above threshold."
        )
        return

    context = []

    for i, (score, text) in enumerate(
        results,
        start=1,
    ):
        print(
            f"Result {i} | Score: {score:.4f}\n"
        )

        print(text[:1000])

        print(
            "\n"
            + "=" * 80
            + "\n"
        )

        context.append(
            f"[Document {i}]\n{text}"
        )

    final_context = "\n\n".join(
        context
    )

    print(
        "\nContext length:",
        len(final_context),
    )


if __name__ == "__main__":
    main()