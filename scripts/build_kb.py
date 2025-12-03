#!/usr/bin/env python3
"""
Script to build the knowledge base from Wikipedia.
Creates FAISS index and saves snippet metadata.
"""

import sys
import json
from pathlib import Path


def check_environment():
    """Check if virtual environment is activated and dependencies are available."""
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("❌ ERROR: Virtual environment is not activated!")
        print("\nPlease activate the virtual environment first:")
        print("  source venv/bin/activate")
        print("\nThen run this script again:")
        print("  python scripts/build_kb.py")
        sys.exit(1)

    # Check required packages
    missing_packages = []
    try:
        import sentence_transformers
    except ImportError:
        missing_packages.append('sentence-transformers')

    try:
        import faiss
    except ImportError:
        missing_packages.append('faiss-cpu')

    try:
        import wikipedia
    except ImportError:
        missing_packages.append('wikipedia')

    try:
        import tqdm
    except ImportError:
        missing_packages.append('tqdm')

    if missing_packages:
        print(f"❌ ERROR: Missing required packages: {', '.join(missing_packages)}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("✓ Environment check passed")


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

from app.utils.wikipedia_kb import build_kb_snippets
from app.core.config import settings


def main():
    # Check environment first
    check_environment()
    print("Building Knowledge Base from Wikipedia...")
    print(f"Using {settings.max_sentences_per_page} sentences per page")

    # Build KB snippets from Wikipedia
    print("\nFetching Wikipedia pages...")
    kb_docs = build_kb_snippets(max_sentences_per_page=settings.max_sentences_per_page)
    print(f"KB built: {len(kb_docs)} snippets")

    # Extract snippets text
    snippets = [d["snippet"] for d in kb_docs]

    # Load embedding model
    print(f"\nLoading embedding model: {settings.embedding_model}")
    embed_model = SentenceTransformer(settings.embedding_model)

    # Encode snippets
    print("\nEncoding KB snippets (this may take some time)...")
    embeddings = embed_model.encode(snippets, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    print("\nBuilding FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity

    # Normalize vectors for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")

    # Create output directories
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    print(f"\nSaving FAISS index to: {settings.faiss_index_path}")
    faiss.write_index(index, str(settings.faiss_index_path))

    # Save KB snippets metadata
    print(f"Saving KB snippets metadata to: {settings.kb_snippets_path}")
    with open(settings.kb_snippets_path, "w", encoding="utf-8") as f:
        json.dump(kb_docs, f, ensure_ascii=False, indent=2)

    print("\n✓ Knowledge Base built successfully!")
    print(f"  - Index file: {settings.faiss_index_path}")
    print(f"  - Metadata file: {settings.kb_snippets_path}")
    print(f"  - Total snippets: {len(kb_docs)}")


if __name__ == "__main__":
    main()
