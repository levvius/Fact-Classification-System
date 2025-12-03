import faiss
from typing import List, Dict
import logging

from app.core.models import ModelManager
from app.core.config import settings
from app.core.exceptions import EvidenceRetrievalException

logger = logging.getLogger(__name__)


def retrieve_proofs(claim: str, top_k: int = None) -> List[Dict[str, any]]:
    """
    Retrieve evidence snippets for a claim using FAISS similarity search.

    Args:
        claim: The claim text to find evidence for
        top_k: Number of top evidence snippets to return. If None, uses settings.top_k_proofs

    Returns:
        List of dicts with keys: snippet, source, retrieval_score

    Raises:
        EvidenceRetrievalException: If evidence retrieval fails
    """
    try:
        if top_k is None:
            top_k = settings.top_k_proofs

        # Get model manager
        mm = ModelManager.get_instance()
        embed_model = mm.get_embed_model()
        index = mm.get_index()
        kb_docs = mm.get_snippets()

        # Encode claim with error handling
        try:
            logger.debug(f"Encoding claim: {claim[:50]}...")
            emb = embed_model.encode([claim], convert_to_numpy=True, show_progress_bar=False)
        except BaseException as e:
            logger.error(f"Encoding failed: {str(e)}", exc_info=True)
            raise EvidenceRetrievalException(
                f"Failed to encode claim: {str(e)}",
                details={"claim": claim[:50], "error_type": type(e).__name__}
            )

        # Normalize embedding
        try:
            faiss.normalize_L2(emb)
        except BaseException as e:
            logger.error(f"Normalization failed: {str(e)}", exc_info=True)
            raise EvidenceRetrievalException(
                f"Failed to normalize embedding: {str(e)}",
                details={"error_type": type(e).__name__}
            )

        # FAISS search
        try:
            logger.debug(f"Searching FAISS index (top_k={top_k})...")
            D, I = index.search(emb, top_k)
        except BaseException as e:
            logger.error(f"FAISS search failed: {str(e)}", exc_info=True)
            raise EvidenceRetrievalException(
                f"FAISS search failed: {str(e)}",
                details={"top_k": top_k, "error_type": type(e).__name__}
            )

        # Build results with index validation
        results = []
        for i, score in zip(I[0], D[0]):
            if i < 0 or i >= len(kb_docs):
                logger.warning(f"FAISS returned out-of-bounds index: {i}")
                continue
            results.append({
                "snippet": kb_docs[i]["snippet"],
                "source": kb_docs[i]["source"],
                "retrieval_score": float(score)
            })

        return results

    except Exception as e:
        raise EvidenceRetrievalException(
            f"Failed to retrieve evidence: {str(e)}",
            details={"claim": claim[:50], "top_k": top_k, "error": str(e)}
        )
