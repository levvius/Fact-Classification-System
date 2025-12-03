from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse
)
from app.services.classifier import classify_text
from app.core.models import ModelManager
from app.core.cache import get_cached_result, cache_result, get_cache_info
from app.core.exceptions import ClassificationException

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Dedicated thread pool for ML operations (prevents event loop blocking)
ml_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-worker")


@router.post("/classify", response_model=ClassifyResponse)
@limiter.limit("10/minute")
async def classify_endpoint(request: Request, classify_req: ClassifyRequest):
    """
    Classify English text as truth, falsehood, or neutral.

    Rate limit: 10 requests per minute per IP.
    Cached results expire after 5 minutes.

    Args:
        request: FastAPI Request object (for rate limiting)
        classify_req: ClassifyRequest with text field

    Returns:
        ClassifyResponse with overall classification, confidence, and claim analysis
    """
    import sys

    # Log with flush for crash debugging
    logger.info("=" * 60)
    logger.info(f"📝 CLASSIFY REQUEST START")
    logger.info(f"   Text length: {len(classify_req.text)} chars")
    logger.info(f"   Preview: {classify_req.text[:100]}...")
    sys.stdout.flush()

    try:
        # Check cache
        cached = get_cached_result(classify_req.text)
        if cached:
            logger.info("✓ Cache hit")
            sys.stdout.flush()
            return cached

        # Run blocking ML in separate thread to avoid blocking event loop
        logger.info("🔍 Starting classification in worker thread...")
        sys.stdout.flush()

        loop = asyncio.get_event_loop()

        # Add timeout protection
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(ml_executor, classify_text, classify_req.text),
                timeout=45.0  # 45 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("❌ Classification timed out after 45 seconds")
            sys.stdout.flush()
            raise ClassificationException(
                "Classification timed out. The text might be too complex or the server is overloaded.",
                details={"timeout": 45, "text_length": len(classify_req.text)}
            )

        logger.info(f"✓ Classification complete: {result['overall_classification']}")
        sys.stdout.flush()

        # Cache result
        cache_result(classify_req.text, result)

        logger.info("✅ REQUEST COMPLETE")
        logger.info("=" * 60)
        sys.stdout.flush()

        return result

    except Exception as e:
        logger.error(f"❌ CLASSIFICATION FAILED: {str(e)}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise ClassificationException(
            f"Failed to classify text: {str(e)}",
            details={"error_type": type(e).__name__}
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and model loading state.

    Returns:
        HealthResponse with status, models_loaded flag, and KB size
    """
    mm = ModelManager.get_instance()

    try:
        # Check if models are loaded
        _ = mm.get_embed_model()
        _ = mm.get_nli()
        _ = mm.get_index()
        snippets = mm.get_snippets()

        return HealthResponse(
            status="healthy",
            models_loaded=True,
            kb_size=len(snippets)
        )
    except Exception:
        return HealthResponse(
            status="not_ready",
            models_loaded=False,
            kb_size=0
        )


@router.get("/cache-info")
async def cache_info_endpoint():
    """Get cache statistics (for debugging)."""
    return get_cache_info()


@router.get("/topics")
async def get_topics():
    """
    Get available Wikipedia topics in the knowledge base.

    Returns topics grouped by category for better UX.

    Returns:
        Dict with total_topics count and categorized topic lists
    """
    from app.utils.wikipedia_kb import SEED_TOPICS

    categories = {
        "People": ["Albert Einstein", "Barack Obama"],
        "Technology": [
            "Python (programming language)",
            "Artificial intelligence",
            "Machine learning",
            "Neural network",
            "Data science",
            "Linux",
            "Microsoft",
            "Google",
            "Tesla, Inc.",
            "Amazon (company)"
        ],
        "Science": [
            "Quantum mechanics",
            "Climate change",
            "COVID-19"
        ],
        "History & Geography": [
            "World War II",
            "New York City",
            "Mount Everest"
        ]
    }

    return {
        "total_topics": len(SEED_TOPICS),
        "categories": categories
    }
