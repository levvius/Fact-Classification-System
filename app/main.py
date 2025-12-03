from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pathlib import Path
import logging

from app.api.routes import router
from app.core.models import ModelManager
from app.core.exceptions import (
    AppBaseException,
    ModelNotLoadedException,
    InputValidationException,
    KnowledgeBaseException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="Classifies English text as truth, falsehood, or neutral using NLI and Wikipedia evidence",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Exception handlers
@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    """Handle model not loaded errors (503)."""
    logger.error(f"Model not loaded: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(KnowledgeBaseException)
async def knowledge_base_handler(request: Request, exc: KnowledgeBaseException):
    """Handle knowledge base errors (503)."""
    logger.error(f"Knowledge base error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(InputValidationException)
async def input_validation_handler(request: Request, exc: InputValidationException):
    """Handle input validation errors (400)."""
    logger.warning(f"Input validation failed: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(AppBaseException)
async def app_base_exception_handler(request: Request, exc: AppBaseException):
    """Handle all other application exceptions (500)."""
    logger.error(f"Application error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    logger.info("Starting up application...")
    logger.info("Loading models...")

    try:
        mm = ModelManager.get_instance()
        mm.load_models()
        logger.info("✓ Models loaded successfully")
        logger.info("=" * 60)
        logger.info("  📡 API Documentation: http://localhost:8000/docs")
        logger.info("  🌐 Web Interface:     http://localhost:8000")
        logger.info("  ❤️  Health Check:      http://localhost:8000/api/v1/health")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Failed to load models: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application...")


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["classification"])


# Mount static files for frontend
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def root():
    """Serve frontend UI or API information."""
    static_path = Path(__file__).parent / "static"
    index_path = static_path / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    # Fallback if no frontend
    return {
        "message": "Text Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
