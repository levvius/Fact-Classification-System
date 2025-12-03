# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REST API for classifying English text as "правда" (truth), "неправда" (falsehood), or "нейтрально" (neutral) using NLI (Natural Language Inference) and Wikipedia evidence verification.

## Core Architecture

The system is a **stateless FastAPI application** with the following pipeline:

1. **Claim Extraction** (`app/services/claim_extractor.py`) - Breaks input text into factual statements
2. **Evidence Retrieval** (`app/services/evidence_retriever.py`) - FAISS vector search against Wikipedia KB
3. **NLI Verification** (`app/services/nli_verifier.py`) - roberta-large-mnli scores claim-evidence entailment
4. **Classification** (`app/services/classifier.py`) - Aggregates scores and applies thresholds

### Key Components

- **ModelManager** (`app/core/models.py`) - Singleton pattern for model lifecycle management
  - Loads models once at startup, reuses across requests
  - Manages: SentenceTransformer (embeddings), HuggingFace pipeline (NLI), FAISS index, KB snippets
  - CRITICAL: Models must be loaded before first request or endpoints will fail

- **Configuration** (`app/core/config.py`) - pydantic-settings with .env support
  - Thresholds: `TRUTH_THRESHOLD` (default 0.85), `FALSEHOOD_THRESHOLD` (default 0.4)
  - Retrieval: `TOP_K_PROOFS` (default 6), `MAX_CLAIMS` (default 8)

- **Classification Logic** (`app/services/classifier.py:51-126`)
  - Per-claim: support >= 0.85 → "правда", < 0.4 → "неправда", else "нейтрально"
  - Overall: ANY "неправда" → overall "неправда" (pessimistic aggregation)

## Stability & Performance (macOS Critical)

### Single-Threaded Mode (REQUIRED on macOS)

The system MUST run in single-threaded mode on macOS to prevent segmentation faults:

**File:** `app/core/models.py:47-55`
```python
# CRITICAL: Single-threaded mode to prevent threading crashes on macOS
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
logger.info("PyTorch threads limited to 1 (single-threaded mode for stability)")
```

**Why:** PyTorch multi-threading causes crashes with roberta-large-mnli (355M parameters) on Apple Silicon.

### ThreadPoolExecutor Pattern

**File:** `app/api/routes.py:26-80`

ML operations run in dedicated thread pool to prevent FastAPI event loop blocking:

```python
ml_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-worker")

# In endpoint:
result = await asyncio.wait_for(
    loop.run_in_executor(ml_executor, classify_text, text),
    timeout=45.0  # 45-second timeout
)
```

**Why:** Synchronous ML operations (NLI, FAISS) would block async FastAPI.

### Device Configuration

**File:** `app/core/models.py:60`

Force CPU-only mode to prevent MPS GPU crashes:

```python
self._embed_model = SentenceTransformer(settings.embedding_model, device='cpu')
```

**Why:** MPS (Metal Performance Shaders) auto-detection causes PyTorch backend crashes.

### Removed Features (Safety)

1. **Signal Handlers** - Removed from `app/main.py`
   - SIGSEGV/SIGABRT handlers caused infinite recursion
   - Cannot safely log when memory is corrupted

2. **Uvicorn --reload** - Removed from `run.sh:90`
   - Process forking incompatible with ML pipeline on macOS
   - Use manual restarts during development

3. **Progress Bars** - Disabled in `evidence_retriever.py:39`
   - `show_progress_bar=False` prevents multiprocessing deadlocks

**Impact of Stability Fixes:**
- ✅ No more segmentation faults on macOS
- ✅ No more infinite recursion from signal handlers
- ✅ Stable multi-claim classification
- ✅ All 106 tests pass reliably
- ✅ System runs crash-free for extended periods

## Frontend Architecture (NEW)

The system now includes a **web interface** served via FastAPI StaticFiles:

- **Location**: `app/static/`
- **Technology**: Vanilla HTML/CSS/JavaScript (zero build tools)
- **Entry Point**: `app/static/index.html`
- **API Integration**: Same-origin requests to `/api/v1/*` (no CORS issues)

### Frontend Structure

```
app/static/
├── index.html              # Main UI structure
├── css/
│   └── styles.css          # Responsive design with gradient header
└── js/
    ├── api.js              # APIClient class - fetch wrapper with error handling
    ├── ui.js               # UIController class - DOM manipulation
    └── app.js              # Main application logic and initialization
```

### Frontend Features

1. **Topics Display** (18 topics in 4 categories)
   - People: Albert Einstein, Barack Obama
   - Technology: AI, ML, Python, Linux, Microsoft, Google, Tesla, Amazon
   - Science: Quantum mechanics, Climate change, COVID-19
   - History & Geography: WWII, New York, Mount Everest

2. **Interactive Input**
   - Real-time character counter (10-5000 chars)
   - Click topic cards to insert example facts
   - Client-side validation with visual feedback

3. **Classification Results**
   - Overall classification badge (правда/неправда/нейтрально)
   - Confidence bar (0-100%)
   - Expandable claims with evidence
   - Wikipedia source links
   - NLI and retrieval scores

4. **Error Handling**
   - 429 Rate Limit: Display countdown timer
   - 422 Validation: Show specific validation errors
   - 503 Models Loading: Retry suggestion with instructions
   - Network: Clear step-by-step server startup instructions

5. **Health Status**
   - Real-time API health indicator (green/red/yellow)
   - Periodic health checks (every 30 seconds)
   - Clear status messages

### Accessing the Frontend

```bash
# Start the server
./run.sh

# Navigate to:
http://localhost:8000
```

The root endpoint (`GET /`) now serves `index.html` if it exists, or falls back to API info.

### Frontend API Endpoints

All endpoints are accessed via `/api/v1/`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend UI |
| `/api/v1/health` | GET | Health check (models status, KB size) |
| `/api/v1/classify` | POST | Classify text (main endpoint) |
| `/api/v1/topics` | GET | Get available Wikipedia topics |
| `/api/v1/cache-info` | GET | Cache statistics (debugging) |
| `/docs` | GET | Swagger UI (auto-generated) |

### Frontend Error Flow

```
User clicks "Classify Text"
    ↓
api.classifyText(text) → fetch POST /api/v1/classify
    ↓
Response handling:
    ├─ 200 OK → ui.renderResults(result)
    ├─ 429 Rate Limit → Show "Wait 60s" message
    ├─ 422 Validation → Show validation errors
    ├─ 503 Not Ready → Show "Models loading" + retry
    └─ Network Error (Failed to fetch) → Show detailed server startup instructions
```

### Development Notes

**No Build Tools Required:**
- Pure HTML/CSS/JavaScript (ES6+)
- No npm, webpack, or bundlers
- Changes are reflected immediately (browser refresh)

**Same-Origin Policy:**
- Frontend served from same domain as API
- No CORS configuration needed
- Fetch requests use relative URLs (`/api/v1/...`)

**Browser Compatibility:**
- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+)
- Requires ES6+ support (Fetch API, async/await, classes)

**Debugging:**
- Open DevTools (F12) → Console for JavaScript errors
- Network tab for API requests/responses
- Error banner shows user-friendly messages

## Common Commands

### First-time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Build Wikipedia Knowledge Base (required before first run)
python scripts/build_kb.py
```

### Running the API
```bash
# Quick start (handles setup + run)
./run.sh

# Manual start
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

**API Testing:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Classify text
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein was born in 1879."}'
```

**Unit & Integration Tests:**
See [Testing Infrastructure](#testing-infrastructure) section below for running unit and integration tests.

## Important Details

### Knowledge Base
- Located at `data/faiss_index/wikipedia.index` + `data/kb_snippets.json`
- Built by `scripts/build_kb.py` - scrapes ~18 Wikipedia topics, ~265 snippets
- Uses FAISS IndexFlatL2 for L2 distance search
- Must exist before starting server (run.sh auto-builds if missing)

### Model Loading
- All models loaded synchronously at startup via `@app.on_event("startup")` in `app/main.py:22-34`
- First request after startup will be slow (~5-10s) while models initialize
- Subsequent requests fast (models cached in memory)

### Python Version

- **Recommended**: Python 3.13.1 (installed via pyenv)
- **Supported**: Python 3.9-3.13
- **Not supported**: Python 3.14 (transformers incompatibility)
- **Migration**: Completed in Фаза 11 (3.14 → 3.13.1)

### Project Structure
```
app/
├── main.py              # FastAPI app, startup/shutdown hooks
├── api/
│   ├── routes.py        # /classify and /health endpoints
│   └── schemas.py       # Pydantic request/response models
├── core/
│   ├── config.py        # Settings (thresholds, paths, model names)
│   └── models.py        # ModelManager singleton
├── services/            # Business logic (claim extraction, retrieval, NLI, classification)
└── utils/
    └── wikipedia_kb.py  # Wikipedia scraping utilities
```

## Testing Infrastructure

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # 90 unit tests with mocks (~5s)
│   ├── test_config.py       # Configuration validation (7 tests)
│   ├── test_models.py       # ModelManager singleton (12 tests)
│   ├── test_claim_extractor.py   # Claim extraction (16 tests)
│   ├── test_evidence_retriever.py  # FAISS retrieval (16 tests)
│   ├── test_nli_verifier.py      # NLI scoring (18 tests)
│   └── test_classifier.py        # Classification logic (21 tests)
└── integration/             # 16 integration tests with real models (~60s)
    ├── test_classification_pipeline.py  # End-to-end classification (4 tests)
    └── test_api_endpoints.py           # API routes testing (12 tests)
```

### Pytest Markers

- `@pytest.mark.unit` - Fast unit tests with mocks
- `@pytest.mark.integration` - Integration tests with real models
- `@pytest.mark.slow` - Slow tests (>10s), typically with model loading

### Key Fixtures (tests/conftest.py)

- `mock_model_manager` - Mocked ModelManager for fast unit tests
- `real_model_manager` - Real ModelManager with loaded models (module scope)
- `test_client` - FastAPI TestClient for API testing
- `mock_embed_model`, `mock_nli_pipeline`, `mock_faiss_index`, `mock_kb_snippets` - Individual mocks

### Running Tests

```bash
# Unit tests only
pytest tests/unit -m unit

# Integration tests only (requires KB)
pytest tests/integration -m integration

# With coverage
pytest tests/unit --cov=app --cov-report=html
```

## Security Features

### Exception Handling

8 custom exceptions in `app/core/exceptions.py`:
- `AppBaseException` - Base exception with HTTP status codes
- `ModelNotLoadedException` - Models not loaded (503)
- `KnowledgeBaseException` - KB not found (503)
- `ClaimExtractionException` - Claim extraction failed (500)
- `EvidenceRetrievalException` - Evidence retrieval failed (500)
- `NLIVerificationException` - NLI verification failed (500)
- `ClassificationException` - Classification failed (500)
- `CacheException` - Cache operation failed (500)

### Rate Limiting

- Library: `slowapi` (token bucket algorithm)
- Default: 10 requests/minute per IP
- Burst: 3 additional requests
- Endpoint: All `/api/v1/*` routes
- Configuration: `app/core/config.py` (RATE_LIMIT_REQUESTS, RATE_LIMIT_BURST)

### Input Validation

XSS protection in `app/api/schemas.py`:
- 10 dangerous patterns detected: `<script>`, `javascript:`, `onerror=`, etc.
- **Minimum text length: 10 characters** (approximately 3 words)
- ValidationError (422) on XSS attempt

### Caching

- Library: `cachetools.TTLCache`
- TTL: 5 minutes
- Max size: 100 entries
- Key: MD5 hash of input text
- Location: `app/core/cache.py`
- Endpoint: `/cache-info` for cache statistics

### Additional Endpoints

- **GET /cache-info** - Returns cache statistics (size, maxsize)
  ```bash
  curl http://localhost:8000/cache-info
  # Returns: {"size": 15, "maxsize": 100}
  ```

## Historical Context

This is a university project for "Технологии проектирования и сопровождения информационных систем" (Information Systems Design and Maintenance Technologies).

**Migration Note**: The project was migrated from a PostgreSQL-based architecture (see `models.psql` in git history) to a stateless API. The database schema (`users`, `requests`, `responses`, `claims`, `proofs`, `tickets`) is no longer active but remains in version control for reference.
