# Fact Classification System

REST API для классификации английского текста как "правда", "неправда" или "нейтрально" с использованием NLI (Natural Language Inference) и Wikipedia.

## 🚀 Quick Start

```bash
# 1. Activate virtual environment (ВАЖНО!)
source venv/bin/activate

# 2. (First time only) Build Knowledge Base
python scripts/build_kb.py

# 3. Start the server
./run.sh

# 4. Open browser
# http://localhost:8000
```

**ВАЖНО**: Все команды Python должны выполняться с активированным виртуальным окружением!

---

## ✨ Features

### Web Interface (NEW!)
- 🎨 Modern, responsive web UI
- 📚 Browse 18 Wikipedia topics across 4 categories
- 🔍 Real-time fact classification
- 📊 Detailed results with evidence from Wikipedia
- ✅ Comprehensive error handling

### API Features
- 🧠 Natural Language Inference (RoBERTa-large-mnli)
- 🔎 FAISS vector search for evidence retrieval
- 📝 Automatic claim extraction from text
- 🌐 265 Wikipedia articles in Knowledge Base
- 🚦 Rate limiting (10 req/min)
- 💾 Response caching (5-minute TTL)
- 🔒 XSS validation and input sanitization

---

## 🔧 Installation

### Prerequisites

- Python 3.9-3.13 (recommended: 3.13.1)
- pip
- Virtual environment (venv)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd IS-hallucination-detection

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Build Knowledge Base (takes 2-5 minutes)
python scripts/build_kb.py
```

**Verification**: After successful setup, these files should exist:
- `data/faiss_index/wikipedia.index` (FAISS index, ~400KB)
- `data/kb_snippets.json` (metadata, ~145KB)

---

## 🎯 Usage

### Web Interface

1. **Start the server**:
   ```bash
   source venv/bin/activate  # Always activate first!
   ./run.sh
   ```

2. **Open browser**:
   ```
   http://localhost:8000
   ```

3. **Use the interface**:
   - Browse available topics (People, Technology, Science, History & Geography)
   - Click a topic to insert an example fact
   - Enter your own text (10-5000 characters)
   - Click "Classify Text"
   - View results with evidence

**Expected behavior**:
- First request: 5-10 seconds (models loading)
- Subsequent requests: 3-5 seconds (models cached)
- Green status indicator: API Ready
- Red status indicator: Models loading or error

### API Usage

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "kb_size": 265
}
```

#### Classify Text
```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein was born in 1879 and won the Nobel Prize in Physics."}'
```

Response:
```json
{
  "overall_classification": "правда",
  "confidence": 0.95,
  "claims": [
    {
      "claim": "Albert Einstein was born in 1879.",
      "classification": "правда",
      "confidence": 0.99,
      "best_evidence": {
        "snippet": "Albert Einstein was born in Ulm...",
        "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
        "nli_score": 0.99,
        "retrieval_score": 0.98
      }
    }
  ]
}
```

#### Get Available Topics
```bash
curl http://localhost:8000/api/v1/topics
```

---

## 🔍 Troubleshooting

### 1. ModuleNotFoundError: sentence_transformers

**Причина**: Виртуальное окружение не активировано

**Решение**:
```bash
source venv/bin/activate
python scripts/build_kb.py  # Now it will work
```

### 2. Network Error on Classify Button

**Причина**: API сервер не запущен

**Решение**:
```bash
source venv/bin/activate
./run.sh  # Start the server
```

Дождитесь сообщения:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
✓ Models loaded successfully
```

### 3. Models Not Loaded (503 Error)

**Причина**: Модели еще загружаются (первый запуск)

**Решение**: Подождите 5-10 секунд после запуска сервера. Модели загружаются автоматически.

### 4. Knowledge Base Missing

**Причина**: `data/faiss_index/wikipedia.index` не существует

**Решение**:
```bash
source venv/bin/activate
python scripts/build_kb.py  # Rebuild KB (2-5 minutes)
```

### 5. Rate Limit Exceeded (429 Error)

**Причина**: Превышен лимит 10 запросов в минуту

**Решение**: Подождите 60 секунд или перезапустите сервер

### 6. Port 8000 Already in Use

**Причина**: Другой процесс использует порт 8000

**Решение**:
```bash
# Find and kill the process
kill $(lsof -t -i:8000)

# Then restart
./run.sh
```

---

## 🧪 Testing

### Unit Tests (90 tests, ~5 seconds)
```bash
source venv/bin/activate
pytest tests/unit -m unit
```

### Integration Tests (16 tests, ~60 seconds)
```bash
source venv/bin/activate
pytest tests/integration -m integration
```

### All Tests with Coverage
```bash
source venv/bin/activate
pytest tests/ --cov=app --cov-report=html
```

---

## 📁 Project Structure

```
IS-hallucination-detection/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── routes.py              # API endpoints (/classify, /health, /topics)
│   │   └── schemas.py             # Pydantic models
│   ├── core/
│   │   ├── config.py              # Configuration
│   │   ├── models.py              # ModelManager singleton
│   │   ├── cache.py               # Response caching
│   │   └── exceptions.py          # Custom exceptions
│   ├── services/
│   │   ├── claim_extractor.py    # Extract claims from text
│   │   ├── evidence_retriever.py # FAISS search
│   │   ├── nli_verifier.py       # NLI scoring
│   │   └── classifier.py         # Main classification logic
│   ├── utils/
│   │   └── wikipedia_kb.py       # KB building utilities
│   └── static/                    # Frontend files (NEW!)
│       ├── index.html             # Main UI
│       ├── css/styles.css         # Responsive design
│       └── js/
│           ├── api.js             # API client
│           ├── ui.js              # UI controller
│           └── app.js             # Main logic
├── scripts/
│   └── build_kb.py                # Build Knowledge Base
├── tests/
│   ├── unit/                      # 90 unit tests
│   └── integration/               # 16 integration tests
├── data/
│   ├── faiss_index/               # FAISS vector index
│   └── kb_snippets.json           # KB metadata
├── requirements.txt               # Python dependencies
├── run.sh                         # Startup script
└── README.md                      # This file
```

---

## ⚙️ Configuration

Configuration is managed via `app/core/config.py`:

### Model Configuration
- `EMBED_MODEL`: `all-MiniLM-L6-v2` (sentence embeddings)
- `NLI_MODEL`: `roberta-large-mnli` (NLI scoring)

### Classification Thresholds
- `TRUTH_THRESHOLD`: 0.85 (>= 85% confidence = правда)
- `FALSEHOOD_THRESHOLD`: 0.4 (< 40% confidence = неправда)

### Retrieval Settings
- `TOP_K_PROOFS`: 6 (retrieve top 6 evidence snippets)
- `MAX_CLAIMS`: 8 (max claims to extract)

### API Settings
- `RATE_LIMIT_REQUESTS`: 10 (requests per minute)
- `CACHE_TTL`: 300 seconds (5 minutes)
- `CACHE_MAX_SIZE`: 100 entries

---

## 📖 How It Works

### Architecture Overview

```
User Input (English text)
    ↓
1. Claim Extraction
   - Split text into sentences
   - Extract factual claims
    ↓
2. Evidence Retrieval
   - FAISS vector search
   - Find top 6 relevant Wikipedia snippets
    ↓
3. NLI Verification
   - RoBERTa-large-mnli model
   - Score claim-evidence entailment
    ↓
4. Classification
   - Aggregate NLI scores
   - Apply thresholds (0.85/0.4)
   - Return verdict: правда/неправда/нейтрально
```

### Classification Logic

**Per-claim scoring:**
- `support >= 0.85` → "правда" (high confidence)
- `0.4 <= support < 0.85` → "нейтрально" (uncertain)
- `support < 0.4` → "неправда" (contradicts evidence)

**Overall aggregation** (pessimistic):
- ANY claim "неправда" → overall "неправда"
- Else, ANY claim "нейтрально" → overall "нейтрально"
- Else → overall "правда"

---

## 🤝 Contributing

This is a university project for "Технологии проектирования и сопровождения информационных систем".

**Course**: Information Systems Design and Maintenance Technologies
**University**: [Your University Name]
**Year**: 2025

---

## 📞 Support

For issues, please check:
1. [Troubleshooting](#troubleshooting) section above
2. Server logs (`uvicorn` output in terminal)
3. Browser console (F12) for frontend errors

---

## 🔗 Additional Resources

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/api/v1/health
- **Frontend**: http://localhost:8000
- **Project Documentation**: See `CLAUDE.md` for detailed architecture

---

## 📝 Recent Updates

### Version 2.0 (Current)
- ✅ Added web interface (HTML/CSS/JavaScript)
- ✅ 18 Wikipedia topics with examples
- ✅ Improved error handling with clear messages
- ✅ Environment checks in build scripts
- ✅ Better startup experience

### Version 1.0
- ✅ REST API with FastAPI
- ✅ NLI-based fact verification
- ✅ FAISS vector search
- ✅ Wikipedia knowledge base
- ✅ Comprehensive testing (106 tests)

---

**Made with ❤️ for accurate fact verification**
