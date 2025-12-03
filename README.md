# IS-hallucination-detection

REST API для классификации текста на английском языке как "правда" (truth), "неправда" (falsehood) или "нейтрально" (neutral) с использованием NLI (Natural Language Inference) и проверки фактов через Wikipedia.

Репозиторий для курса "Технологии проектирования и сопровождения информационных систем".

## Описание

Приложение анализирует английский текст, извлекает из него утверждения (claims), проверяет их с помощью базы знаний Wikipedia и NLI-модели, и возвращает классификацию для каждого утверждения и всего текста в целом.

### Как это работает

1. **Извлечение утверждений** - текст разбивается на предложения, из которых выбираются фактические утверждения
2. **Поиск доказательств** - для каждого утверждения ищутся релевантные фрагменты из Wikipedia через FAISS векторный поиск
3. **NLI верификация** - модель roberta-large-mnli оценивает, насколько доказательства подтверждают утверждение
4. **Классификация** - на основе confidence score выдается вердикт:
   - `support >= 0.85` → "правда"
   - `0.4 <= support < 0.85` → "нейтрально"
   - `support < 0.4` → "неправда"

## Установка

### Требования

- **Python 3.9-3.13** (рекомендуется 3.13.1)
  - ⚠️ Python 3.14 не поддерживается из-за несовместимости с transformers
  - Python 3.8 и ниже не поддерживаются
- 2.5-3GB RAM
- 2.5GB свободного места на диске

### Шаги установки

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/IS-hallucination-detection.git
cd IS-hallucination-detection
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Постройте базу знаний из Wikipedia (выполняется один раз, ~5-10 минут):
```bash
python scripts/build_kb.py
```

## Запуск

### Быстрый старт

```bash
./run.sh
```

Или вручную:
```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу:
- **API**: http://localhost:8000
- **Документация (Swagger UI)**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/api/v1/health

## Использование API

### Классификация текста

**Endpoint:** `POST /api/v1/classify`

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Albert Einstein was born in 1879. Python is a statically typed language."
  }'
```

**Пример ответа:**
```json
{
  "overall_classification": "неправда",
  "confidence": 0.85,
  "claims": [
    {
      "claim": "Albert Einstein was born in 1879.",
      "classification": "правда",
      "confidence": 0.99,
      "best_evidence": {
        "snippet": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
        "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
        "nli_score": 0.99,
        "retrieval_score": 0.98
      }
    },
    {
      "claim": "Python is a statically typed language.",
      "classification": "неправда",
      "confidence": 0.92,
      "best_evidence": {
        "snippet": "Python uses dynamic typing and a combination of reference counting...",
        "source": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "nli_score": 0.08,
        "retrieval_score": 0.95
      }
    }
  ]
}
```

### Health Check

**Endpoint:** `GET /api/v1/health`

```bash
curl "http://localhost:8000/api/v1/health"
```

**Ответ:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "kb_size": 265
}
```

### Cache Statistics

**Endpoint:** `GET /cache-info`

```bash
curl "http://localhost:8000/cache-info"
```

**Ответ:**
```json
{
  "size": 15,
  "maxsize": 100
}
```

## Security Features

### Rate Limiting

API endpoints are protected with rate limiting to prevent abuse:
- **Library**: slowapi (token bucket algorithm)
- **Default limit**: 10 requests per minute per IP
- **Burst allowance**: 3 additional requests
- **Applies to**: All `/api/v1/*` endpoints
- **Configuration**: `app/core/config.py` (RATE_LIMIT_REQUESTS, RATE_LIMIT_BURST)

### Input Validation

XSS protection in `app/api/schemas.py`:
- Detects 10 dangerous patterns: `<script>`, `javascript:`, `onerror=`, `onclick=`, `<iframe>`, `<object>`, `<embed>`, `eval()`, `document.cookie`, `window.location`
- Minimum text length: 10 characters (approximately 3 words)
- Returns **422 Validation Error** on XSS attempt

### Exception Handling

8 custom exceptions in `app/core/exceptions.py`:
- `ModelNotLoadedException` - Models not loaded (503)
- `KnowledgeBaseException` - KB not found (503)
- `ClaimExtractionException` - Claim extraction failed (500)
- `EvidenceRetrievalException` - Evidence retrieval failed (500)
- `NLIVerificationException` - NLI verification failed (500)
- `ClassificationException` - Classification failed (500)
- `CacheException` - Cache operation failed (500)

### Caching

Response caching for improved performance:
- **Library**: cachetools.TTLCache
- **TTL**: 5 minutes
- **Max size**: 100 entries
- **Key**: MD5 hash of input text
- **Implementation**: `app/core/cache.py`
- **Cache info endpoint**: `GET /cache-info` for statistics

## Testing

This project includes comprehensive test coverage with unit and integration tests.

### Running Tests

**Unit Tests** (fast, use mocks):
```bash
# All unit tests
pytest tests/unit -m unit -v

# With coverage report
pytest tests/unit -m unit --cov=app --cov-report=html

# Specific test file
pytest tests/unit/test_config.py -v
```

**Integration Tests** (slow, use real models):
```bash
# Build Knowledge Base first (required)
python scripts/build_kb.py

# Run integration tests
pytest tests/integration -m integration -v

# Skip slow tests
pytest tests/integration -m "integration and not slow" -v
```

**All Tests:**
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Test Structure

- `tests/unit/` - 90 unit tests with mocks (fast, ~5s)
  - test_config.py: Configuration validation (7 tests)
  - test_models.py: ModelManager singleton (12 tests)
  - test_claim_extractor.py: Claim extraction (16 tests)
  - test_evidence_retriever.py: FAISS retrieval (16 tests)
  - test_nli_verifier.py: NLI scoring (18 tests)
  - test_classifier.py: Classification logic (21 tests)

- `tests/integration/` - 16 integration tests with real models (slow, ~60s)
  - test_classification_pipeline.py: End-to-end classification (4 tests)
  - test_api_endpoints.py: API routes testing (12 tests)

- `tests/conftest.py` - Shared fixtures (mock_model_manager, real_model_manager, etc.)

### Coverage

- Target: >= 85% for service layer
- Current: ~90% for app/services/, ~95% for app/core/

## Структура проекта

```
IS-hallucination-detection/
├── app/
│   ├── api/              # API endpoints и схемы
│   ├── core/             # Конфигурация и управление моделями
│   ├── services/         # Бизнес-логика (извлечение, поиск, верификация)
│   └── utils/            # Утилиты (построение KB)
├── scripts/              # Скрипты (build_kb.py)
├── tests/                # Тесты
├── data/                 # FAISS индекс и метаданные KB (создается автоматически)
├── models/               # Кэш ML моделей (создается автоматически)
├── requirements.txt      # Python зависимости
└── run.sh               # Скрипт запуска
```

## Конфигурация

Параметры можно настроить через переменные окружения (см. `.env.example`):

- `TRUTH_THRESHOLD` - порог для классификации как "правда" (default: 0.85)
- `FALSEHOOD_THRESHOLD` - порог для классификации как "неправда" (default: 0.4)
- `TOP_K_PROOFS` - количество доказательств для проверки (default: 6)
- `MAX_CLAIMS` - максимум утверждений для извлечения (default: 8)

## Технологии

- **FastAPI** (0.123.5) - веб-фреймворк для API
- **sentence-transformers** (3.4.1, all-MiniLM-L6-v2) - векторные представления текста
- **transformers** (4.57.3, roberta-large-mnli) - NLI модель для верификации
- **FAISS** (1.13.0) - векторный поиск
- **Wikipedia API** (1.4.0) - база знаний
- **NumPy** (2.3.5) - математические операции
- **PyTorch** (2.9.1) - deep learning framework
- **slowapi** (0.1.9) - rate limiting
- **cachetools** (6.2.2) - response caching

## Лицензия

См. файл LICENSE
