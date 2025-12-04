import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.models import ModelManager


@pytest.fixture
def test_client():
    """FastAPI test client for API endpoint testing."""
    return TestClient(app)


@pytest.fixture
def mock_embed_model():
    """
    Mock SentenceTransformer embedding model.

    Returns 384-dimensional embeddings (all-MiniLM-L6-v2 default size).
    """
    mock = Mock()
    mock.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    return mock


@pytest.fixture
def mock_nli_pipeline():
    """
    Mock HuggingFace NLI pipeline (roberta-large-mnli).

    Returns entailment classification results.
    """
    mock = Mock()
    mock.return_value = [{"label": "ENTAILMENT", "score": 0.92}]
    return mock


@pytest.fixture
def mock_faiss_index():
    """
    Mock FAISS index for vector similarity search.

    Returns distances and indices for top-k search results.
    Returns all 6 KB snippets to match the mock_kb_snippets fixture.
    """
    mock = Mock()
    # Return: (distances, indices)
    # distances: lower is more similar
    # indices: indices in knowledge base
    mock.search.return_value = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),  # 6 distances
        np.array([[0, 1, 2, 3, 4, 5]])  # 6 indices
    )
    return mock


@pytest.fixture
def mock_kb_snippets():
    """
    Mock knowledge base snippets metadata.

    Simulates Wikipedia knowledge base with diverse topics.
    """
    return [
        {
            "snippet": "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
            "source": "https://en.wikipedia.org/wiki/Albert_Einstein"
        },
        {
            "snippet": "Python is a dynamically typed, high-level programming language.",
            "source": "https://en.wikipedia.org/wiki/Python_(programming_language)"
        },
        {
            "snippet": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "source": "https://en.wikipedia.org/wiki/Earth"
        },
        {
            "snippet": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            "source": "https://en.wikipedia.org/wiki/Water"
        },
        {
            "snippet": "Artificial intelligence was founded as an academic discipline in 1956.",
            "source": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        },
        {
            "snippet": "The structure of DNA was discovered by James Watson and Francis Crick in 1953.",
            "source": "https://en.wikipedia.org/wiki/DNA"
        }
    ]


@pytest.fixture
def mock_model_manager(mock_embed_model, mock_nli_pipeline, mock_faiss_index, mock_kb_snippets):
    """
    Complete mocked ModelManager with all models initialized.

    Use this fixture for unit tests that need ModelManager but don't
    want to load real models (faster, no GPU/CPU heavy operations).

    Yields:
        ModelManager instance with mocked models

    Cleanup:
        Resets ModelManager singleton after test
    """
    # Reset singleton
    ModelManager._instance = None

    # Create new instance
    mm = ModelManager.get_instance()

    # Inject mocks
    mm._embed_model = mock_embed_model
    mm._nli_pipeline = mock_nli_pipeline
    mm._faiss_index = mock_faiss_index
    mm._kb_snippets = mock_kb_snippets

    yield mm

    # Cleanup
    ModelManager._instance = None


@pytest.fixture
def sample_text():
    """Sample input text for testing classification pipeline."""
    return "Albert Einstein was born in 1879. Python is a statically typed language."


@pytest.fixture
def sample_claims():
    """Sample extracted claims for testing."""
    return [
        "Albert Einstein was born in 1879.",
        "Python is a statically typed language."
    ]


@pytest.fixture
def sample_classify_request():
    """Sample ClassifyRequest payload for API testing."""
    return {
        "text": "Albert Einstein was born on March 14, 1879 in Germany. He developed the theory of relativity."
    }


@pytest.fixture
def sample_classify_response():
    """Sample ClassifyResponse for validation testing."""
    return {
        "overall_classification": "правда",
        "confidence": 0.89,
        "claims": [
            {
                "claim": "Albert Einstein was born on March 14, 1879 in Germany.",
                "classification": "правда",
                "confidence": 0.92,
                "best_evidence": {
                    "snippet": "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
                    "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                    "nli_score": 0.95,
                    "retrieval_score": 0.12
                }
            },
            {
                "claim": "He developed the theory of relativity.",
                "classification": "правда",
                "confidence": 0.86,
                "best_evidence": {
                    "snippet": "Einstein developed the theory of relativity.",
                    "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                    "nli_score": 0.88,
                    "retrieval_score": 0.15
                }
            }
        ]
    }


# Integration test fixtures (real models)

@pytest.fixture(scope="module")
def real_model_manager():
    """
    Real ModelManager with actual models loaded.

    Use this for integration tests that need real model inference.
    Module-scoped to load models only once per test module.

    Note: This is slow (~10-30s) and requires significant memory/CPU.
    """
    mm = ModelManager.get_instance()
    if mm._embed_model is None:
        mm.load_models()
    yield mm
    # Note: We don't reset singleton here to reuse across integration tests


@pytest.fixture
def factual_text():
    """Factual text that should classify as 'правда' (truth)."""
    return "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921."


@pytest.fixture
def false_text():
    """False text that should classify as 'неправда' (falsehood)."""
    return "Albert Einstein was born in 1990. Python is a statically typed language."


@pytest.fixture
def neutral_text():
    """Neutral/ambiguous text that should classify as 'нейтрально'."""
    return "The future of artificial intelligence is uncertain and depends on many factors."


@pytest.fixture
def mixed_text():
    """Text with mixed true and false claims."""
    return "Einstein was born in 1879. Python requires static type declarations before runtime."


# XSS validation test fixtures

@pytest.fixture
def xss_payloads():
    """Various XSS/injection attack payloads for validation testing."""
    return [
        "<script>alert('XSS')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "<iframe src='javascript:alert(1)'></iframe>",
        "eval(document.cookie)",
        "<embed src='malicious.swf'>",
        "<object data='malicious.pdf'>",
        "<input onclick='alert(1)'>",
        "<body onload='alert(1)'>",
        "document.cookie='stolen'",
    ]


@pytest.fixture
def invalid_short_texts():
    """Texts that are too short and should fail validation."""
    return [
        "Hi",  # 1 word
        "Hello there",  # 2 words
        "",  # Empty
        "   ",  # Whitespace only
    ]
