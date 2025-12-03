import pytest
from app.core.config import settings


@pytest.mark.unit
def test_default_settings():
    """Test that default configuration values are correct."""
    assert settings.truth_threshold == 0.85
    assert settings.falsehood_threshold == 0.4
    assert settings.top_k_proofs == 6
    assert settings.max_claims == 8


@pytest.mark.unit
def test_threshold_ordering():
    """Test that truth threshold is greater than falsehood threshold."""
    assert settings.truth_threshold > settings.falsehood_threshold


@pytest.mark.unit
def test_api_configuration():
    """Test API configuration defaults."""
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert isinstance(settings.api_debug, bool)


@pytest.mark.unit
def test_model_configuration():
    """Test model configuration values."""
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.nli_model == "roberta-large-mnli"
    assert settings.device in ["cpu", "cuda"]


@pytest.mark.unit
def test_rate_limiting_configuration():
    """Test rate limiting configuration."""
    assert isinstance(settings.rate_limit_enabled, bool)
    assert settings.rate_limit_requests == 10
    assert settings.rate_limit_burst == 3


@pytest.mark.unit
def test_paths_configuration():
    """Test that path configurations are set."""
    assert settings.project_root.exists()
    assert settings.models_cache_dir.name == "models"
    assert settings.data_dir.name == "data"
    assert settings.faiss_index_path.name == "wikipedia.index"
    assert settings.kb_snippets_path.name == "kb_snippets.json"
