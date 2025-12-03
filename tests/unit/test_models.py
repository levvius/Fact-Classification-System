import pytest
from unittest.mock import Mock, patch
import numpy as np
from app.core.models import ModelManager
from app.core.exceptions import ModelNotLoadedException


@pytest.mark.unit
def test_singleton_pattern():
    """Test that ModelManager implements singleton pattern correctly."""
    # Reset singleton
    ModelManager._instance = None

    instance1 = ModelManager.get_instance()
    instance2 = ModelManager.get_instance()

    assert instance1 is instance2
    assert id(instance1) == id(instance2)

    # Cleanup
    ModelManager._instance = None


@pytest.mark.unit
def test_get_embed_model_without_load_raises_exception():
    """Test that get_embed_model raises ModelNotLoadedException when models not loaded."""
    # Reset singleton
    ModelManager._instance = None

    mm = ModelManager.get_instance()

    with pytest.raises(ModelNotLoadedException) as exc_info:
        mm.get_embed_model()

    assert "Models not loaded" in str(exc_info.value)

    # Cleanup
    ModelManager._instance = None


@pytest.mark.unit
def test_get_nli_without_load_raises_exception():
    """Test that get_nli raises ModelNotLoadedException when models not loaded."""
    # Reset singleton
    ModelManager._instance = None

    mm = ModelManager.get_instance()

    with pytest.raises(ModelNotLoadedException) as exc_info:
        mm.get_nli()

    assert "Models not loaded" in str(exc_info.value) or "NLI model not loaded" in str(exc_info.value)

    # Cleanup
    ModelManager._instance = None


@pytest.mark.unit
def test_get_index_without_load_raises_exception():
    """Test that get_index raises ModelNotLoadedException when models not loaded."""
    # Reset singleton
    ModelManager._instance = None

    mm = ModelManager.get_instance()

    with pytest.raises(ModelNotLoadedException) as exc_info:
        mm.get_index()

    assert "Models not loaded" in str(exc_info.value) or "FAISS index not loaded" in str(exc_info.value)

    # Cleanup
    ModelManager._instance = None


@pytest.mark.unit
def test_get_snippets_without_load_raises_exception():
    """Test that get_snippets raises ModelNotLoadedException when models not loaded."""
    # Reset singleton
    ModelManager._instance = None

    mm = ModelManager.get_instance()

    with pytest.raises(ModelNotLoadedException) as exc_info:
        mm.get_snippets()

    assert "Models not loaded" in str(exc_info.value) or "KB snippets not loaded" in str(exc_info.value)

    # Cleanup
    ModelManager._instance = None


@pytest.mark.unit
def test_model_manager_with_mocked_models(mock_model_manager):
    """Test that ModelManager works correctly with mocked models."""
    # mock_model_manager fixture already has models loaded

    # Test that we can retrieve models
    embed_model = mock_model_manager.get_embed_model()
    nli_pipeline = mock_model_manager.get_nli()
    faiss_index = mock_model_manager.get_index()
    kb_snippets = mock_model_manager.get_snippets()

    assert embed_model is not None
    assert nli_pipeline is not None
    assert faiss_index is not None
    assert kb_snippets is not None
    assert len(kb_snippets) == 6


@pytest.mark.unit
def test_embed_model_encode(mock_model_manager):
    """Test that embed model returns correct shape embeddings."""
    embed_model = mock_model_manager.get_embed_model()

    # Mock returns (1, 384) shape
    embeddings = embed_model.encode("Test text")

    assert embeddings.shape == (1, 384)
    assert embeddings.dtype == np.float32


@pytest.mark.unit
def test_nli_pipeline_output_format(mock_model_manager):
    """Test that NLI pipeline returns expected output format."""
    nli_pipeline = mock_model_manager.get_nli()

    result = nli_pipeline({"text": "claim", "text_pair": "evidence"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "label" in result[0]
    assert "score" in result[0]
    assert result[0]["label"] == "ENTAILMENT"
    assert 0.0 <= result[0]["score"] <= 1.0


@pytest.mark.unit
def test_faiss_index_search_format(mock_model_manager):
    """Test that FAISS index search returns correct format."""
    faiss_index = mock_model_manager.get_index()

    # Create dummy query vector (384 dimensions)
    query = np.random.rand(1, 384).astype(np.float32)

    distances, indices = faiss_index.search(query, k=6)

    assert distances.shape == (1, 6)
    assert indices.shape == (1, 6)
    assert len(indices[0]) == 6


@pytest.mark.unit
def test_kb_snippets_structure(mock_model_manager):
    """Test that knowledge base snippets have correct structure."""
    kb_snippets = mock_model_manager.get_snippets()

    assert len(kb_snippets) == 6

    for snippet in kb_snippets:
        assert "snippet" in snippet
        assert "source" in snippet
        assert isinstance(snippet["snippet"], str)
        assert isinstance(snippet["source"], str)
        assert snippet["source"].startswith("https://")


@pytest.mark.unit
def test_multiple_model_manager_calls_return_same_models(mock_model_manager):
    """Test that multiple calls to get_* methods return the same model instances."""
    embed_model1 = mock_model_manager.get_embed_model()
    embed_model2 = mock_model_manager.get_embed_model()

    nli_pipeline1 = mock_model_manager.get_nli()
    nli_pipeline2 = mock_model_manager.get_nli()

    assert embed_model1 is embed_model2
    assert nli_pipeline1 is nli_pipeline2
