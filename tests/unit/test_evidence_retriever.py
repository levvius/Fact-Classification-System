import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.services.evidence_retriever import retrieve_proofs
from app.core.exceptions import EvidenceRetrievalException


@pytest.mark.unit
def test_retrieve_proofs_returns_correct_count(mock_model_manager):
    """Test that retrieve_proofs returns the requested number of proofs."""
    claim = "Albert Einstein was born in 1879."

    proofs = retrieve_proofs(claim, top_k=6)

    assert len(proofs) == 6


@pytest.mark.unit
def test_retrieve_proofs_uses_default_top_k(mock_model_manager):
    """Test that retrieve_proofs uses settings.top_k_proofs when top_k not specified."""
    claim = "Albert Einstein was born in 1879."

    proofs = retrieve_proofs(claim)

    # Default top_k_proofs is 6
    assert len(proofs) == 6


@pytest.mark.unit
def test_retrieve_proofs_structure(mock_model_manager):
    """Test that each proof has the correct structure."""
    claim = "Albert Einstein was born in 1879."

    proofs = retrieve_proofs(claim, top_k=3)

    assert len(proofs) == 3
    for proof in proofs:
        assert "snippet" in proof
        assert "source" in proof
        assert "retrieval_score" in proof
        assert isinstance(proof["snippet"], str)
        assert isinstance(proof["source"], str)
        assert isinstance(proof["retrieval_score"], float)


@pytest.mark.unit
def test_retrieve_proofs_snippet_content(mock_model_manager):
    """Test that proofs contain actual snippet content from KB."""
    claim = "Albert Einstein was born in 1879."

    proofs = retrieve_proofs(claim, top_k=6)

    # Check that snippets are not empty
    assert all(len(proof["snippet"]) > 0 for proof in proofs)
    # Check that sources are valid URLs
    assert all(proof["source"].startswith("https://") for proof in proofs)


@pytest.mark.unit
def test_retrieve_proofs_retrieval_scores(mock_model_manager):
    """Test that retrieval scores are valid floats."""
    claim = "Python is a programming language."

    proofs = retrieve_proofs(claim, top_k=5)

    # All retrieval scores should be valid floats
    assert all(isinstance(proof["retrieval_score"], float) for proof in proofs)
    # Scores should be non-negative (L2 distance or similarity)
    assert all(proof["retrieval_score"] >= 0 for proof in proofs)


@pytest.mark.unit
def test_retrieve_proofs_custom_top_k(mock_model_manager):
    """Test retrieve_proofs with custom top_k values."""
    claim = "Water boils at 100 degrees Celsius."

    # Test with top_k=1
    proofs_1 = retrieve_proofs(claim, top_k=1)
    assert len(proofs_1) == 1

    # Test with top_k=3
    proofs_3 = retrieve_proofs(claim, top_k=3)
    assert len(proofs_3) == 3

    # Test with top_k=5
    proofs_5 = retrieve_proofs(claim, top_k=5)
    assert len(proofs_5) == 5


@pytest.mark.unit
def test_retrieve_proofs_different_claims(mock_model_manager):
    """Test retrieve_proofs with different claim texts."""
    claims = [
        "Albert Einstein was born in 1879.",
        "Python is a programming language.",
        "Earth orbits the Sun."
    ]

    for claim in claims:
        proofs = retrieve_proofs(claim, top_k=3)
        assert len(proofs) == 3
        assert all("snippet" in p and "source" in p and "retrieval_score" in p for p in proofs)


@pytest.mark.unit
def test_retrieve_proofs_empty_claim(mock_model_manager):
    """Test retrieve_proofs with empty claim string."""
    claim = ""

    # Should not crash, may return proofs based on empty embedding
    proofs = retrieve_proofs(claim, top_k=3)

    # Should still return structured results
    assert isinstance(proofs, list)
    assert len(proofs) == 3


@pytest.mark.unit
def test_retrieve_proofs_long_claim(mock_model_manager):
    """Test retrieve_proofs with very long claim text."""
    claim = " ".join([
        "This is a very long claim that contains multiple sentences and ideas.",
        "It talks about various topics including science, technology, and history.",
        "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "Python is a widely used programming language in data science and web development."
    ])

    proofs = retrieve_proofs(claim, top_k=4)

    assert len(proofs) == 4
    assert all("snippet" in p for p in proofs)


@pytest.mark.unit
def test_retrieve_proofs_special_characters(mock_model_manager):
    """Test retrieve_proofs with claims containing special characters."""
    claim = "Einstein's E=mc² equation relates mass & energy."

    proofs = retrieve_proofs(claim, top_k=3)

    assert len(proofs) == 3
    assert all("snippet" in p and "source" in p for p in proofs)


@pytest.mark.unit
def test_retrieve_proofs_calls_model_manager_methods(mock_model_manager):
    """Test that retrieve_proofs correctly calls ModelManager methods."""
    claim = "Test claim for method verification."

    # Call retrieve_proofs
    proofs = retrieve_proofs(claim, top_k=3)

    # Verify that get_embed_model was called
    mock_model_manager.get_embed_model.assert_called()

    # Verify that get_index was called
    mock_model_manager.get_index.assert_called()

    # Verify that get_snippets was called
    mock_model_manager.get_snippets.assert_called()


@pytest.mark.unit
def test_retrieve_proofs_embedding_encode_called(mock_model_manager):
    """Test that embedding model encode is called with claim."""
    claim = "Albert Einstein was born in 1879."

    embed_model = mock_model_manager.get_embed_model()

    proofs = retrieve_proofs(claim, top_k=3)

    # Verify encode was called with claim
    embed_model.encode.assert_called()
    # Check that claim was passed (might be in a list)
    call_args = embed_model.encode.call_args
    assert claim in str(call_args)


@pytest.mark.unit
def test_retrieve_proofs_faiss_search_called(mock_model_manager):
    """Test that FAISS index search is called."""
    claim = "Python is a programming language."

    faiss_index = mock_model_manager.get_index()

    proofs = retrieve_proofs(claim, top_k=5)

    # Verify search was called with top_k
    faiss_index.search.assert_called()
    call_args = faiss_index.search.call_args
    # Check that top_k was passed
    assert 5 in str(call_args) or call_args[0][1] == 5


@pytest.mark.unit
def test_retrieve_proofs_maps_indices_to_snippets(mock_model_manager):
    """Test that FAISS indices are correctly mapped to KB snippets."""
    claim = "Earth is the third planet from the Sun."

    kb_snippets = mock_model_manager.get_snippets()

    proofs = retrieve_proofs(claim, top_k=6)

    # Verify that returned snippets match KB snippets
    returned_snippets = [p["snippet"] for p in proofs]
    kb_snippet_texts = [s["snippet"] for s in kb_snippets]

    # All returned snippets should be from KB
    assert all(snippet in kb_snippet_texts for snippet in returned_snippets)
