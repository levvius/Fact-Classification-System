import pytest
from unittest.mock import patch, Mock
from app.services.classifier import assess_claim, classify_text
from app.core.exceptions import ClassificationException


@pytest.mark.unit
def test_assess_claim_returns_correct_structure(mock_model_manager):
    """Test that assess_claim returns dict with correct keys."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock retrieve_proofs to return 3 proofs
        mock_retrieve.return_value = [
            {"snippet": "Proof 1", "source": "https://example.com/1", "retrieval_score": 0.1},
            {"snippet": "Proof 2", "source": "https://example.com/2", "retrieval_score": 0.2},
            {"snippet": "Proof 3", "source": "https://example.com/3", "retrieval_score": 0.3}
        ]

        # Mock nli_score to return different scores
        mock_nli.side_effect = [0.92, 0.78, 0.65]

        result = assess_claim("Test claim")

        assert "claim" in result
        assert "support" in result
        assert "best_proof" in result
        assert "all_proofs" in result
        assert result["claim"] == "Test claim"
        assert isinstance(result["support"], float)


@pytest.mark.unit
def test_assess_claim_high_support_score(mock_model_manager):
    """Test assess_claim with high NLI scores returns high support."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.95

        result = assess_claim("Test claim")

        # Support should be 0.95 (max of NLI scores)
        assert result["support"] == 0.95


@pytest.mark.unit
def test_assess_claim_selects_best_proof(mock_model_manager):
    """Test that assess_claim selects proof with highest NLI score."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_retrieve.return_value = [
            {"snippet": "Weak evidence", "source": "https://example.com/1", "retrieval_score": 0.1},
            {"snippet": "Strong evidence", "source": "https://example.com/2", "retrieval_score": 0.2},
            {"snippet": "Medium evidence", "source": "https://example.com/3", "retrieval_score": 0.15}
        ]
        # NLI scores: 0.6, 0.9, 0.7
        mock_nli.side_effect = [0.6, 0.9, 0.7]

        result = assess_claim("Test claim")

        # Best proof should be the one with NLI score 0.9
        assert result["best_proof"]["snippet"] == "Strong evidence"
        assert result["best_proof"]["nli_score"] == 0.9


@pytest.mark.unit
def test_assess_claim_uses_default_top_k(mock_model_manager):
    """Test that assess_claim uses settings.top_k_proofs when top_k not specified."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_retrieve.return_value = []
        mock_nli.return_value = 0.5

        assess_claim("Test claim")

        # Should call retrieve_proofs with default top_k (6)
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("top_k") == 6


@pytest.mark.unit
def test_assess_claim_custom_top_k(mock_model_manager):
    """Test assess_claim with custom top_k parameter."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_retrieve.return_value = []
        mock_nli.return_value = 0.5

        assess_claim("Test claim", top_k=3)

        # Should call retrieve_proofs with top_k=3
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs.get("top_k") == 3


@pytest.mark.unit
def test_classify_text_truth_classification(mock_model_manager):
    """Test classify_text with text that should classify as 'правда'."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock one claim
        mock_extract.return_value = ["Einstein was born in 1879."]

        # Mock high NLI score (>= 0.85 = truth threshold)
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.92

        result = classify_text("Einstein was born in 1879.")

        assert result["overall_classification"] == "правда"
        assert result["confidence"] > 0.0
        assert len(result["claims"]) == 1
        assert result["claims"][0]["classification"] == "правда"


@pytest.mark.unit
def test_classify_text_falsehood_classification(mock_model_manager):
    """Test classify_text with text that should classify as 'неправда'."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock one claim
        mock_extract.return_value = ["Einstein was born in 1990."]

        # Mock low NLI score (< 0.4 = falsehood threshold)
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.25

        result = classify_text("Einstein was born in 1990.")

        assert result["overall_classification"] == "неправда"
        assert result["confidence"] > 0.0
        assert len(result["claims"]) == 1
        assert result["claims"][0]["classification"] == "неправда"


@pytest.mark.unit
def test_classify_text_neutral_classification(mock_model_manager):
    """Test classify_text with text that should classify as 'нейтрально'."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock one claim
        mock_extract.return_value = ["The future is uncertain."]

        # Mock medium NLI score (0.4 <= score < 0.85)
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.6

        result = classify_text("The future is uncertain.")

        assert result["overall_classification"] == "нейтрально"
        assert result["confidence"] > 0.0
        assert len(result["claims"]) == 1
        assert result["claims"][0]["classification"] == "нейтрально"


@pytest.mark.unit
def test_classify_text_mixed_claims_falsehood_priority(mock_model_manager):
    """Test that any 'неправда' claim makes overall classification 'неправда'."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock two claims: one true, one false
        mock_extract.return_value = [
            "Einstein was born in 1879.",  # True
            "Python is statically typed."   # False
        ]

        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]

        # First claim: high score (truth), second claim: low score (falsehood)
        mock_nli.side_effect = [0.92, 0.92, 0.15, 0.15]  # 2 calls per claim (retrieve + assess)

        result = classify_text("Mixed text")

        # Overall should be 'неправда' because one claim is false
        assert result["overall_classification"] == "неправда"
        assert len(result["claims"]) == 2


@pytest.mark.unit
def test_classify_text_mixed_claims_neutral_priority(mock_model_manager):
    """Test that 'нейтрально' has priority over 'правда' but not 'неправда'."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock two claims: one true, one neutral
        mock_extract.return_value = [
            "Einstein was born in 1879.",  # True
            "The future is uncertain."     # Neutral
        ]

        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]

        # First claim: high score (truth), second claim: medium score (neutral)
        mock_nli.side_effect = [0.92, 0.92, 0.6, 0.6]

        result = classify_text("Mixed text")

        # Overall should be 'нейтрально' because one claim is neutral
        assert result["overall_classification"] == "нейтрально"


@pytest.mark.unit
def test_classify_text_multiple_truth_claims(mock_model_manager):
    """Test classify_text with all claims being true."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Mock three true claims
        mock_extract.return_value = [
            "Einstein was born in 1879.",
            "Python is dynamically typed.",
            "Earth orbits the Sun."
        ]

        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]

        # All high NLI scores
        mock_nli.return_value = 0.9

        result = classify_text("All true text")

        assert result["overall_classification"] == "правда"
        assert len(result["claims"]) == 3
        assert all(c["classification"] == "правда" for c in result["claims"])


@pytest.mark.unit
def test_classify_text_confidence_calculation(mock_model_manager):
    """Test that confidence is calculated correctly."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_extract.return_value = ["Test claim"]
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.95

        result = classify_text("Test text")

        # Confidence for truth should be the support score (0.95)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == 0.95


@pytest.mark.unit
def test_classify_text_falsehood_confidence(mock_model_manager):
    """Test that falsehood confidence is 1.0 - support."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_extract.return_value = ["False claim"]
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.2  # Low score -> falsehood

        result = classify_text("False text")

        # Confidence for falsehood should be 1.0 - 0.2 = 0.8
        assert result["overall_classification"] == "неправда"
        assert result["claims"][0]["confidence"] == 0.8


@pytest.mark.unit
def test_classify_text_includes_best_evidence(mock_model_manager):
    """Test that each claim includes best evidence information."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_extract.return_value = ["Test claim"]
        mock_retrieve.return_value = [
            {"snippet": "Best evidence", "source": "https://example.com", "retrieval_score": 0.15}
        ]
        mock_nli.return_value = 0.88

        result = classify_text("Test text")

        claim = result["claims"][0]
        assert "best_evidence" in claim
        assert claim["best_evidence"] is not None
        assert claim["best_evidence"]["snippet"] == "Best evidence"
        assert claim["best_evidence"]["source"] == "https://example.com"
        assert claim["best_evidence"]["nli_score"] == 0.88
        assert claim["best_evidence"]["retrieval_score"] == 0.15


@pytest.mark.unit
def test_classify_text_empty_claims_list(mock_model_manager):
    """Test classify_text behavior when no claims are extracted."""
    with patch('app.services.classifier.extract_claims') as mock_extract:

        mock_extract.return_value = []

        # Should handle empty claims list gracefully
        # This might raise an exception or return a default classification
        # depending on implementation
        try:
            result = classify_text("Short text")
            # If it doesn't raise, verify the structure
            assert "overall_classification" in result
            assert "confidence" in result
            assert "claims" in result
        except ClassificationException:
            # Expected behavior - classification requires at least one claim
            pass


@pytest.mark.unit
def test_classify_text_calls_extract_claims(mock_model_manager):
    """Test that classify_text calls extract_claims with the input text."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_extract.return_value = ["Claim"]
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]
        mock_nli.return_value = 0.8

        classify_text("Input text to classify")

        # Verify extract_claims was called with the text
        mock_extract.assert_called_once_with("Input text to classify")


@pytest.mark.unit
def test_assess_claim_aggregates_max_score(mock_model_manager):
    """Test that assess_claim uses max NLI score as support."""
    with patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        mock_retrieve.return_value = [
            {"snippet": "Proof 1", "source": "https://example.com/1", "retrieval_score": 0.1},
            {"snippet": "Proof 2", "source": "https://example.com/2", "retrieval_score": 0.2},
            {"snippet": "Proof 3", "source": "https://example.com/3", "retrieval_score": 0.3}
        ]

        # Different NLI scores, max is 0.85
        mock_nli.side_effect = [0.6, 0.85, 0.7]

        result = assess_claim("Test claim")

        # Support should be the max score (0.85)
        assert result["support"] == 0.85


@pytest.mark.unit
def test_classify_text_overall_confidence_averaging(mock_model_manager):
    """Test that overall confidence is averaged correctly for multiple claims."""
    with patch('app.services.classifier.extract_claims') as mock_extract, \
         patch('app.services.classifier.retrieve_proofs') as mock_retrieve, \
         patch('app.services.classifier.nli_score') as mock_nli:

        # Two true claims with different confidences
        mock_extract.return_value = ["Claim 1", "Claim 2"]
        mock_retrieve.return_value = [
            {"snippet": "Evidence", "source": "https://example.com", "retrieval_score": 0.1}
        ]

        # Scores: 0.90 and 0.88 (both >= 0.85, so both are truth)
        mock_nli.side_effect = [0.90, 0.90, 0.88, 0.88]

        result = classify_text("Two true claims")

        # Overall confidence should be average: (0.90 + 0.88) / 2 = 0.89
        assert result["overall_classification"] == "правда"
        assert result["confidence"] == pytest.approx(0.89, rel=0.01)
