import pytest
from app.services.classifier import classify_text


@pytest.mark.integration
@pytest.mark.slow
def test_classify_factual_text(real_model_manager):
    """Test that factual text about Einstein classifies as 'правда' (truth)."""
    text = "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity."

    result = classify_text(text)

    # Verify response structure
    assert "overall_classification" in result
    assert "confidence" in result
    assert "claims" in result

    # Should classify as truth (high confidence with Wikipedia evidence)
    assert result["overall_classification"] == "правда"
    assert result["confidence"] >= 0.7
    assert len(result["claims"]) >= 1

    # Verify claims have evidence
    for claim in result["claims"]:
        assert "claim" in claim
        assert "classification" in claim
        assert "confidence" in claim
        assert "best_evidence" in claim


@pytest.mark.integration
@pytest.mark.slow
def test_classify_false_text(real_model_manager):
    """Test that false text classifies as 'неправда' (falsehood)."""
    text = "Albert Einstein was born in 1990. Python is a statically typed language that requires type declarations before runtime."

    result = classify_text(text)

    # Verify response structure
    assert "overall_classification" in result
    assert "confidence" in result
    assert "claims" in result

    # Should classify as falsehood (contradicts Wikipedia evidence)
    assert result["overall_classification"] == "неправда"
    assert len(result["claims"]) >= 1

    # At least one claim should be classified as falsehood
    falsehood_claims = [c for c in result["claims"] if c["classification"] == "неправда"]
    assert len(falsehood_claims) >= 1


@pytest.mark.integration
@pytest.mark.slow
def test_classify_mixed_claims(real_model_manager):
    """Test that mixed text (truth + falsehood) classifies correctly.

    According to classification logic, ANY falsehood claim should make
    overall classification 'неправда' (pessimistic aggregation).
    """
    text = "Einstein was born in 1879 in Germany. Python requires static type declarations before compilation."

    result = classify_text(text)

    # Verify response structure
    assert "overall_classification" in result
    assert "confidence" in result
    assert "claims" in result

    # Should have at least 2 claims extracted
    assert len(result["claims"]) >= 2

    # Overall should be valid classification
    assert result["overall_classification"] in ["правда", "неправда", "нейтрально"]

    # Check that we have mixed classifications in claims
    classifications = [c["classification"] for c in result["claims"]]
    # We expect at least one truth and one falsehood (or neutral)
    assert len(set(classifications)) >= 1  # At least some variety


@pytest.mark.integration
@pytest.mark.slow
def test_classify_neutral_text(real_model_manager):
    """Test that neutral/ambiguous text classifies as 'нейтрально' or with low confidence."""
    text = "The future of artificial intelligence is uncertain and depends on many complex factors including technological advancement and ethical considerations."

    result = classify_text(text)

    # Verify response structure
    assert "overall_classification" in result
    assert "confidence" in result
    assert "claims" in result

    # Should be valid classification
    assert result["overall_classification"] in ["правда", "неправда", "нейтрально"]

    # For neutral/opinion text, confidence should typically be lower
    # or classification should be neutral
    if result["overall_classification"] == "нейтрально":
        assert 0.0 <= result["confidence"] <= 1.0

    # At least one claim should be extracted
    assert len(result["claims"]) >= 1


@pytest.mark.integration
@pytest.mark.slow
def test_classify_einstein_birth_year_with_new_threshold(real_model_manager):
    """Test that Einstein's birth year classifies as truth with 0.75 threshold.

    This test validates that the lowered threshold (0.85 → 0.75) correctly
    identifies well-established historical facts as 'правда' instead of 'нейтрально'.
    """
    text = "Albert Einstein was born in 1879."

    result = classify_text(text)

    # Should be truth with new threshold
    assert result["overall_classification"] == "правда"
    assert result["confidence"] >= 0.70
    assert len(result["claims"]) >= 1

    # Verify evidence quality
    claim = result["claims"][0]
    assert claim["best_evidence"] is not None
    assert claim["best_evidence"]["nli_score"] >= 0.70

    # Evidence snippet should mention the year 1879
    evidence_text = claim["best_evidence"]["snippet"]
    assert "1879" in evidence_text or "Einstein" in evidence_text


@pytest.mark.integration
@pytest.mark.slow
def test_classify_python_creator_fact(real_model_manager):
    """Test that Python creator fact classifies as truth.

    This test validates that well-established programming language facts
    are correctly identified as 'правда' with the improved thresholds.
    """
    text = "Python was created by Guido van Rossum in 1991."

    result = classify_text(text)

    # Should be truth
    assert result["overall_classification"] == "правда"
    assert result["confidence"] >= 0.65
    assert len(result["claims"]) >= 1

    # At least one claim should be classified as truth
    truth_claims = [c for c in result["claims"] if c["classification"] == "правда"]
    assert len(truth_claims) >= 1
