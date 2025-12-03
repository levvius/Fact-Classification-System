import pytest
from unittest.mock import Mock
from app.services.nli_verifier import nli_score
from app.core.exceptions import NLIVerificationException


@pytest.mark.unit
def test_nli_score_returns_float(mock_model_manager):
    """Test that nli_score returns a float value."""
    claim = "Albert Einstein was born in 1879."
    evidence = "Albert Einstein was born on March 14, 1879, in Ulm, Germany."

    score = nli_score(claim, evidence)

    assert isinstance(score, float)


@pytest.mark.unit
def test_nli_score_range(mock_model_manager):
    """Test that nli_score returns value in range [0.0, 1.0]."""
    claim = "Python is a programming language."
    evidence = "Python is a dynamically typed, high-level programming language."

    score = nli_score(claim, evidence)

    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_with_entailment(mock_model_manager):
    """Test nli_score with entailing evidence."""
    # Mock returns ENTAILMENT with score 0.92
    claim = "Einstein was born in Germany."
    evidence = "Albert Einstein was born in Ulm, Germany."

    score = nli_score(claim, evidence)

    # Mock returns 0.92
    assert score > 0.0
    assert isinstance(score, float)


@pytest.mark.unit
def test_nli_score_formats_input_correctly(mock_model_manager):
    """Test that nli_score formats input with </s></s> separator."""
    claim = "Test claim."
    evidence = "Test evidence."

    nli_pipeline = mock_model_manager.get_nli()

    score = nli_score(claim, evidence)

    # Verify NLI pipeline was called
    nli_pipeline.assert_called_once()

    # Check the input format contains the separator
    call_args = nli_pipeline.call_args[0][0]
    assert "</s></s>" in call_args
    assert evidence in call_args
    assert claim in call_args


@pytest.mark.unit
def test_nli_score_with_different_claims(mock_model_manager):
    """Test nli_score with various claim-evidence pairs."""
    pairs = [
        ("Einstein was born in 1879.", "Albert Einstein was born on March 14, 1879."),
        ("Python is a language.", "Python is a programming language."),
        ("Water boils at 100°C.", "Water boils at 100 degrees Celsius at standard pressure.")
    ]

    for claim, evidence in pairs:
        score = nli_score(claim, evidence)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_empty_claim(mock_model_manager):
    """Test nli_score with empty claim."""
    claim = ""
    evidence = "Some evidence text."

    score = nli_score(claim, evidence)

    # Should return a valid score even with empty input
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_empty_evidence(mock_model_manager):
    """Test nli_score with empty evidence."""
    claim = "Some claim text."
    evidence = ""

    score = nli_score(claim, evidence)

    # Should return a valid score even with empty evidence
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_long_texts(mock_model_manager):
    """Test nli_score with long claim and evidence texts."""
    claim = " ".join(["This is a very long claim."] * 50)
    evidence = " ".join(["This is very long evidence."] * 50)

    score = nli_score(claim, evidence)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_special_characters(mock_model_manager):
    """Test nli_score with special characters in text."""
    claim = "Einstein's E=mc² equation is famous."
    evidence = "Albert Einstein's equation E=mc² relates mass & energy."

    score = nli_score(claim, evidence)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_nli_score_calls_model_manager(mock_model_manager):
    """Test that nli_score calls ModelManager.get_nli()."""
    claim = "Test claim."
    evidence = "Test evidence."

    score = nli_score(claim, evidence)

    # Verify get_nli was called
    mock_model_manager.get_nli.assert_called()


@pytest.mark.unit
def test_nli_score_extracts_entailment_label(mock_model_manager):
    """Test that nli_score correctly extracts ENTAILMENT score from result."""
    claim = "Python is a language."
    evidence = "Python is a programming language."

    # Mock NLI pipeline returns ENTAILMENT with 0.92 score
    nli_pipeline = mock_model_manager.get_nli()
    nli_pipeline.return_value = [{"label": "ENTAILMENT", "score": 0.92}]

    score = nli_score(claim, evidence)

    # Should extract the 0.92 score
    assert score == 0.92


@pytest.mark.unit
def test_nli_score_handles_contradiction_label(mock_model_manager):
    """Test that nli_score returns 0.0 when no entailment label found."""
    claim = "Test claim."
    evidence = "Test evidence."

    # Mock NLI pipeline returns CONTRADICTION
    nli_pipeline = mock_model_manager.get_nli()
    nli_pipeline.return_value = [{"label": "CONTRADICTION", "score": 0.95}]

    score = nli_score(claim, evidence)

    # Should return 0.0 when no ENTAILMENT label found
    assert score == 0.0


@pytest.mark.unit
def test_nli_score_handles_neutral_label(mock_model_manager):
    """Test that nli_score returns 0.0 for NEUTRAL label."""
    claim = "Test claim."
    evidence = "Test evidence."

    # Mock NLI pipeline returns NEUTRAL
    nli_pipeline = mock_model_manager.get_nli()
    nli_pipeline.return_value = [{"label": "NEUTRAL", "score": 0.88}]

    score = nli_score(claim, evidence)

    # Should return 0.0 when no ENTAILMENT label found
    assert score == 0.0


@pytest.mark.unit
def test_nli_score_handles_case_insensitive_label(mock_model_manager):
    """Test that nli_score handles case-insensitive label matching."""
    claim = "Test claim."
    evidence = "Test evidence."

    # Mock NLI pipeline returns lowercase "entailment"
    nli_pipeline = mock_model_manager.get_nli()
    nli_pipeline.return_value = [{"label": "entailment", "score": 0.85}]

    score = nli_score(claim, evidence)

    # Should match case-insensitively and extract score
    assert score == 0.85


@pytest.mark.unit
def test_nli_score_multiple_labels_picks_entailment(mock_model_manager):
    """Test that nli_score picks ENTAILMENT when multiple labels present."""
    claim = "Test claim."
    evidence = "Test evidence."

    # Mock NLI pipeline returns multiple labels
    nli_pipeline = mock_model_manager.get_nli()
    nli_pipeline.return_value = [
        {"label": "CONTRADICTION", "score": 0.02},
        {"label": "NEUTRAL", "score": 0.05},
        {"label": "ENTAILMENT", "score": 0.93}
    ]

    score = nli_score(claim, evidence)

    # Should find and extract ENTAILMENT score
    assert score == 0.93


@pytest.mark.unit
def test_nli_score_premise_hypothesis_order(mock_model_manager):
    """Test that nli_score uses correct premise-hypothesis order."""
    claim = "Claim text."
    evidence = "Evidence text."

    nli_pipeline = mock_model_manager.get_nli()

    score = nli_score(claim, evidence)

    # Verify the input format: premise </s></s> hypothesis
    call_args = nli_pipeline.call_args[0][0]
    # Evidence (premise) should come before claim (hypothesis)
    evidence_pos = call_args.find(evidence)
    claim_pos = call_args.find(claim)
    assert evidence_pos < claim_pos
