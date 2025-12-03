import pytest
from app.services.claim_extractor import extract_claims
from app.core.exceptions import ClaimExtractionException


@pytest.mark.unit
def test_extract_simple_claims():
    """Test extraction of simple factual claims."""
    text = "Albert Einstein was born in 1879. He developed the theory of relativity. He won the Nobel Prize in Physics in 1921."

    claims = extract_claims(text)

    assert len(claims) > 0
    assert all(isinstance(claim, str) for claim in claims)
    # All sentences meet min_len and have factual indicators
    assert len(claims) <= 3


@pytest.mark.unit
def test_extract_claims_with_max_claims_limit():
    """Test that max_claims parameter limits the number of returned claims."""
    # Create text with many claims
    text = " ".join([
        "Albert Einstein was born in 1879.",
        "He developed the theory of relativity.",
        "He won the Nobel Prize in Physics in 1921.",
        "Python is a high-level programming language.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "DNA was discovered in 1953.",
        "Artificial intelligence was founded in 1956.",
        "The speed of light is 299,792,458 meters per second.",
        "The capital of France is Paris."
    ])

    claims = extract_claims(text, max_claims=3)

    assert len(claims) == 3
    assert all(isinstance(claim, str) for claim in claims)


@pytest.mark.unit
def test_extract_claims_uses_default_max_claims():
    """Test that extract_claims uses settings.max_claims when max_claims not specified."""
    # Create text with more than 8 claims (default max_claims)
    text = " ".join([
        f"Statement number {i} is factual and contains information."
        for i in range(1, 20)
    ])

    claims = extract_claims(text)

    # Should not exceed default max_claims (8)
    assert len(claims) <= 8


@pytest.mark.unit
def test_extract_claims_empty_text():
    """Test that empty text returns empty list."""
    claims = extract_claims("")

    assert claims == []


@pytest.mark.unit
def test_extract_claims_whitespace_only():
    """Test that whitespace-only text returns empty list."""
    claims = extract_claims("   \n\t   ")

    assert claims == []


@pytest.mark.unit
def test_extract_claims_filters_by_min_len():
    """Test that claims shorter than min_len are filtered out."""
    text = "Short. This is a longer sentence that should be extracted as a claim."

    claims = extract_claims(text, min_len=30)

    # "Short." should be filtered out (less than 30 chars)
    assert len(claims) == 1
    assert len(claims[0]) >= 30


@pytest.mark.unit
def test_extract_claims_with_factual_verbs():
    """Test that sentences with factual verbs are extracted."""
    text = (
        "Albert Einstein was born in Germany. "
        "He developed the theory of relativity. "
        "The discovery was announced in 1905."
    )

    claims = extract_claims(text)

    assert len(claims) > 0
    # All sentences have factual verbs (was, developed, was announced)
    assert all(len(claim) >= 30 for claim in claims)


@pytest.mark.unit
def test_extract_claims_with_digits():
    """Test that sentences with digits are extracted."""
    text = (
        "Python 3.10 was released in 2021. "
        "It includes many new features. "
        "The version numbering follows semantic versioning 2.0."
    )

    claims = extract_claims(text)

    # Should extract sentences with digits
    assert len(claims) > 0
    # At least the sentences with digits should be included
    assert any("3.10" in claim or "2021" in claim or "2.0" in claim for claim in claims)


@pytest.mark.unit
def test_extract_claims_fallback_to_longest():
    """Test fallback to longest sentences when no factual indicators found."""
    # Sentences without factual verbs or digits, but long enough
    text = (
        "The beautiful sunset over the ocean creates wonderful memories "
        "and feelings of peace and tranquility. "
        "Short one. "
        "Another long descriptive sentence about the wonderful nature "
        "and amazing landscapes around the world."
    )

    claims = extract_claims(text, min_len=50)

    # Should fall back to longest sentences
    assert len(claims) > 0
    # All claims should meet min_len
    assert all(len(claim) >= 50 for claim in claims)


@pytest.mark.unit
def test_extract_claims_preserves_sentence_content():
    """Test that extracted claims preserve original sentence content."""
    text = "Albert Einstein was born on March 14, 1879, in Ulm, Germany."

    claims = extract_claims(text)

    assert len(claims) == 1
    # Should preserve the full sentence (after strip)
    assert "Albert Einstein" in claims[0]
    assert "1879" in claims[0]
    assert "Germany" in claims[0]


@pytest.mark.unit
def test_extract_claims_mixed_content():
    """Test extraction from text with mixed factual and non-factual content."""
    text = (
        "This is an opinion about something. "
        "Albert Einstein was born in 1879. "
        "I think this is interesting. "
        "Python is a programming language. "
        "Maybe we should consider this."
    )

    claims = extract_claims(text)

    # Should extract factual claims
    assert len(claims) > 0
    # Factual claims should be prioritized
    assert any("Einstein" in claim or "1879" in claim for claim in claims)


@pytest.mark.unit
def test_extract_claims_multiple_sentences_per_claim():
    """Test that each sentence is treated as a separate claim."""
    text = "Einstein was born in 1879. He won the Nobel Prize in 1921."

    claims = extract_claims(text)

    # Each sentence should be extracted separately
    assert len(claims) == 2
    assert any("1879" in claim for claim in claims)
    assert any("1921" in claim for claim in claims)


@pytest.mark.unit
def test_extract_claims_custom_min_len():
    """Test extract_claims with custom min_len parameter."""
    text = (
        "Short text here with a number 5. "
        "This is a much longer sentence that contains more information and details."
    )

    # With high min_len, only long sentence should be extracted
    claims = extract_claims(text, min_len=60)

    assert len(claims) == 1
    assert len(claims[0]) >= 60


@pytest.mark.unit
def test_extract_claims_ordering():
    """Test that claims are returned in order of extraction."""
    text = "First sentence with year 2020. Second sentence with year 2021. Third sentence with year 2022."

    claims = extract_claims(text)

    # Claims should be in original order
    assert len(claims) == 3
    assert "2020" in claims[0]
    assert "2021" in claims[1]
    assert "2022" in claims[2]
