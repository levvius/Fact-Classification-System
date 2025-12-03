import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_root_endpoint(test_client):
    """Test that root endpoint returns redirect or info."""
    response = test_client.get("/")

    # FastAPI root can return 200 with info or 307 redirect to /docs
    assert response.status_code in [200, 307], f"Expected 200 or 307, got {response.status_code}"


@pytest.mark.integration
def test_health_endpoint(test_client):
    """Test that health endpoint returns healthy status."""
    response = test_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.integration
@pytest.mark.slow
def test_classify_endpoint_success(test_client, real_model_manager):
    """Test successful classification with valid text."""
    payload = {
        "text": "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He is known for his theory of relativity."
    }

    response = test_client.post("/api/v1/classify", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "overall_classification" in data
    assert "confidence" in data
    assert "claims" in data

    # Verify valid classification
    assert data["overall_classification"] in ["правда", "неправда", "нейтрально"]

    # Verify confidence is a valid float
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0

    # Verify claims structure
    assert isinstance(data["claims"], list)
    assert len(data["claims"]) >= 1

    for claim in data["claims"]:
        assert "claim" in claim
        assert "classification" in claim
        assert "confidence" in claim
        assert "best_evidence" in claim


@pytest.mark.integration
def test_classify_validation_error_short_text(test_client):
    """Test that short text (< 3 words) returns validation error."""
    payload = {"text": "Hi there"}

    response = test_client.post("/api/v1/classify", json=payload)

    # Should return 422 Validation Error
    assert response.status_code == 422
    data = response.json()

    # FastAPI validation error structure
    assert "detail" in data


@pytest.mark.integration
def test_classify_validation_error_xss_attack(test_client):
    """Test that XSS payload returns validation error."""
    payload = {"text": "<script>alert('XSS attack')</script> This is a test."}

    response = test_client.post("/api/v1/classify", json=payload)

    # Should return 422 Validation Error due to XSS pattern detection
    assert response.status_code == 422
    data = response.json()

    # FastAPI validation error structure
    assert "detail" in data


@pytest.mark.integration
def test_classify_validation_error_empty_text(test_client):
    """Test that empty text returns validation error."""
    payload = {"text": ""}

    response = test_client.post("/api/v1/classify", json=payload)

    # Should return 422 Validation Error
    assert response.status_code == 422
    data = response.json()

    # FastAPI validation error structure
    assert "detail" in data


@pytest.mark.integration
def test_classify_validation_error_missing_text(test_client):
    """Test that missing 'text' field returns validation error."""
    payload = {}

    response = test_client.post("/api/v1/classify", json=payload)

    # Should return 422 Validation Error
    assert response.status_code == 422
    data = response.json()

    # FastAPI validation error structure
    assert "detail" in data


@pytest.mark.integration
def test_cache_info_endpoint(test_client):
    """Test that cache-info endpoint returns cache statistics."""
    response = test_client.get("/cache-info")

    assert response.status_code == 200
    data = response.json()

    # Verify cache info structure
    assert "size" in data
    assert "maxsize" in data

    # Verify values are integers
    assert isinstance(data["size"], int)
    assert isinstance(data["maxsize"], int)

    # Verify maxsize is 100 (as configured)
    assert data["maxsize"] == 100

    # Size should be non-negative and <= maxsize
    assert 0 <= data["size"] <= data["maxsize"]


@pytest.mark.integration
@pytest.mark.slow
def test_classify_endpoint_caching(test_client, real_model_manager):
    """Test that classification results are cached for identical requests."""
    payload = {
        "text": "Albert Einstein was born in 1879 in Germany. This is a factual statement."
    }

    # First request - should hit the model
    response1 = test_client.post("/api/v1/classify", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()

    # Second identical request - should use cache
    response2 = test_client.post("/api/v1/classify", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()

    # Results should be identical
    assert data1["overall_classification"] == data2["overall_classification"]
    assert data1["confidence"] == data2["confidence"]
    assert len(data1["claims"]) == len(data2["claims"])


@pytest.mark.integration
@pytest.mark.slow
def test_classify_endpoint_different_texts(test_client, real_model_manager):
    """Test that different texts produce different classifications."""
    payload1 = {
        "text": "Albert Einstein was born on March 14, 1879, in Germany."
    }
    payload2 = {
        "text": "Python is a statically typed programming language."
    }

    response1 = test_client.post("/api/v1/classify", json=payload1)
    response2 = test_client.post("/api/v1/classify", json=payload2)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # First should be truth, second should be falsehood
    assert data1["overall_classification"] == "правда"
    assert data2["overall_classification"] == "неправда"


@pytest.mark.integration
def test_docs_endpoint(test_client):
    """Test that Swagger docs endpoint is accessible."""
    response = test_client.get("/docs")

    # Should return 200 with HTML
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
