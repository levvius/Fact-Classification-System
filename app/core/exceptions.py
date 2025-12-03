"""
Custom exception classes for the application.

These exceptions provide structured error handling with appropriate HTTP status codes
and detailed error messages for better debugging and user feedback.
"""
from typing import Dict, Any, Optional


class AppBaseException(Exception):
    """
    Base exception class for all application-specific exceptions.

    Attributes:
        message: Human-readable error message
        details: Additional context about the error
        status_code: HTTP status code (default: 500)
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            **self.details
        }


class ModelNotLoadedException(AppBaseException):
    """
    Raised when attempting to use models that haven't been loaded yet.

    HTTP Status: 503 Service Unavailable
    """

    def __init__(self, message: str = "Models are not loaded yet. Please try again later.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=503)


class ClaimExtractionException(AppBaseException):
    """
    Raised when claim extraction from text fails.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, message: str = "Failed to extract claims from text", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=500)


class EvidenceRetrievalException(AppBaseException):
    """
    Raised when FAISS evidence retrieval fails.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, message: str = "Failed to retrieve evidence from knowledge base", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=500)


class NLIVerificationException(AppBaseException):
    """
    Raised when NLI model verification fails.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, message: str = "Failed to verify claim with NLI model", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=500)


class ClassificationException(AppBaseException):
    """
    Raised when the classification pipeline fails.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, message: str = "Failed to classify text", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=500)


class InputValidationException(AppBaseException):
    """
    Raised when input validation fails (e.g., malicious patterns detected).

    HTTP Status: 400 Bad Request
    """

    def __init__(self, message: str = "Input validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=400)


class KnowledgeBaseException(AppBaseException):
    """
    Raised when knowledge base loading or access fails.

    HTTP Status: 503 Service Unavailable
    """

    def __init__(self, message: str = "Failed to load or access knowledge base", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=503)


class CacheException(AppBaseException):
    """
    Raised when cache operations fail.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, message: str = "Cache operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details, status_code=500)
