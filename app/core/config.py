import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "roberta-large-mnli"
    device: str = "cpu"  # or "cuda" for GPU

    # Classification Thresholds
    truth_threshold: float = 0.75  # support >= 0.75 -> "правда" (lowered from 0.85)
    falsehood_threshold: float = 0.4  # support < 0.4 -> "неправда"

    # Evidence Retrieval
    top_k_proofs: int = 10  # increased from 6 for better evidence quality
    max_claims: int = 8

    # Aggregation Strategy
    use_weighted_aggregation: bool = True  # weighted voting vs pessimistic aggregation
    neutral_vote_weight: float = 0.5  # weight multiplier for neutral claims (0.0-1.0)

    # NLI Configuration
    use_nli_context: bool = True  # add "Established fact:" prefix to guide NLI model

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    models_cache_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"
    faiss_index_path: Path = data_dir / "faiss_index" / "wikipedia.index"
    kb_snippets_path: Path = data_dir / "kb_snippets.json"

    # Wikipedia KB
    max_sentences_per_page: int = 15

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 10  # requests per minute
    rate_limit_burst: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
