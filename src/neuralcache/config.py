from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Scoring weights
    weight_dense: float = 1.0
    weight_narrative: float = 0.6
    weight_pheromone: float = 0.3
    weight_diversity: float = 0.2  # used in MMR
    epsilon_greedy: float = 0.05

    # Narrative
    narrative_dim: int = 768
    narrative_ema_alpha: float = 0.01
    narrative_success_gate: float = 0.5  # only update if success >= gate

    # Embeddings
    embedding_backend: str = "hash"
    embedding_model: str | None = None

    # Pheromone
    pheromone_decay_half_life_s: float = 1800.0  # 30min half-life
    pheromone_exposure_penalty: float = 0.1

    # API
    api_title: str = "NeuralCache API"
    api_version: str = "0.2.0"
    max_top_k: int = 100
    max_documents: int = 128
    max_text_length: int = 8192
    max_batch_size: int = 16
    feedback_cache_size: int = 1024
    api_tokens: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int | None = None
    metrics_enabled: bool = True

    # Storage
    storage_backend: str = "sqlite"
    storage_dir: str = "storage"
    storage_db_name: str = "neuralcache.db"
    narrative_store_path: str = "narrative.json"
    pheromone_store_path: str = "pheromones.json"

    model_config = SettingsConfigDict(env_prefix="NEURALCACHE_", env_file=".env", extra="ignore")
