from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CRSettings(BaseModel):
    on: bool = Field(default=False, description="Enable hierarchical CR search")
    d1: int = 256
    d2: int = 64
    k2: int = 16
    k1_per_bucket: int = 12
    top_coarse: int = 3
    top_topics_per_coarse: int = 2
    max_candidates: int = 256
    index_npz_path: str = "cr_index.npz"
    index_meta_path: str = "cr_index.meta.json"


class Settings(BaseSettings):
    # Scoring weights
    weight_dense: float = 1.0
    weight_narrative: float = 0.6
    weight_pheromone: float = 0.3
    weight_diversity: float = 0.2  # used in MMR
    epsilon_greedy: float = 0.05
    mmr_lambda_default: float = 0.5
    deterministic: bool = False
    deterministic_seed: int = 1337

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
    storage_persistence_enabled: bool = True
    storage_retention_days: float | None = None
    storage_retention_sweep_interval_s: float = 0.0  # disabled if <=0
    storage_retention_sweep_on_start: bool = True
    narrative_store_path: str = "narrative.json"
    pheromone_store_path: str = "pheromones.json"

    # --- Cognitive Gating (AUTO-Gate) ---
    gating_mode: Literal["off", "auto", "on"] = "auto"
    gating_threshold: float = 0.7
    gating_min_candidates: int = 100
    gating_max_candidates: int = 400
    gating_entropy_temp: float = 1.0

    cr: CRSettings = CRSettings()

    # Multi-tenancy / Namespace isolation
    namespace_header: str = Field(
        default="X-NeuralCache-Namespace",
        description="HTTP header clients can set to select a namespace (tenant)"
    )
    default_namespace: str = Field(
        default="default", description="Namespace used when none provided"
    )
    namespace_pattern: str = Field(
        default=r"^[a-zA-Z0-9_.-]{1,64}$",
        description="Regex pattern namespace names must match"
    )

    model_config = SettingsConfigDict(env_prefix="NEURALCACHE_", env_file=".env", extra="ignore")
