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

    # Pheromone
    pheromone_decay_half_life_s: float = 1800.0  # 30min half-life
    pheromone_exposure_penalty: float = 0.1

    # API
    api_title: str = "NeuralCache API"
    api_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_prefix="NEURALCACHE_", env_file=".env", extra="ignore")
