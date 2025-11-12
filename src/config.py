from typing import List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data loading and processing configuration."""
    
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    features_cache_dir: Path = Field(default=Path("data/features"))
    train_cutoff_date: int = Field(default=20101231, description="YYYYMMDD format")
    
    @field_validator("raw_data_dir", "processed_data_dir", "features_cache_dir")
    @classmethod
    def create_dir_if_not_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    lag_windows: List[int] = Field(default=[3, 7, 14, 30])
    return_features: List[str] = Field(
        default=[
            "returns_close_prev_mktres_10",
            "returns_close_prev_raw_10",
            "returns_open_prev_mktres_1",
            "returns_open_prev_raw_1",
            "open",
            "close"
        ]
    )


class ModelConfig(BaseModel):
    """Model training configuration."""
    
    model_type: str = Field(default="lightgbm")
    objective: str = Field(default="binary")
    learning_rate: float = Field(default=0.1, ge=0.001, le=0.5)
    num_leaves: int = Field(default=100, ge=20, le=5000)
    min_data_in_leaf: int = Field(default=100, ge=10, le=10000)
    num_iterations: int = Field(default=300, ge=50, le=2000)
    max_depth: int = Field(default=-1)
    early_stopping_rounds: int = Field(default=50)
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    train_window_years: int = Field(default=3, ge=1, le=10)
    test_window_months: int = Field(default=6, ge=1, le=24)
    commission_bps: float = Field(default=1.0, ge=0.0, le=100.0)
    spread_bps: float = Field(default=5.0, ge=0.0, le=100.0)
    market_impact_coef: float = Field(default=0.1, ge=0.0, le=1.0)


class Config(BaseModel):
    """Master configuration."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    log_level: str = Field(default="INFO")
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


config = Config()
