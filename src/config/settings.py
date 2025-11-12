from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, Field, validator, field_validator
from loguru import logger


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    
    data_dir: str = Field(default="data/raw", description="Directory containing raw data")
    cache_dir: str = Field(default="data/processed", description="Directory for cached data")
    train_start_date: str = Field(default="2023-01-01", description="Training start date")
    train_end_date: str = Field(default="2023-12-31", description="Training end date")
    test_start_date: str = Field(default="2024-01-01", description="Test start date")
    test_end_date: str = Field(default="2024-12-31", description="Test end date")
    min_trading_days: int = Field(default=100, description="Minimum trading days required")
    
    @field_validator('data_dir', 'cache_dir')
    @classmethod
    def validate_paths(cls, v):
        """Ensure paths exist or can be created."""
        path = Path(v)
        if not path.exists():
            logger.warning(f"Path {v} does not exist, will be created")
        return v


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    # Lag features
    lag_windows: List[int] = Field(
        default=[3, 7, 14, 30],
        description="Lag windows in days"
    )
    lag_features: List[str] = Field(
        default=[
            'returnsClosePrevRaw1',
            'returnsOpenPrevRaw1',
            'returnsClosePrevMktres1',
            'returnsOpenPrevMktres1'
        ],
        description="Features to create lags for"
    )
    
    # News features
    use_news: bool = Field(default=True, description="Whether to use news features")
    news_decay_half_life: float = Field(
        default=24.0,
        description="Half-life for news decay in hours"
    )
    news_min_articles: int = Field(
        default=1,
        description="Minimum articles for reliable statistics"
    )
    
    @field_validator('lag_windows')
    @classmethod
    def validate_lag_windows(cls, v):
        """Ensure lag windows are positive and sorted."""
        if not all(w > 0 for w in v):
            raise ValueError("All lag windows must be positive")
        return sorted(v)


class ModelConfig(BaseModel):
    """Model training configuration."""
    
    model_type: str = Field(default="lgbm", description="Model type (lgbm, xgboost, etc)")
    
    # LightGBM parameters
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    num_leaves: int = Field(default=100, ge=20, le=2000)
    min_data_in_leaf: int = Field(default=100, ge=10, le=10000)
    num_iterations: int = Field(default=300, ge=10, le=10000)
    max_bin: int = Field(default=255, ge=63, le=1023)
    feature_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    bagging_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    bagging_freq: int = Field(default=5, ge=0, le=100)
    
    # Training parameters
    early_stopping_rounds: int = Field(default=50, description="Early stopping rounds")
    validation_fraction: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of training data for validation"
    )
    random_state: int = Field(default=42, description="Random seed")
    
    def to_lgbm_params(self) -> Dict[str, Any]:
        """Convert to LightGBM parameter dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_data_in_leaf': self.min_data_in_leaf,
            'num_iterations': self.num_iterations,
            'max_bin': self.max_bin,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'random_state': self.random_state,
            'verbose': -1
        }


class OptimizationConfig(BaseModel):
    """Hyperparameter optimization configuration."""
    
    enabled: bool = Field(default=False, description="Whether to run optimization")
    n_trials: int = Field(default=100, ge=10, le=1000)
    cv_folds: int = Field(default=3, ge=2, le=10)
    optimization_metric: str = Field(default="log_loss", description="Metric to optimize")
    direction: str = Field(default="minimize", description="Optimization direction")
    n_jobs: int = Field(default=1, description="Number of parallel jobs")
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        """Ensure direction is valid."""
        if v not in ['minimize', 'maximize']:
            raise ValueError("Direction must be 'minimize' or 'maximize'")
        return v


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    # Walk-forward parameters
    train_window_years: float = Field(default=1.0, ge=0.25, le=10.0)
    test_window_months: int = Field(default=3, ge=1, le=12)
    step_months: Optional[int] = Field(default=None, description="Step size (defaults to test window)")
    
    # Transaction costs
    commission_bps: float = Field(default=1.0, ge=0, le=100)
    spread_bps: float = Field(default=5.0, ge=0, le=100)
    market_impact_coef: float = Field(default=0.1, ge=0, le=10)
    
    # Risk parameters
    risk_free_rate: float = Field(default=0.02, ge=0, le=0.2)
    periods_per_year: int = Field(default=252, ge=1, le=365)


class Config(BaseModel):
    """Main configuration container."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Global settings
    output_dir: str = Field(default="outputs", description="Directory for outputs")
    log_level: str = Field(default="INFO", description="Logging level")
    seed: int = Field(default=42, description="Global random seed")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


def load_config(path: str) -> Config:
    """
    Load configuration from YAML or JSON file.
    
    Parameters
    ----------
    path : str
        Path to configuration file
        
    Returns
    -------
    Config
        Loaded and validated configuration
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    logger.info(f"Loading configuration from {path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    config = Config(**config_dict)
    
    logger.info("Configuration loaded and validated successfully")
    
    return config


def save_config(config: Config, path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Parameters
    ----------
    config : Config
        Configuration to save
    path : str
        Path to save configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.model_dump()
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Configuration saved to {path}")


def create_default_config() -> Config:
    """
    Create default configuration.
    
    Returns
    -------
    Config
        Default configuration
    """
    return Config()


def merge_configs(base: Config, override: Dict[str, Any]) -> Config:
    """
    Merge override values into base configuration.
    
    Parameters
    ----------
    base : Config
        Base configuration
    override : Dict[str, Any]
        Override values
        
    Returns
    -------
    Config
        Merged configuration
    """
    base_dict = base.model_dump()
    
    def deep_update(d: dict, u: dict) -> dict:
        """Deep merge two dictionaries."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    merged_dict = deep_update(base_dict, override)
    
    return Config(**merged_dict)
