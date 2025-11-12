import pytest
import tempfile
from pathlib import Path
import yaml
import json

from src.config.settings import (
    Config,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    BacktestConfig,
    OptimizationConfig,
    load_config,
    save_config,
    create_default_config,
    merge_configs
)


class TestDataConfig:
    """Test data configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        
        assert config.data_dir == "data/raw"
        assert config.cache_dir == "data/processed"
        assert config.min_trading_days == 100
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            data_dir="custom/data",
            min_trading_days=200
        )
        
        assert config.data_dir == "custom/data"
        assert config.min_trading_days == 200


class TestFeatureConfig:
    """Test feature configuration."""
    
    def test_default_lag_windows(self):
        """Test default lag windows."""
        config = FeatureConfig()
        
        assert config.lag_windows == [3, 7, 14, 30]
        assert config.use_news == True
    
    def test_lag_window_sorting(self):
        """Test lag windows are sorted."""
        config = FeatureConfig(lag_windows=[30, 7, 14, 3])
        
        assert config.lag_windows == [3, 7, 14, 30]
    
    def test_invalid_lag_windows(self):
        """Test validation of negative lag windows."""
        with pytest.raises(ValueError):
            FeatureConfig(lag_windows=[3, -7, 14])


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_parameters(self):
        """Test default model parameters."""
        config = ModelConfig()
        
        assert config.learning_rate == 0.1
        assert config.num_leaves == 100
        assert config.num_iterations == 300
    
    def test_parameter_bounds(self):
        """Test parameter validation bounds."""
        with pytest.raises(ValueError):
            ModelConfig(learning_rate=1.5)
        
        with pytest.raises(ValueError):
            ModelConfig(num_leaves=10)
    
    def test_to_lgbm_params(self):
        """Test conversion to LightGBM parameters."""
        config = ModelConfig(
            learning_rate=0.05,
            num_leaves=50
        )
        
        lgbm_params = config.to_lgbm_params()
        
        assert lgbm_params['learning_rate'] == 0.05
        assert lgbm_params['num_leaves'] == 50
        assert 'verbose' in lgbm_params


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_default_values(self):
        """Test default optimization values."""
        config = OptimizationConfig()
        
        assert config.enabled == False
        assert config.n_trials == 100
        assert config.direction == "minimize"
    
    def test_direction_validation(self):
        """Test direction validation."""
        with pytest.raises(ValueError):
            OptimizationConfig(direction="invalid")


class TestBacktestConfig:
    """Test backtest configuration."""
    
    def test_default_values(self):
        """Test default backtest values."""
        config = BacktestConfig()
        
        assert config.train_window_years == 1.0
        assert config.test_window_months == 3
        assert config.commission_bps == 1.0


class TestConfig:
    """Test main configuration container."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.features, FeatureConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.backtest, BacktestConfig)
    
    def test_nested_configuration(self):
        """Test nested configuration access."""
        config = Config(
            model=ModelConfig(learning_rate=0.05),
            features=FeatureConfig(lag_windows=[3, 7])
        )
        
        assert config.model.learning_rate == 0.05
        assert config.features.lag_windows == [3, 7]
    
    def test_log_level_validation(self):
        """Test log level validation."""
        config = Config(log_level="DEBUG")
        assert config.log_level == "DEBUG"
        
        with pytest.raises(ValueError):
            Config(log_level="INVALID")


class TestConfigIO:
    """Test configuration loading and saving."""
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = Config(
            model=ModelConfig(learning_rate=0.05),
            seed=123
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            save_config(config, str(config_path))
            
            assert config_path.exists()
            
            loaded_config = load_config(str(config_path))
            
            assert loaded_config.model.learning_rate == 0.05
            assert loaded_config.seed == 123
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        config = Config(
            model=ModelConfig(num_leaves=50),
            seed=456
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            save_config(config, str(config_path))
            
            loaded_config = load_config(str(config_path))
            
            assert loaded_config.model.num_leaves == 50
            assert loaded_config.seed == 456
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
    
    def test_load_invalid_format(self):
        """Test loading file with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "config.txt"
            invalid_path.write_text("invalid")
            
            with pytest.raises(ValueError):
                load_config(str(invalid_path))


class TestConfigManipulation:
    """Test configuration manipulation functions."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        assert isinstance(config, Config)
        assert config.seed == 42
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base = Config(
            model=ModelConfig(learning_rate=0.1, num_leaves=100),
            seed=42
        )
        
        override = {
            'model': {'learning_rate': 0.05},
            'seed': 123
        }
        
        merged = merge_configs(base, override)
        
        assert merged.model.learning_rate == 0.05
        assert merged.model.num_leaves == 100
        assert merged.seed == 123
    
    def test_deep_merge(self):
        """Test deep merging of nested configs."""
        base = Config()
        
        override = {
            'features': {
                'lag_windows': [3, 5],
                'use_news': False
            }
        }
        
        merged = merge_configs(base, override)
        
        assert merged.features.lag_windows == [3, 5]
        assert merged.features.use_news == False


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        config = Config(
            data=DataConfig(
                train_start_date="2023-01-01",
                train_end_date="2023-12-31"
            ),
            model=ModelConfig(
                learning_rate=0.05,
                num_iterations=100
            ),
            backtest=BacktestConfig(
                train_window_years=0.5,
                commission_bps=2.0
            )
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "full_config.yaml"
            
            save_config(config, str(config_path))
            
            loaded = load_config(str(config_path))
            
            assert loaded.data.train_start_date == "2023-01-01"
            assert loaded.model.learning_rate == 0.05
            assert loaded.backtest.commission_bps == 2.0
    
    def test_config_modification(self):
        """Test modifying configuration."""
        config = Config()
        
        config.model.learning_rate = 0.05
        config.features.lag_windows = [3, 7]
        
        assert config.model.learning_rate == 0.05
        assert config.features.lag_windows == [3, 7]
    
    def test_load_default_yaml(self):
        """Test loading default.yaml if it exists."""
        default_path = Path("configs/default.yaml")
        
        if default_path.exists():
            config = load_config(str(default_path))
            
            assert isinstance(config, Config)
            assert config.model.model_type == "lgbm"
