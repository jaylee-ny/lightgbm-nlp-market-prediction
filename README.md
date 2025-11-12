# Two Sigma Financial News Trading Strategy

A machine learning pipeline for developing and backtesting trading strategies using market data and news sentiment.

## Features

- **Walk-Forward Backtesting**: Proper temporal validation with no look-ahead bias
- **Feature Engineering**: Lag features and news sentiment aggregation
- **Model Training**: LightGBM with Bayesian hyperparameter optimization
- **Risk Metrics**: Comprehensive performance evaluation (Sharpe, Sortino, max drawdown, etc.)
- **Transaction Costs**: Realistic cost modeling with commission, spread, and market impact
- **Configuration Management**: Type-safe configs with validation

## Installation
```bash
# Clone repository
git clone <repository-url>
cd two-sigma-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
```

## Quick Start
```bash
# Run with default configuration
python -m src.cli run --data-dir data/raw --output-dir outputs/experiment1

# Run with custom configuration
python -m src.cli run --config configs/fast_experiment.yaml

# Run with hyperparameter optimization
python -m src.cli run --optimize --n-trials 50
```

## Project Structure
```
src/
├── config/          # Configuration management
├── data/            # Data loading and validation
├── features/        # Feature engineering
├── models/          # Model implementations
├── backtest/        # Backtesting engine
├── evaluation/      # Performance metrics
├── pipeline/        # Main pipeline
└── cli.py          # Command-line interface

configs/
├── default.yaml            # Production configuration
└── fast_experiment.yaml    # Quick testing configuration

tests/
├── unit/           # Unit tests
└── integration/    # Integration tests
```

## Configuration

Edit `configs/default.yaml` to customize:

- Data paths and date ranges
- Feature engineering parameters
- Model hyperparameters
- Backtest settings
- Transaction costs

## Results

After running, check the output directory for:

- `predictions.csv`: Out-of-sample predictions
- `feature_importance.csv`: Feature importance rankings
- `performance_report.txt`: Comprehensive metrics
- `config.yaml`: Configuration used
- `pipeline.log`: Execution logs

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_models.py -v
```

## License

MIT License
