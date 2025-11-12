import argparse
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

from src.config.settings import load_config, Config
from src.pipeline.main_pipeline import TradingPipeline


def load_data(data_dir: str):
    """
    Load market and news data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing data files
        
    Returns
    -------
    tuple
        (market_df, news_df)
    """
    data_path = Path(data_dir)
    
    # Try to load market data
    market_file = data_path / 'market_train.csv'
    if not market_file.exists():
        raise FileNotFoundError(f"Market data not found: {market_file}")
    
    logger.info(f"Loading market data from {market_file}")
    market_df = pd.read_csv(market_file)
    market_df['time'] = pd.to_datetime(market_df['time'])
    
    # Try to load news data
    news_file = data_path / 'news_train.csv'
    if news_file.exists():
        logger.info(f"Loading news data from {news_file}")
        news_df = pd.read_csv(news_file)
        news_df['time'] = pd.to_datetime(news_df['time'])
    else:
        logger.warning("News data not found, continuing without news features")
        news_df = None
    
    return market_df, news_df


def run_pipeline(args):
    """Run the complete pipeline."""
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Override config with command-line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.optimize:
        config.optimization.enabled = True
        if args.n_trials:
            config.optimization.n_trials = args.n_trials
    
    # Load data
    market_df, news_df = load_data(config.data.data_dir)
    
    # Filter by date range
    if config.data.train_start_date:
        start_date = pd.to_datetime(config.data.train_start_date)
        market_df = market_df[market_df['time'] >= start_date]
        if news_df is not None:
            news_df = news_df[news_df['time'] >= start_date]
    
    if config.data.test_end_date:
        end_date = pd.to_datetime(config.data.test_end_date)
        market_df = market_df[market_df['time'] <= end_date]
        if news_df is not None:
            news_df = news_df[news_df['time'] <= end_date]
    
    logger.info(f"Data loaded: {len(market_df)} market rows")
    if news_df is not None:
        logger.info(f"Data loaded: {len(news_df)} news rows")
    
    # Create and run pipeline
    pipeline = TradingPipeline(config)
    results = pipeline.run(market_df, news_df)
    
    logger.info("\nPipeline completed successfully!")
    
    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Trading Strategy Development Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m src.cli run
  
  # Run with custom configuration
  python -m src.cli run --config configs/fast_experiment.yaml
  
  # Run with optimization
  python -m src.cli run --optimize --n-trials 50
  
  # Specify data and output directories
  python -m src.cli run --data-dir data/raw --output-dir outputs/experiment1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the complete pipeline')
    run_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    run_parser.add_argument(
        '--data-dir', '-d',
        type=str,
        help='Directory containing data files'
    )
    run_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Directory for output files'
    )
    run_parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    run_parser.add_argument(
        '--n-trials',
        type=int,
        help='Number of optimization trials'
    )
    
    args = parser.parse_args()
    
    if args.command == 'run':
        try:
            run_pipeline(args)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
