from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from src.config.settings import Config
from src.utils.time_utils import TimeSeriesSplitter
from src.features.lag_features import LagFeatureGenerator
from src.features.news_aggregator import NewsAggregator, NewsCoverageFiller
from src.models.lgbm_model import LGBMModel
from src.models.optimizer import BayesianOptimizer
from src.backtest.engine import BacktestEngine
from src.backtest.costs import TransactionCostModel, CostAnalyzer
from src.evaluation.metrics import PerformanceEvaluator


class TradingPipeline:
    """
    End-to-end pipeline for trading strategy development.
    
    Orchestrates:
    1. Data loading and validation
    2. Feature engineering
    3. Model training (with optional optimization)
    4. Walk-forward backtesting
    5. Performance evaluation
    6. Results export
    
    Parameters
    ----------
    config : Config
        Configuration object
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level=config.log_level
        )
        logger.add(
            self.output_dir / "pipeline.log",
            level=config.log_level,
            rotation="10 MB"
        )
        
        logger.info("="*60)
        logger.info("Trading Pipeline Initialized")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
    
    def run(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Parameters
        ----------
        market_df : pd.DataFrame
            Market data
        news_df : pd.DataFrame, optional
            News data
            
        Returns
        -------
        Dict[str, Any]
            Pipeline results
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Pipeline Execution")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Data validation
        logger.info("\n[1/5] Validating data...")
        self._validate_data(market_df, news_df)
        
        # Step 2: Hyperparameter optimization (if enabled)
        if self.config.optimization.enabled:
            logger.info("\n[2/5] Running hyperparameter optimization...")
            best_params = self._run_optimization(market_df, news_df)
            self.results['best_params'] = best_params
        else:
            logger.info("\n[2/5] Skipping optimization (disabled)")
            best_params = self.config.model.to_lgbm_params()
        
        # Step 3: Walk-forward backtesting
        logger.info("\n[3/5] Running walk-forward backtest...")
        backtest_results = self._run_backtest(market_df, news_df, best_params)
        self.results['backtest_results'] = backtest_results
        
        # Step 4: Performance evaluation
        logger.info("\n[4/5] Evaluating performance...")
        performance_metrics = self._evaluate_performance(backtest_results)
        self.results['performance_metrics'] = performance_metrics
        
        # Step 5: Export results
        logger.info("\n[5/5] Exporting results...")
        self._export_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline Execution Complete")
        logger.info(f"Total time: {duration:.1f} seconds")
        logger.info("="*60)
        
        return self.results
    
    def _validate_data(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame]
    ):
        """Validate input data."""
        logger.info(f"Market data: {len(market_df)} rows")
        
        required_cols = [
            'time', 'assetCode', 'assetName',
            'returnsOpenNextMktres10'
        ]
        
        missing_cols = [col for col in required_cols if col not in market_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check date range
        min_date = market_df['time'].min()
        max_date = market_df['time'].max()
        logger.info(f"Date range: {min_date} to {max_date}")
        
        # Check for duplicates
        duplicates = market_df.duplicated(subset=['time', 'assetCode']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        if news_df is not None:
            logger.info(f"News data: {len(news_df)} rows")
            
            if 'time' not in news_df.columns or 'assetName' not in news_df.columns:
                raise ValueError("News data missing required columns")
        
        logger.info("Data validation passed")
    
    def _run_optimization(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization."""
        # Prepare features for optimization
        train_end = pd.to_datetime(self.config.data.train_end_date)
        train_df = market_df[market_df['time'] <= train_end].copy()
        
        # Create features
        lag_generator = LagFeatureGenerator(
            lag_windows=self.config.features.lag_windows,
            feature_columns=self.config.features.lag_features
        )
        lag_generator.fit(train_df)
        X = lag_generator.transform(train_df)
        
        if news_df is not None and self.config.features.use_news:
            news_agg = NewsAggregator(
                decay_half_life=self.config.features.news_decay_half_life
            )
            news_agg.fit(news_df)
            news_features = news_agg.transform(news_df)
            
            filler = NewsCoverageFiller()
            filler.fit(news_features)
            X = filler.transform(X, news_features)
        
        # Prepare target
        y = train_df['returnsOpenNextMktres10'].values
        y_binary = (y > 0).astype(int)
        
        # Drop non-feature columns
        drop_cols = ['time', 'assetCode', 'assetName', 'returnsOpenNextMktres10']
        feature_cols = [col for col in X.columns if col not in drop_cols]
        X = X[feature_cols].fillna(0)
        
        # Run optimization
        optimizer = BayesianOptimizer(
            n_trials=self.config.optimization.n_trials,
            cv_folds=self.config.optimization.cv_folds,
            random_state=self.config.seed
        )
        
        best_params = optimizer.optimize(
            X, y_binary,
            objective_metric=self.config.optimization.optimization_metric,
            direction=self.config.optimization.direction
        )
        
        # Save optimization history
        history = optimizer.get_optimization_history()
        history.to_csv(self.output_dir / 'optimization_history.csv', index=False)
        
        return best_params
    
    def _run_backtest(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        model_params: Dict[str, Any]
    ):
        """Run walk-forward backtesting."""
        # Create time splitter
        splitter = TimeSeriesSplitter(
            train_window_years=self.config.backtest.train_window_years,
            test_window_months=self.config.backtest.test_window_months
        )
        
        # Create backtest engine
        engine = BacktestEngine(
            time_splitter=splitter,
            lag_feature_config={
                'lag_windows': self.config.features.lag_windows,
                'feature_columns': self.config.features.lag_features
            },
            news_aggregator_config={
                'decay_half_life': self.config.features.news_decay_half_life
            } if self.config.features.use_news else None,
            model_params=model_params
        )
        
        # Run backtest
        results = engine.run(market_df, news_df)
        
        return {
            'fold_results': results,
            'aggregated_metrics': engine.get_aggregated_metrics(),
            'all_predictions': engine.get_all_predictions(),
            'feature_importance': engine.get_feature_importance_summary()
        }
    
    def _evaluate_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategy performance."""
        predictions_df = backtest_results['all_predictions']
        
        # Calculate returns (simplified - assumes equal weighting)
        predictions_df['position'] = (predictions_df['prediction'] > 0.5).astype(int)
        predictions_df['position'] = predictions_df['position'] * 2 - 1  # -1 or 1
        
        # Group by time and calculate average return
        daily_returns = predictions_df.groupby('fold').apply(
            lambda x: (x['actual'] * 2 - 1).mean() * 0.01  # Simplified return calculation
        ).values
        
        # Calculate performance metrics
        evaluator = PerformanceEvaluator(
            periods_per_year=self.config.backtest.periods_per_year,
            risk_free_rate=self.config.backtest.risk_free_rate
        )
        
        metrics = evaluator.evaluate_strategy(daily_returns)
        
        # Add transaction cost analysis
        cost_model = TransactionCostModel(
            commission_bps=self.config.backtest.commission_bps,
            spread_bps=self.config.backtest.spread_bps,
            market_impact_coef=self.config.backtest.market_impact_coef
        )
        
        turnover_stats = CostAnalyzer.analyze_turnover(predictions_df)
        
        cost_impact = CostAnalyzer.estimate_total_costs(
            daily_returns,
            turnover_stats['mean_turnover'],
            cost_model
        )
        
        # Create summary report
        report = evaluator.create_summary_report(metrics)
        
        return {
            'metrics': metrics,
            'turnover_stats': turnover_stats,
            'cost_impact': cost_impact,
            'report': report
        }
    
    def _export_results(self):
        """Export results to files."""
        # Export configuration
        from src.config.settings import save_config
        save_config(self.config, str(self.output_dir / 'config.yaml'))
        
        # Export performance report
        if 'performance_metrics' in self.results:
            report = self.results['performance_metrics']['report']
            with open(self.output_dir / 'performance_report.txt', 'w') as f:
                f.write(report)
            
            logger.info("\nPerformance Report:")
            logger.info(report)
        
        # Export predictions
        if 'backtest_results' in self.results:
            predictions_df = self.results['backtest_results']['all_predictions']
            predictions_df.to_csv(
                self.output_dir / 'predictions.csv',
                index=False
            )
            
            # Export feature importance
            importance_df = self.results['backtest_results']['feature_importance']
            importance_df.to_csv(
                self.output_dir / 'feature_importance.csv',
                index=False
            )
            
            logger.info(f"\nTop 10 Most Important Features:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"  {i+1}. {row['feature']}: {row['mean_importance']:.2f}")
        
        # Export aggregated metrics
        if 'backtest_results' in self.results:
            agg_metrics = self.results['backtest_results']['aggregated_metrics']
            agg_df = pd.DataFrame(agg_metrics).T
            agg_df.to_csv(self.output_dir / 'aggregated_metrics.csv')
        
        logger.info(f"\nAll results exported to: {self.output_dir}")
