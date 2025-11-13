from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from src.utils.time_utils import TimeSeriesSplitter
from src.features.lag_features import LagFeatureGenerator
from src.models.lgbm_model import LGBMModel
from src.models.base_model import ModelMetrics

# Import the updated news aggregator
import sys
sys.path.append('/home/claude')
from news_aggregator_updated import (
    CompetitionNewsAggregator, 
    EnhancedNewsCoverageFiller,
    validate_news_features
)


@dataclass
class BacktestResult:
    """Results from a single backtest fold."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    predictions: np.ndarray
    actual: np.ndarray
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    model_params: Dict[str, Any] = field(default_factory=dict)


class UpdatedBacktestEngine:
    """
    Walk-forward backtesting with competition news features and time-decay weighting.
    
    Parameters
    ----------
    time_splitter : TimeSeriesSplitter
        Splitter for walk-forward validation
    lag_feature_config : Dict[str, Any], optional
        Configuration for lag features
    news_aggregator_config : Dict[str, Any], optional
        Configuration for news aggregation (decay_half_life, use_cross_sectional)
    model_params : Dict[str, Any], optional
        Model hyperparameters
    validate_features : bool
        Whether to run feature validation checks
    """
    
    def __init__(
        self,
        time_splitter: TimeSeriesSplitter,
        lag_feature_config: Optional[Dict[str, Any]] = None,
        news_aggregator_config: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        validate_features: bool = True
    ):
        self.time_splitter = time_splitter
        self.lag_feature_config = lag_feature_config or {}
        self.news_aggregator_config = news_aggregator_config or {
            'decay_half_life': 24.0,
            'use_cross_sectional': True
        }
        self.model_params = model_params or {}
        self.validate_features = validate_features
        
        self.results_ = []
        self.aggregated_metrics_ = {}
    
    def run(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame] = None,
        target_column: str = 'returnsOpenNextMktres10'
    ) -> List[BacktestResult]:
        """Run walk-forward backtest with competition news features."""
        logger.info("="*60)
        logger.info("Starting Walk-Forward Backtest")
        logger.info("="*60)
        
        if target_column not in market_df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Validate news features if provided
        if news_df is not None and self.validate_features:
            required_news_cols = ['time', 'assetName', 'sentimentNegative', 
                                 'sentimentNeutral', 'sentimentPositive']
            missing = [col for col in required_news_cols if col not in news_df.columns]
            if missing:
                logger.warning(f"News data missing expected columns: {missing}")
        
        self.results_ = []
        
        for fold, (train_df, test_df) in enumerate(
            self.time_splitter.split(market_df), start=1
        ):
            logger.info(f"\nFold {fold}/{self.time_splitter.get_n_splits(market_df)}")
            logger.info("-" * 60)
            
            result = self._run_single_fold(
                fold=fold,
                train_df=train_df,
                test_df=test_df,
                news_df=news_df,
                target_column=target_column
            )
            
            if result is not None:
                self.results_.append(result)
                self._log_fold_results(result)
        
        self._aggregate_results()
        self._log_final_results()
        
        logger.info("\n" + "="*60)
        logger.info("Backtest Complete")
        logger.info("="*60)
        
        return self.results_
    
    def _run_single_fold(
        self,
        fold: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        target_column: str
    ) -> Optional[BacktestResult]:
        """Run backtest for a single fold."""
        
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Extract target
        y_train = train_df.pop(target_column).values
        y_test = test_df.pop(target_column).values
        
        # Convert to binary classification (market-relative return direction)
        y_train_binary = (y_train > 0).astype(int)
        y_test_binary = (y_test > 0).astype(int)
        
        if len(np.unique(y_train_binary)) < 2:
            logger.warning(f"Fold {fold}: Only one class in training set, skipping")
            return None
        
        # Prepare features with updated news aggregation
        X_train = self._prepare_features(
            train_df, news_df, is_train=True
        )
        
        X_test = self._prepare_features(
            test_df, news_df, is_train=False
        )
        
        if X_train is None or X_test is None:
            logger.warning(f"Fold {fold}: Feature preparation failed, skipping")
            return None
        
        # Validate features if requested
        if self.validate_features:
            train_validation = validate_news_features(X_train)
            test_validation = validate_news_features(X_test)
            
            if not all(train_validation.values()):
                logger.warning(f"Fold {fold}: Training features failed validation")
            if not all(test_validation.values()):
                logger.warning(f"Fold {fold}: Test features failed validation")
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Train model
        model = LGBMModel(
            model_params=self.model_params,
            early_stopping_rounds=50,
            verbose=-1
        )
        
        # Create validation split
        split_idx = int(len(X_train) * 0.8)
        X_train_fit = X_train.iloc[:split_idx]
        y_train_fit = y_train_binary[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train_binary[split_idx:]
        
        model.fit(X_train_fit, y_train_fit, X_val, y_val)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = ModelMetrics.calculate_binary_metrics(
            y_test_binary, y_pred, y_pred_proba
        )
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Log top features
        logger.info("\nTop 5 Features:")
        for i, row in feature_importance.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.0f}")
        
        result = BacktestResult(
            fold=fold,
            train_start=train_df['time'].min(),
            train_end=train_df['time'].max(),
            test_start=test_df['time'].min(),
            test_end=test_df['time'].max(),
            train_size=len(X_train),
            test_size=len(X_test),
            predictions=y_pred_proba[:, 1],
            actual=y_test_binary,
            metrics=metrics,
            feature_importance=feature_importance,
            model_params=model.get_params()
        )
        
        return result
    
    def _prepare_features(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        is_train: bool
    ) -> Optional[pd.DataFrame]:
        """Prepare features with fit/transform pattern to prevent leakage."""
        market_df = market_df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(market_df['time']):
            market_df['time'] = pd.to_datetime(market_df['time'])
        
        if is_train:
            self.lag_generator_ = LagFeatureGenerator(**self.lag_feature_config)
            self.lag_generator_.fit(market_df)
        
        X = self.lag_generator_.transform(market_df)
        
        if news_df is not None:
            logger.debug("Processing news features...")
            
            news_in_period = news_df[
                (news_df['time'] >= market_df['time'].min()) &
                (news_df['time'] <= market_df['time'].max())
            ].copy()
            
            if len(news_in_period) > 0:
                if is_train:
                    self.news_aggregator_ = CompetitionNewsAggregator(
                        **self.news_aggregator_config
                    )
                    self.news_aggregator_.fit(news_in_period)
                    
                    self.news_coverage_filler_ = EnhancedNewsCoverageFiller()
                
                news_agg = self.news_aggregator_.transform(news_in_period)
                
                logger.debug(f"Aggregated {len(news_in_period)} news items to {len(news_agg)} asset-days")
                logger.debug(f"News features: {[col for col in news_agg.columns if col.startswith('news_')]}")
                
                if is_train:
                    self.news_coverage_filler_.fit(news_agg)
                
                X = self.news_coverage_filler_.transform(X, news_agg)
                
                logger.info(f"Added {sum(col.startswith('news_') for col in X.columns)} news features")
            else:
                logger.warning("No news data in this period")
        
        drop_cols = ['time', 'assetCode', 'assetName']
        feature_cols = [col for col in X.columns if col not in drop_cols]
        
        X = X[feature_cols]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        if X.isna().any().any():
            logger.warning("NaN values remain after fillna")
            logger.warning(f"Columns with NaN: {X.columns[X.isna().any()].tolist()}")
            return None
        
        return X
    
    def _log_fold_results(self, result: BacktestResult):
        """Log results from a single fold."""
        logger.info(f"Fold {result.fold} Results:")
        logger.info(f"  Accuracy:  {result.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {result.metrics['precision']:.4f}")
        logger.info(f"  Recall:    {result.metrics['recall']:.4f}")
        logger.info(f"  F1:        {result.metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {result.metrics.get('roc_auc', np.nan):.4f}")
        logger.info(f"  Log-Loss:  {result.metrics.get('log_loss', np.nan):.4f}")
    
    def _aggregate_results(self):
        """Aggregate metrics across all folds."""
        if not self.results_:
            return
        
        metric_names = self.results_[0].metrics.keys()
        
        for metric in metric_names:
            values = [r.metrics[metric] for r in self.results_ if not np.isnan(r.metrics[metric])]
            
            if values:
                self.aggregated_metrics_[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    def _log_final_results(self):
        """Log aggregated results."""
        if not self.aggregated_metrics_:
            return
        
        logger.info("\n" + "="*60)
        logger.info("Aggregated Results Across All Folds")
        logger.info("="*60)
        
        for metric, stats in self.aggregated_metrics_.items():
            logger.info(
                f"{metric:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics across folds."""
        return self.aggregated_metrics_.copy()
    
    def get_all_predictions(self) -> pd.DataFrame:
        """Get all predictions from all folds."""
        if not self.results_:
            return pd.DataFrame()
        
        all_data = []
        
        for result in self.results_:
            fold_df = pd.DataFrame({
                'fold': result.fold,
                'prediction': result.predictions,
                'actual': result.actual
            })
            all_data.append(fold_df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get average feature importance across folds."""
        if not self.results_:
            return pd.DataFrame()
        
        all_importance = []
        
        for result in self.results_:
            importance = result.feature_importance.copy()
            importance['fold'] = result.fold
            all_importance.append(importance)
        
        combined = pd.concat(all_importance, ignore_index=True)
        
        summary = combined.groupby('feature').agg({
            'importance': ['mean', 'std', 'count']
        }).reset_index()
        
        summary.columns = ['feature', 'mean_importance', 'std_importance', 'n_folds']
        summary = summary.sort_values('mean_importance', ascending=False)
        
        return summary.reset_index(drop=True)
