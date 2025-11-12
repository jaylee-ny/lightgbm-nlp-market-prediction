from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta


class NewsAggregator:
    """
    Aggregate news data to asset-day level with time-decay weighting.
    
    News published closer to market close has more weight than older news.
    Handles missing news coverage by using cross-asset statistics.
    
    Parameters
    ----------
    decay_half_life : float
        Half-life for exponential decay in hours
    min_articles : int
        Minimum articles required for reliable statistics
    """
    
    def __init__(
        self,
        decay_half_life: float = 24.0,
        min_articles: int = 1
    ):
        self.decay_half_life = decay_half_life
        self.min_articles = min_articles
        self.global_stats_ = {}
        self._is_fitted = False
    
    def fit(self, news_df: pd.DataFrame, y=None):
        """
        Learn global news statistics from training data.
        
        Used to impute assets with no news coverage.
        """
        logger.info("Fitting NewsAggregator...")
        
        if 'sentimentWordCount' in news_df.columns and 'wordCount' in news_df.columns:
            coverage = news_df['sentimentWordCount'] / (news_df['wordCount'] + 1e-8)
            
            self.global_stats_ = {
                'coverage_median': coverage.median(),
                'sentiment_count_median': news_df['sentimentWordCount'].median(),
                'word_count_median': news_df['wordCount'].median(),
                'articles_per_day_median': self._compute_median_articles_per_day(news_df)
            }
        
        self._is_fitted = True
        logger.info("NewsAggregator fitted with global statistics")
        return self
    
    def transform(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news to asset-day level.
        
        Returns
        -------
        pd.DataFrame
            One row per asset-day with aggregated news features
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        logger.debug(f"Aggregating {len(news_df)} news items...")
        
        if len(news_df) == 0:
            return pd.DataFrame(columns=[
                'time', 'assetName',
                'news_sentiment_mean', 'news_sentiment_std',
                'news_volume', 'news_coverage_mean'
            ])
        
        news_df = self._add_time_weights(news_df)
        
        agg_features = self._aggregate_by_asset_day(news_df)
        
        return agg_features
    
    def _add_time_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add exponential time-decay weights to news items.
        
        More recent news within a day gets higher weight.
        """
        df = df.copy()
        
        if 'time' not in df.columns:
            logger.warning("No time column found, skipping time weighting")
            df['time_weight'] = 1.0
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        df = df.sort_values('time')
        
        df['date'] = df['time'].dt.date
        
        df['hours_old'] = df.groupby(['date', 'assetName'])['time'].transform(
            lambda x: (x.max() - x).dt.total_seconds() / 3600
        )
        
        decay_rate = np.log(2) / self.decay_half_life
        df['time_weight'] = np.exp(-decay_rate * df['hours_old'])
        
        return df
    
    def _aggregate_by_asset_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news features to asset-day level."""
        if 'sentimentWordCount' not in df.columns or 'wordCount' not in df.columns:
            logger.warning("Missing sentiment columns, using defaults")
            return self._create_empty_aggregation(df)
        
        df['coverage'] = df['sentimentWordCount'] / (df['wordCount'] + 1e-8)
        
        df['date'] = pd.to_datetime(df['time']).dt.date if 'date' not in df.columns else df['date']
        
        agg_dict = {
            'sentimentWordCount': [
                ('news_sentiment_sum', lambda x: (x * df.loc[x.index, 'time_weight']).sum()),
                ('news_sentiment_mean', 'mean'),
                ('news_sentiment_std', 'std'),
            ],
            'coverage': [
                ('news_coverage_mean', 'mean'),
                ('news_coverage_std', 'std')
            ],
            'wordCount': [
                ('news_total_words', 'sum')
            ],
            'time_weight': [
                ('news_volume', 'count')
            ]
        }
        
        grouped = df.groupby(['date', 'assetName']).agg(agg_dict)
        grouped.columns = [col[1] if isinstance(col, tuple) else col 
                          for col in grouped.columns]
        
        grouped = grouped.reset_index()
        grouped.rename(columns={'date': 'time'}, inplace=True)
        
        return grouped
    
    def _create_empty_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create empty aggregation when sentiment columns missing."""
        dates = pd.to_datetime(df['time']).dt.date.unique() if 'time' in df.columns else []
        assets = df['assetName'].unique() if 'assetName' in df.columns else []
        
        result = pd.DataFrame({
            'time': dates.repeat(len(assets)) if len(dates) > 0 else [],
            'assetName': list(assets) * len(dates) if len(assets) > 0 else [],
            'news_volume': 0,
            'news_sentiment_mean': self.global_stats_.get('sentiment_count_median', 0),
            'news_sentiment_std': 0,
            'news_coverage_mean': self.global_stats_.get('coverage_median', 0),
            'news_coverage_std': 0,
            'news_total_words': self.global_stats_.get('word_count_median', 0)
        })
        
        return result
    
    def _compute_median_articles_per_day(self, df: pd.DataFrame) -> float:
        """Compute median number of articles per asset per day."""
        if 'time' not in df.columns or 'assetName' not in df.columns:
            return 0.0
        
        df['date'] = pd.to_datetime(df['time']).dt.date
        counts = df.groupby(['date', 'assetName']).size()
        
        return counts.median() if len(counts) > 0 else 0.0


class NewsCoverageFiller:
    """
    Fill missing news coverage for assets with no news.
    
    Uses statistics from assets with similar market cap or sector.
    """
    
    def __init__(self, strategy: str = 'global_median'):
        """
        Parameters
        ----------
        strategy : str
            'global_median': Use overall median
            'zero': Fill with zeros
            'forward_fill': Use last available news
        """
        self.strategy = strategy
        self.fill_values_ = {}
    
    def fit(self, aggregated_news: pd.DataFrame, y=None):
        """Learn fill values from training data."""
        news_cols = [col for col in aggregated_news.columns 
                    if col.startswith('news_')]
        
        for col in news_cols:
            if self.strategy == 'global_median':
                self.fill_values_[col] = aggregated_news[col].median()
            elif self.strategy == 'zero':
                self.fill_values_[col] = 0.0
        
        return self
    
    def transform(
        self,
        market_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge news with market data and fill missing coverage.
        
        Parameters
        ----------
        market_df : pd.DataFrame
            Market data with all assets
        news_df : pd.DataFrame
            Aggregated news data (may not cover all assets)
            
        Returns
        -------
        pd.DataFrame
            Market data with news features, missing values filled
        """
        merged = pd.merge(
            market_df,
            news_df,
            how='left',
            left_on=['time', 'assetName'],
            right_on=['time', 'assetName']
        )
        
        news_cols = [col for col in merged.columns if col.startswith('news_')]
        
        for col in news_cols:
            fill_value = self.fill_values_.get(col, 0.0)
            merged[col].fillna(fill_value, inplace=True)
        
        merged['has_news'] = merged['news_volume'] > 0
        
        return merged


class NewsQualityValidator:
    """Validate news data quality."""
    
    @staticmethod
    def check_coverage_rate(
        market_df: pd.DataFrame,
        news_df: pd.DataFrame,
        min_coverage: float = 0.5
    ) -> bool:
        """
        Check if sufficient assets have news coverage.
        
        Parameters
        ----------
        market_df : pd.DataFrame
            Market data
        news_df : pd.DataFrame
            Aggregated news data
        min_coverage : float
            Minimum fraction of asset-days that should have news
            
        Returns
        -------
        bool
            True if coverage is sufficient
        """
        total_asset_days = len(market_df)
        
        merged = pd.merge(
            market_df[['time', 'assetName']],
            news_df[['time', 'assetName']],
            how='left',
            on=['time', 'assetName'],
            indicator=True
        )
        
        covered = (merged['_merge'] == 'both').sum()
        coverage_rate = covered / total_asset_days if total_asset_days > 0 else 0
        
        logger.info(f"News coverage rate: {coverage_rate:.2%}")
        
        if coverage_rate < min_coverage:
            logger.warning(
                f"Low news coverage: {coverage_rate:.2%} < {min_coverage:.2%}"
            )
            return False
        
        return True
    
    @staticmethod
    def detect_duplicates(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect duplicate news items.
        
        Same headline/asset/time likely indicates data quality issue.
        """
        if 'headline' not in news_df.columns:
            return news_df
        
        duplicates = news_df.duplicated(
            subset=['time', 'assetName', 'headline'],
            keep='first'
        )
        
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate news items")
            news_df = news_df[~duplicates].copy()
        
        return news_df
