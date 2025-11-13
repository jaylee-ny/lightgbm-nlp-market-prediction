from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta


class CompetitionNewsAggregator:
    """
    Aggregate Two Sigma news features with exponential time-decay weighting.
    
    Parameters
    ----------
    decay_half_life : float
        Half-life for exponential decay in hours (default: 24.0)
    min_articles : int
        Minimum articles required for reliable statistics
    use_cross_sectional : bool
        Whether to compute relative sentiment features
    """
    
    def __init__(
        self,
        decay_half_life: float = 24.0,
        min_articles: int = 1,
        use_cross_sectional: bool = True
    ):
        self.decay_half_life = decay_half_life
        self.min_articles = min_articles
        self.use_cross_sectional = use_cross_sectional
        self.global_stats_ = {}
        self._is_fitted = False
        
        # Competition feature names
        self.sentiment_cols = ['sentimentNegative', 'sentimentNeutral', 'sentimentPositive']
        self.novelty_cols = ['noveltyCount12H', 'noveltyCount24H', 'noveltyCount3D', 
                            'noveltyCount5D', 'noveltyCount7D']
        self.volume_cols = ['volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D',
                           'volumeCounts5D', 'volumeCounts7D']
    
    def fit(self, news_df: pd.DataFrame, y=None):
        """Learn global statistics for imputation from training data."""
        logger.info("Fitting CompetitionNewsAggregator...")
        
        # Validate required columns exist
        required_cols = ['time', 'assetName'] + self.sentiment_cols
        missing_cols = [col for col in required_cols if col not in news_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Compute global statistics for imputation
        self.global_stats_ = {
            'sentiment_negative_median': news_df['sentimentNegative'].median(),
            'sentiment_neutral_median': news_df['sentimentNeutral'].median(),
            'sentiment_positive_median': news_df['sentimentPositive'].median(),
            'relevance_median': news_df['relevance'].median() if 'relevance' in news_df.columns else 0.5,
            'articles_per_day_median': self._compute_median_articles_per_day(news_df)
        }
        
        # Compute novelty statistics if available
        for col in self.novelty_cols:
            if col in news_df.columns:
                self.global_stats_[f'{col}_median'] = news_df[col].median()
        
        self._is_fitted = True
        logger.info(f"Fitted with global stats: {self.global_stats_}")
        return self
    
    def transform(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news to asset-day level with time-decay weighting.
        
        Returns aggregated features: sentiment scores, novelty metrics, volume.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        logger.debug(f"Aggregating {len(news_df)} news items...")
        
        if len(news_df) == 0:
            return self._create_empty_aggregation()
        
        # Add time weights (exponential decay)
        news_df = self._add_time_weights(news_df)
        
        # Aggregate by asset-day
        agg_features = self._aggregate_by_asset_day(news_df)
        
        # Add cross-sectional features if requested
        if self.use_cross_sectional:
            agg_features = self._add_cross_sectional_features(agg_features)
        
        return agg_features
    
    def _add_time_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add exponential time-decay weights: weight = exp(-λ * hours_old)
        where λ = ln(2) / half_life
        """
        df = df.copy()
        
        if 'time' not in df.columns:
            logger.warning("No time column found, using uniform weights")
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
        
        df['time_weight'] = df.groupby(['date', 'assetName'])['time_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
        )
        
        return df
    
    def _aggregate_by_asset_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news features to asset-day level using time-weighted averages."""
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['time']).dt.date
        
        for col in self.sentiment_cols:
            if col not in df.columns:
                logger.warning(f"Missing {col}, filling with neutral value")
                if 'Negative' in col or 'Positive' in col:
                    df[col] = 0.3
                else:
                    df[col] = 0.4
        
        df['sentiment_score'] = df['sentimentPositive'] - df['sentimentNegative']
        
        agg_dict = {}
        
        for col in self.sentiment_cols:
            agg_dict[col] = [
                (f'news_{col.lower()}_mean', lambda x: np.average(
                    x, weights=df.loc[x.index, 'time_weight']
                )),
                (f'news_{col.lower()}_std', 'std')
            ]
        
        agg_dict['sentiment_score'] = [
            ('news_sentiment_score', lambda x: np.average(
                x, weights=df.loc[x.index, 'time_weight']
            ))
        ]
        
        if 'relevance' in df.columns:
            agg_dict['relevance'] = [
                ('news_relevance_mean', lambda x: np.average(
                    x, weights=df.loc[x.index, 'time_weight']
                ))
            ]
        
        for col in self.novelty_cols:
            if col in df.columns:
                agg_dict[col] = [
                    (f'news_{col.lower()}_mean', 'mean')
                ]
        
        for col in self.volume_cols:
            if col in df.columns:
                agg_dict[col] = [
                    (f'news_{col.lower()}_sum', 'sum')
                ]
        
        agg_dict['time_weight'] = [
            ('news_volume', 'count')
        ]
        
        if 'firstMentionSentence' in df.columns:
            agg_dict['firstMentionSentence'] = [
                ('news_first_mention_mean', 'mean')
            ]
        
        if 'urgency' in df.columns:
            agg_dict['urgency'] = [
                ('news_urgency_mean', 'mean')
            ]
        
        grouped = df.groupby(['date', 'assetName']).agg(agg_dict)
        
        grouped.columns = [col[1] if isinstance(col, tuple) else col 
                          for col in grouped.columns]
        
        grouped = grouped.reset_index()
        grouped.rename(columns={'date': 'time'}, inplace=True)
        
        return grouped
    
    def _add_cross_sectional_features(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment relative to market average and z-score."""
        if 'news_sentiment_score' not in agg_df.columns:
            return agg_df
        
        market_sentiment = agg_df.groupby('time')['news_sentiment_score'].transform('mean')
        agg_df['news_sentiment_relative'] = (
            agg_df['news_sentiment_score'] - market_sentiment
        )
        
        market_std = agg_df.groupby('time')['news_sentiment_score'].transform('std')
        agg_df['news_sentiment_zscore'] = (
            (agg_df['news_sentiment_score'] - market_sentiment) / 
            (market_std + 1e-6)
        )
        
        return agg_df
    
    def _create_empty_aggregation(self) -> pd.DataFrame:
        """Create empty aggregation when no news data available."""
        return pd.DataFrame(columns=[
            'time', 'assetName',
            'news_sentiment_score',
            'news_sentimentnegative_mean', 'news_sentimentneutral_mean', 
            'news_sentimentpositive_mean',
            'news_volume', 'news_relevance_mean'
        ])
    
    def _compute_median_articles_per_day(self, df: pd.DataFrame) -> float:
        """Compute median number of articles per asset per day."""
        if 'time' not in df.columns or 'assetName' not in df.columns:
            return 0.0
        
        df['date'] = pd.to_datetime(df['time']).dt.date
        counts = df.groupby(['date', 'assetName']).size()
        
        return counts.median() if len(counts) > 0 else 0.0


class EnhancedNewsCoverageFiller:
    """Fill missing news coverage with training set statistics."""
    
    def __init__(self, strategy: str = 'global_median'):
        self.strategy = strategy
        self.fill_values_ = {}
        self.feature_names_ = []
    
    def fit(self, aggregated_news: pd.DataFrame, y=None):
        """Learn fill values from training data."""
        news_cols = [col for col in aggregated_news.columns 
                    if col.startswith('news_')]
        
        self.feature_names_ = news_cols
        
        for col in news_cols:
            if self.strategy == 'global_median':
                self.fill_values_[col] = aggregated_news[col].median()
            elif self.strategy == 'zero':
                self.fill_values_[col] = 0.0
            else:
                self.fill_values_[col] = 0.0
        
        logger.info(f"Learned fill values for {len(self.fill_values_)} features")
        return self
    
    def transform(
        self,
        market_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge news with market data and fill missing coverage."""
        if 'time' in market_df.columns:
            market_df = market_df.copy()
            market_df['date'] = pd.to_datetime(market_df['time']).dt.date
        
        if 'time' in news_df.columns:
            news_df = news_df.copy()
            if not isinstance(news_df['time'].iloc[0], pd.Timestamp):
                news_df['date'] = pd.to_datetime(news_df['time']).dt.date
            else:
                news_df['date'] = news_df['time'].dt.date if pd.api.types.is_datetime64_any_dtype(news_df['time']) else news_df['time']
        
        merged = pd.merge(
            market_df,
            news_df.drop(columns=['time'] if 'time' in news_df.columns else []),
            how='left',
            left_on=['date', 'assetName'],
            right_on=['date', 'assetName']
        )
        
        for col in self.feature_names_:
            if col in merged.columns:
                fill_value = self.fill_values_.get(col, 0.0)
                merged[col].fillna(fill_value, inplace=True)
            else:
                merged[col] = self.fill_values_.get(col, 0.0)
        
        merged['has_news'] = merged.get('news_volume', 0) > 0
        
        if 'date' in merged.columns:
            merged = merged.drop(columns=['date'])
        
        return merged


def validate_news_features(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate news features: no NaN, sentiment in [-1.5, 1.5], probabilities in [0, 1].
    """
    results = {}
    
    news_cols = [col for col in df.columns if col.startswith('news_')]
    results['no_nan_values'] = not df[news_cols].isna().any().any()
    
    if 'news_sentiment_score' in df.columns:
        scores = df['news_sentiment_score'].dropna()
        results['sentiment_in_range'] = (scores >= -1.5).all() and (scores <= 1.5).all()
    
    prob_cols = ['news_sentimentnegative_mean', 'news_sentimentneutral_mean', 
                 'news_sentimentpositive_mean']
    for col in prob_cols:
        if col in df.columns:
            values = df[col].dropna()
            results[f'{col}_valid'] = (values >= 0).all() and (values <= 1).all()
    
    logger.info("News feature validation:")
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}")
    
    return results

if __name__ == "__main__":
    # Quick test
    news_df = pd.DataFrame({
        'time': pd.date_range('2024-01-01 09:00', periods=100, freq='H'),
        'assetName': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 100),
        'sentimentNegative': np.random.uniform(0.1, 0.4, 100),
        'sentimentNeutral': np.random.uniform(0.3, 0.6, 100),
        'sentimentPositive': np.random.uniform(0.1, 0.4, 100),
        'noveltyCount24H': np.random.randint(0, 5, 100),
        'relevance': np.random.uniform(0.5, 1.0, 100)
    })
    
    aggregator = CompetitionNewsAggregator(decay_half_life=24.0)
    aggregator.fit(news_df)
    agg_features = aggregator.transform(news_df)
    
    print(f"\nFeatures: {[col for col in agg_features.columns if col.startswith('news_')]}")
    validate_news_features(agg_features)
