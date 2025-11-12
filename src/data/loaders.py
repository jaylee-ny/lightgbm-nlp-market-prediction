from typing import Tuple
from pathlib import Path
import pandas as pd
from loguru import logger

from src.config import DataConfig


class DataLoader:
    """Load market and news data from CSV files."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration instance
        """
        self.config = config
        
    def load_market_data(self, filename: str = "marketdata_sample.csv") -> pd.DataFrame:
        """
        Load market data from CSV.
        
        Args:
            filename: Name of market data file
            
        Returns:
            Market data DataFrame with parsed datetime
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.config.raw_data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Market data not found at {file_path}. "
                f"Download data using: kaggle competitions download -c two-sigma-financial-news"
            )
        
        logger.info(f"Loading market data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        required_columns = [
            'time', 'assetCode', 'assetName', 'open', 'close', 'volume',
            'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
            'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
            'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
            'returnsClosePrevMktres10', 'returnsOpenPrevMktres10'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"Loaded {len(df):,} market observations")
        logger.debug(f"Date range: {df['time'].min()} to {df['time'].max()}")
        logger.debug(f"Unique assets: {df['assetCode'].nunique()}")
        
        return df
    
    def load_news_data(self, filename: str = "news_sample.csv") -> pd.DataFrame:
        """
        Load news data from CSV.
        
        Args:
            filename: Name of news data file
            
        Returns:
            News data DataFrame with parsed datetime
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.config.raw_data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"News data not found at {file_path}. "
                f"Download data using: kaggle competitions download -c two-sigma-financial-news"
            )
        
        logger.info(f"Loading news data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        required_columns = ['time', 'assetName', 'sentimentWordCount', 'wordCount']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"Loaded {len(df):,} news items")
        logger.debug(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        return df
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both market and news data.
        
        Returns:
            Tuple of (market_data, news_data)
        """
        market_data = self.load_market_data()
        news_data = self.load_news_data()
        return market_data, news_data
