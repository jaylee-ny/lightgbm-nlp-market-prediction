from typing import List, Dict
import numpy as np
import pandas as pd
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed, NLP features disabled")


class NewsNLPProcessor:
    """Process news text using FinBERT for sentiment analysis."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize NLP processor with FinBERT.
        
        Args:
            model_name: HuggingFace model name
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. Install: pip install transformers torch")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"NLP processor initialized with {model_name} on {self.device}")
    
    def process_sentiment_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Process sentiment for batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of shape (n_texts, 3) with [negative, neutral, positive] probabilities
        """
        all_sentiments = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_sentiments.append(probs.cpu().numpy())
        
        return np.vstack(all_sentiments)
    
    def compute_sentiment_score(self, sentiments: np.ndarray) -> np.ndarray:
        """
        Convert [neg, neu, pos] probabilities to single sentiment score.
        
        Args:
            sentiments: Array of shape (n, 3)
            
        Returns:
            Sentiment scores in range [-1, 1]
        """
        return sentiments[:, 2] - sentiments[:, 0]
    
    def aggregate_to_asset_level(
        self,
        news_df: pd.DataFrame,
        text_column: str = 'headline',
        asset_column: str = 'asset_name'
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment to asset-day level.
        
        Args:
            news_df: DataFrame with news data
            text_column: Column containing text
            asset_column: Column containing asset identifier
            
        Returns:
            DataFrame with aggregated sentiment per asset-day
        """
        texts = news_df[text_column].fillna("").tolist()
        
        logger.info(f"Processing sentiment for {len(texts)} news items")
        sentiments = self.process_sentiment_batch(texts)
        
        news_df['sentiment_negative'] = sentiments[:, 0]
        news_df['sentiment_neutral'] = sentiments[:, 1]
        news_df['sentiment_positive'] = sentiments[:, 2]
        news_df['sentiment_score'] = self.compute_sentiment_score(sentiments)
        
        aggregated = news_df.groupby(['time', asset_column]).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_negative': 'mean',
            'sentiment_positive': 'mean'
        }).reset_index()
        
        aggregated.columns = [
            'time', asset_column,
            'sentiment_mean', 'sentiment_std', 'news_count',
            'sentiment_negative', 'sentiment_positive'
        ]
        
        return aggregated
