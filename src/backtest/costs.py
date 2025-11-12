"""
Transaction cost modeling for realistic backtesting.

Includes commission, bid-ask spread, and market impact using
square-root model. Critical for understanding true profitability.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


class TransactionCostModel:
    """
    Calculate transaction costs for trades.
    
    Components:
    1. Commission: Fixed cost per trade
    2. Bid-Ask Spread: Cost of crossing the spread
    3. Market Impact: Price impact from large orders
    
    Parameters
    ----------
    commission_bps : float
        Commission in basis points (1 bps = 0.01%)
    spread_bps : float
        Half-spread in basis points
    market_impact_coef : float
        Coefficient for square-root market impact model
    min_trade_size : float
        Minimum trade size to execute
    """
    
    def __init__(
        self,
        commission_bps: float = 1.0,
        spread_bps: float = 5.0,
        market_impact_coef: float = 0.1,
        min_trade_size: float = 0.0
    ):
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.market_impact_coef = market_impact_coef
        self.min_trade_size = min_trade_size
        
        logger.info(
            f"TransactionCostModel: commission={commission_bps}bps, "
            f"spread={spread_bps}bps, impact_coef={market_impact_coef}"
        )
    
    def calculate_total_cost(
        self,
        position_value: float,
        daily_volume: float,
        trade_direction: str = 'buy'
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost.
        
        Parameters
        ----------
        position_value : float
            Dollar value of position
        daily_volume : float
            Average daily dollar volume
        trade_direction : str
            'buy' or 'sell'
            
        Returns
        -------
        Dict[str, float]
            Breakdown of costs
        """
        if position_value < self.min_trade_size:
            return {
                'commission': 0.0,
                'spread': 0.0,
                'market_impact': 0.0,
                'total': 0.0,
                'total_bps': 0.0
            }
        
        commission = self._calculate_commission(position_value)
        
        spread_cost = self._calculate_spread_cost(position_value)
        
        market_impact = self._calculate_market_impact(
            position_value,
            daily_volume
        )
        
        total_cost = commission + spread_cost + market_impact
        total_cost_bps = (total_cost / position_value) * 10000
        
        return {
            'commission': commission,
            'spread': spread_cost,
            'market_impact': market_impact,
            'total': total_cost,
            'total_bps': total_cost_bps
        }
    
    def _calculate_commission(self, position_value: float) -> float:
        """Calculate commission cost."""
        return position_value * (self.commission_bps / 10000)
    
    def _calculate_spread_cost(self, position_value: float) -> float:
        """
        Calculate bid-ask spread cost.
        
        When buying, pay the ask (half-spread above mid).
        When selling, receive the bid (half-spread below mid).
        """
        return position_value * (self.spread_bps / 10000)
    
    def _calculate_market_impact(
        self,
        position_value: float,
        daily_volume: float
    ) -> float:
        """
        Calculate market impact using square-root model.
        
        Impact ~ sqrt(trade_size / daily_volume)
        
        Based on: Almgren, Chriss (2000)
        """
        if daily_volume <= 0:
            logger.warning("Zero or negative daily volume, setting high impact")
            return position_value * 0.01
        
        participation_rate = position_value / daily_volume
        
        if participation_rate > 0.5:
            logger.warning(
                f"High participation rate: {participation_rate:.2%} of daily volume"
            )
        
        impact_bps = self.market_impact_coef * np.sqrt(participation_rate * 10000)
        
        impact_bps = min(impact_bps, 100.0)
        
        return position_value * (impact_bps / 10000)
    
    def calculate_roundtrip_cost(
        self,
        position_value: float,
        daily_volume: float
    ) -> float:
        """
        Calculate cost for full roundtrip (buy then sell).
        
        Parameters
        ----------
        position_value : float
            Dollar value of position
        daily_volume : float
            Average daily dollar volume
            
        Returns
        -------
        float
            Total roundtrip cost
        """
        buy_cost = self.calculate_total_cost(
            position_value,
            daily_volume,
            'buy'
        )
        
        sell_cost = self.calculate_total_cost(
            position_value,
            daily_volume,
            'sell'
        )
        
        return buy_cost['total'] + sell_cost['total']


class PortfolioRebalancer:
    """
    Calculate transaction costs for portfolio rebalancing.
    
    Tracks position changes and computes costs for transitions.
    """
    
    def __init__(self, cost_model: TransactionCostModel):
        self.cost_model = cost_model
        self.current_positions = {}
    
    def calculate_rebalance_costs(
        self,
        target_positions: Dict[str, float],
        prices: Dict[str, float],
        volumes: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate costs to rebalance to target positions.
        
        Parameters
        ----------
        target_positions : Dict[str, float]
            Target number of shares per asset
        prices : Dict[str, float]
            Current prices per asset
        volumes : Dict[str, float]
            Average daily dollar volumes per asset
            
        Returns
        -------
        Dict[str, any]
            Rebalancing summary with costs
        """
        total_cost = 0.0
        trades = []
        
        all_assets = set(target_positions.keys()) | set(self.current_positions.keys())
        
        for asset in all_assets:
            current_shares = self.current_positions.get(asset, 0.0)
            target_shares = target_positions.get(asset, 0.0)
            
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) < 1e-6:
                continue
            
            price = prices.get(asset, 0.0)
            volume = volumes.get(asset, 1e9)
            
            if price <= 0:
                logger.warning(f"Invalid price for {asset}: {price}")
                continue
            
            trade_value = abs(shares_to_trade) * price
            
            trade_direction = 'buy' if shares_to_trade > 0 else 'sell'
            
            cost_breakdown = self.cost_model.calculate_total_cost(
                trade_value,
                volume,
                trade_direction
            )
            
            trades.append({
                'asset': asset,
                'shares_traded': shares_to_trade,
                'direction': trade_direction,
                'trade_value': trade_value,
                'cost': cost_breakdown['total'],
                'cost_bps': cost_breakdown['total_bps']
            })
            
            total_cost += cost_breakdown['total']
        
        return {
            'total_cost': total_cost,
            'n_trades': len(trades),
            'trades': trades
        }
    
    def update_positions(self, new_positions: Dict[str, float]):
        """Update current positions after rebalancing."""
        self.current_positions = new_positions.copy()


class CostAnalyzer:
    """Analyze transaction costs from backtest results."""
    
    @staticmethod
    def analyze_turnover(
        predictions: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Analyze portfolio turnover from predictions.
        
        Parameters
        ----------
        predictions : pd.DataFrame
            Predictions with columns: time, asset, prediction
        threshold : float
            Threshold for binary position (long if pred > threshold)
            
        Returns
        -------
        Dict[str, float]
            Turnover statistics
        """
        predictions = predictions.sort_values('time')
        
        predictions['position'] = (predictions['prediction'] > threshold).astype(int)
        
        predictions['position_change'] = predictions.groupby('asset')['position'].diff().fillna(0)
        
        turnover_per_period = predictions.groupby('time')['position_change'].apply(
            lambda x: (abs(x).sum()) / 2
        )
        
        return {
            'mean_turnover': turnover_per_period.mean(),
            'median_turnover': turnover_per_period.median(),
            'max_turnover': turnover_per_period.max(),
            'total_trades': predictions['position_change'].ne(0).sum()
        }
    
    @staticmethod
    def estimate_total_costs(
        returns: np.ndarray,
        turnover_rate: float,
        cost_model: TransactionCostModel,
        portfolio_value: float = 1e6,
        avg_daily_volume: float = 1e7
    ) -> Dict[str, float]:
        """
        Estimate total costs for a return series.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        turnover_rate : float
            Turnover per period (0.5 = 50% of portfolio traded)
        cost_model : TransactionCostModel
            Cost model to use
        portfolio_value : float
            Total portfolio value
        avg_daily_volume : float
            Average daily volume per position
            
        Returns
        -------
        Dict[str, float]
            Cost statistics
        """
        n_periods = len(returns)
        
        costs_per_period = []
        for _ in range(n_periods):
            trade_value = portfolio_value * turnover_rate
            
            cost = cost_model.calculate_total_cost(
                trade_value,
                avg_daily_volume
            )
            
            costs_per_period.append(cost['total'])
        
        total_costs = sum(costs_per_period)
        total_return = np.sum(returns)
        returns_after_costs = total_return - (total_costs / portfolio_value)
        
        return {
            'total_costs': total_costs,
            'cost_per_period': np.mean(costs_per_period),
            'costs_bps': (total_costs / portfolio_value) * 10000,
            'gross_return': total_return,
            'net_return': returns_after_costs,
            'cost_drag': total_return - returns_after_costs
        }
    
    @staticmethod
    def breakeven_analysis(
        cost_model: TransactionCostModel,
        holding_periods: np.ndarray,
        portfolio_value: float = 1e6,
        avg_daily_volume: float = 1e7
    ) -> pd.DataFrame:
        """
        Analyze breakeven returns for different holding periods.
        
        Parameters
        ----------
        cost_model : TransactionCostModel
            Cost model
        holding_periods : np.ndarray
            Holding periods to analyze (in days)
        portfolio_value : float
            Portfolio value
        avg_daily_volume : float
            Average daily volume
            
        Returns
        -------
        pd.DataFrame
            Breakeven analysis
        """
        results = []
        
        for period in holding_periods:
            roundtrip_cost = cost_model.calculate_roundtrip_cost(
                portfolio_value,
                avg_daily_volume
            )
            
            breakeven_return = roundtrip_cost / portfolio_value
            breakeven_return_pct = breakeven_return * 100
            annualized_breakeven = (breakeven_return * 252) / period
            
            results.append({
                'holding_period_days': period,
                'roundtrip_cost': roundtrip_cost,
                'breakeven_return_pct': breakeven_return_pct,
                'annualized_breakeven_pct': annualized_breakeven * 100
            })
        
        return pd.DataFrame(results)
