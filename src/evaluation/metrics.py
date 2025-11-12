from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class RiskMetrics:
    """
    Calculate risk and performance metrics.
    
    All return-based metrics assume daily returns unless specified.
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        risk_free_rate : float
            Annual risk-free rate (e.g., 0.02 for 2%)
        periods_per_year : int
            Number of periods per year (252 for daily)
            
        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return np.nan
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return np.nan
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        return sharpe * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio.
        
        Like Sharpe but only penalizes downside volatility.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        risk_free_rate : float
            Annual risk-free rate
        periods_per_year : int
            Number of periods per year
            
        Returns
        -------
        float
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return np.nan
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else np.nan
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return np.nan
        
        sortino = np.mean(excess_returns) / downside_std
        
        return sortino * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
            
        Returns
        -------
        Dict[str, float]
            max_drawdown: Maximum drawdown (positive number)
            max_drawdown_duration: Duration in periods
            current_drawdown: Current drawdown
        """
        if len(returns) == 0:
            return {
                'max_drawdown': np.nan,
                'max_drawdown_duration': 0,
                'current_drawdown': np.nan
            }
        
        cumulative = np.cumprod(1 + returns)
        
        running_max = np.maximum.accumulate(cumulative)
        
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = abs(np.min(drawdown))
        
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(cumulative[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        dd_duration = max_dd_idx - peak_idx
        
        current_dd = abs(drawdown[-1])
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': dd_duration,
            'current_drawdown': current_dd
        }
    
    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        periods_per_year : int
            Number of periods per year
            
        Returns
        -------
        float
            Calmar ratio
        """
        if len(returns) == 0:
            return np.nan
        
        annual_return = np.mean(returns) * periods_per_year
        
        dd_metrics = RiskMetrics.max_drawdown(returns)
        max_dd = dd_metrics['max_drawdown']
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else np.nan
        
        return annual_return / max_dd
    
    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
            
        Returns
        -------
        float
            VaR (positive number representing potential loss)
        """
        if len(returns) == 0:
            return np.nan
        
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        
        return var
    
    @staticmethod
    def conditional_value_at_risk(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Average loss in worst (1 - confidence_level) cases.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        confidence_level : float
            Confidence level
            
        Returns
        -------
        float
            CVaR (positive number)
        """
        if len(returns) == 0:
            return np.nan
        
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        
        worst_returns = returns[returns <= -var]
        
        if len(worst_returns) == 0:
            return var
        
        cvar = -np.mean(worst_returns)
        
        return cvar
    
    @staticmethod
    def omega_ratio(
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.
        
        Ratio of probability-weighted gains to losses relative to threshold.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
        threshold : float
            Return threshold
            
        Returns
        -------
        float
            Omega ratio
        """
        if len(returns) == 0:
            return np.nan
        
        excess_returns = returns - threshold
        
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]
        
        if len(losses) == 0:
            return np.inf if len(gains) > 0 else np.nan
        
        gain_sum = np.sum(gains)
        loss_sum = abs(np.sum(losses))
        
        if loss_sum == 0:
            return np.inf
        
        return gain_sum / loss_sum
    
    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio.
        
        Measures excess return per unit of tracking error.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns
        periods_per_year : int
            Number of periods per year
            
        Returns
        -------
        float
            Annualized Information Ratio
        """
        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")
        
        if len(returns) == 0:
            return np.nan
        
        active_returns = returns - benchmark_returns
        
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return np.nan
        
        ir = np.mean(active_returns) / tracking_error
        
        return ir * np.sqrt(periods_per_year)
    
    @staticmethod
    def tail_ratio(returns: np.ndarray) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).
        
        Measures asymmetry in tails.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns
            
        Returns
        -------
        float
            Tail ratio
        """
        if len(returns) == 0:
            return np.nan
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return np.nan
        
        return abs(p95 / p5)
    
    @staticmethod
    def downside_capture(
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate downside capture ratio.
        
        Strategy return during benchmark down periods / benchmark return.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray
            Benchmark returns
            
        Returns
        -------
        float
            Downside capture ratio (lower is better)
        """
        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")
        
        down_periods = benchmark_returns < 0
        
        if not down_periods.any():
            return np.nan
        
        strategy_down = np.mean(returns[down_periods])
        benchmark_down = np.mean(benchmark_returns[down_periods])
        
        if benchmark_down == 0:
            return np.nan
        
        return strategy_down / benchmark_down


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation.
    
    Combines multiple metrics and provides summary reports.
    """
    
    def __init__(
        self,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        periods_per_year : int
            Number of periods per year
        risk_free_rate : float
            Annual risk-free rate
        """
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
    
    def evaluate_strategy(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a strategy.
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns
        benchmark_returns : np.ndarray, optional
            Benchmark returns for relative metrics
            
        Returns
        -------
        Dict[str, float]
            Dictionary of all metrics
        """
        logger.info("Evaluating strategy performance...")
        
        metrics = {}
        
        # Basic statistics
        metrics['total_return'] = np.prod(1 + returns) - 1
        metrics['annual_return'] = np.mean(returns) * self.periods_per_year
        metrics['annual_volatility'] = np.std(returns) * np.sqrt(self.periods_per_year)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = RiskMetrics.sharpe_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )
        metrics['sortino_ratio'] = RiskMetrics.sortino_ratio(
            returns, self.risk_free_rate, self.periods_per_year
        )
        metrics['calmar_ratio'] = RiskMetrics.calmar_ratio(
            returns, self.periods_per_year
        )
        
        # Drawdown metrics
        dd_metrics = RiskMetrics.max_drawdown(returns)
        metrics.update(dd_metrics)
        
        # Risk metrics
        metrics['value_at_risk_95'] = RiskMetrics.value_at_risk(returns, 0.95)
        metrics['cvar_95'] = RiskMetrics.conditional_value_at_risk(returns, 0.95)
        metrics['omega_ratio'] = RiskMetrics.omega_ratio(returns)
        metrics['tail_ratio'] = RiskMetrics.tail_ratio(returns)
        
        # Win rate
        winning_periods = np.sum(returns > 0)
        metrics['win_rate'] = winning_periods / len(returns) if len(returns) > 0 else np.nan
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        metrics['avg_win'] = np.mean(wins) if len(wins) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = (
            abs(metrics['avg_win'] / metrics['avg_loss'])
            if metrics['avg_loss'] != 0 else np.nan
        )
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            metrics['information_ratio'] = RiskMetrics.information_ratio(
                returns, benchmark_returns, self.periods_per_year
            )
            metrics['downside_capture'] = RiskMetrics.downside_capture(
                returns, benchmark_returns
            )
            
            # Beta and alpha
            if len(returns) == len(benchmark_returns) and len(returns) > 1:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                
                if benchmark_variance > 0:
                    metrics['beta'] = covariance / benchmark_variance
                    
                    benchmark_annual_return = np.mean(benchmark_returns) * self.periods_per_year
                    metrics['alpha'] = (
                        metrics['annual_return'] - 
                        (self.risk_free_rate + metrics['beta'] * 
                         (benchmark_annual_return - self.risk_free_rate))
                    )
                else:
                    metrics['beta'] = np.nan
                    metrics['alpha'] = np.nan
        
        logger.info(f"Evaluation complete. Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def create_summary_report(
        self,
        metrics: Dict[str, float]
    ) -> str:
        """
        Create human-readable summary report.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary from evaluate_strategy
            
        Returns
        -------
        str
            Formatted report
        """
        report_lines = [
            "=" * 60,
            "STRATEGY PERFORMANCE SUMMARY",
            "=" * 60,
            "",
            "Return Metrics:",
            f"  Total Return:        {metrics.get('total_return', np.nan)*100:>8.2f}%",
            f"  Annual Return:       {metrics.get('annual_return', np.nan)*100:>8.2f}%",
            f"  Annual Volatility:   {metrics.get('annual_volatility', np.nan)*100:>8.2f}%",
            "",
            "Risk-Adjusted Returns:",
            f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', np.nan):>8.2f}",
            f"  Sortino Ratio:       {metrics.get('sortino_ratio', np.nan):>8.2f}",
            f"  Calmar Ratio:        {metrics.get('calmar_ratio', np.nan):>8.2f}",
            "",
            "Drawdown Metrics:",
            f"  Max Drawdown:        {metrics.get('max_drawdown', np.nan)*100:>8.2f}%",
            f"  DD Duration (days):  {metrics.get('max_drawdown_duration', 0):>8.0f}",
            f"  Current Drawdown:    {metrics.get('current_drawdown', np.nan)*100:>8.2f}%",
            "",
            "Risk Metrics:",
            f"  VaR (95%):          {metrics.get('value_at_risk_95', np.nan)*100:>8.2f}%",
            f"  CVaR (95%):         {metrics.get('cvar_95', np.nan)*100:>8.2f}%",
            f"  Omega Ratio:        {metrics.get('omega_ratio', np.nan):>8.2f}",
            "",
            "Win/Loss Statistics:",
            f"  Win Rate:           {metrics.get('win_rate', np.nan)*100:>8.2f}%",
            f"  Avg Win:            {metrics.get('avg_win', np.nan)*100:>8.2f}%",
            f"  Avg Loss:           {metrics.get('avg_loss', np.nan)*100:>8.2f}%",
            f"  Win/Loss Ratio:     {metrics.get('win_loss_ratio', np.nan):>8.2f}",
            ""
        ]
        
        # Add benchmark-relative metrics if available
        if 'information_ratio' in metrics:
            report_lines.extend([
                "Benchmark-Relative Metrics:",
                f"  Information Ratio:  {metrics.get('information_ratio', np.nan):>8.2f}",
                f"  Beta:              {metrics.get('beta', np.nan):>8.2f}",
                f"  Alpha:             {metrics.get('alpha', np.nan)*100:>8.2f}%",
                f"  Downside Capture:  {metrics.get('downside_capture', np.nan)*100:>8.2f}%",
                ""
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def compare_strategies(
        self,
        strategy_returns: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Parameters
        ----------
        strategy_returns : Dict[str, np.ndarray]
            Dictionary mapping strategy names to returns
            
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        results = []
        
        for name, returns in strategy_returns.items():
            metrics = self.evaluate_strategy(returns)
            metrics['strategy'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Reorder columns to put strategy name first
        cols = ['strategy'] + [col for col in df.columns if col != 'strategy']
        df = df[cols]
        
        return df
