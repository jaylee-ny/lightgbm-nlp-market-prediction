import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import RiskMetrics, PerformanceEvaluator


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.randn(252) * 0.01
    
    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = RiskMetrics.sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert -10 < sharpe < 10
    
    def test_sharpe_ratio_with_risk_free_rate(self, sample_returns):
        """Test Sharpe with non-zero risk-free rate."""
        sharpe = RiskMetrics.sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = RiskMetrics.sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        dd_metrics = RiskMetrics.max_drawdown(sample_returns)
        
        assert 'max_drawdown' in dd_metrics
        assert 'max_drawdown_duration' in dd_metrics
        assert 'current_drawdown' in dd_metrics
        
        assert dd_metrics['max_drawdown'] >= 0
        assert dd_metrics['max_drawdown_duration'] >= 0
    
    def test_max_drawdown_known_case(self):
        """Test max drawdown with known values."""
        returns = np.array([0.1, 0.05, -0.15, -0.1, 0.2])
        
        dd_metrics = RiskMetrics.max_drawdown(returns)
        
        assert dd_metrics['max_drawdown'] > 0
        assert dd_metrics['max_drawdown'] < 0.3
    
    def test_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = RiskMetrics.calmar_ratio(sample_returns)
        
        assert isinstance(calmar, (float, np.floating))
    
    def test_value_at_risk(self, sample_returns):
        """Test VaR calculation."""
        var_95 = RiskMetrics.value_at_risk(sample_returns, 0.95)
        
        assert var_95 > 0
        assert var_95 < 0.1
    
    def test_conditional_var(self, sample_returns):
        """Test CVaR calculation."""
        cvar_95 = RiskMetrics.conditional_value_at_risk(sample_returns, 0.95)
        
        assert cvar_95 > 0
        
        var_95 = RiskMetrics.value_at_risk(sample_returns, 0.95)
        assert cvar_95 >= var_95
    
    def test_omega_ratio(self, sample_returns):
        """Test Omega ratio calculation."""
        omega = RiskMetrics.omega_ratio(sample_returns)
        
        assert isinstance(omega, (float, np.floating))
        assert omega > 0
    
    def test_information_ratio(self):
        """Test Information Ratio calculation."""
        np.random.seed(42)
        strategy_returns = np.random.randn(252) * 0.01 + 0.0005
        benchmark_returns = np.random.randn(252) * 0.01
        
        ir = RiskMetrics.information_ratio(strategy_returns, benchmark_returns)
        
        assert isinstance(ir, float)
    
    def test_tail_ratio(self, sample_returns):
        """Test tail ratio calculation."""
        tail = RiskMetrics.tail_ratio(sample_returns)
        
        assert tail > 0
    
    def test_downside_capture(self):
        """Test downside capture ratio."""
        np.random.seed(42)
        strategy_returns = np.random.randn(252) * 0.01
        benchmark_returns = np.random.randn(252) * 0.015
        
        dc = RiskMetrics.downside_capture(strategy_returns, benchmark_returns)
        
        assert isinstance(dc, (float, np.floating))
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        empty_returns = np.array([])
        
        assert np.isnan(RiskMetrics.sharpe_ratio(empty_returns))
        assert np.isnan(RiskMetrics.sortino_ratio(empty_returns))
        dd = RiskMetrics.max_drawdown(empty_returns)
        assert np.isnan(dd['max_drawdown'])
    
    def test_zero_volatility_handling(self):
        """Test handling of zero volatility."""
        constant_returns = np.ones(252) * 0.001
        
        sharpe = RiskMetrics.sharpe_ratio(constant_returns)
        assert np.isnan(sharpe)


class TestPerformanceEvaluator:
    """Test comprehensive performance evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return PerformanceEvaluator(periods_per_year=252, risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.randn(252) * 0.01 + 0.0003
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator can be initialized."""
        assert evaluator.periods_per_year == 252
        assert evaluator.risk_free_rate == 0.02
    
    def test_evaluate_strategy_basic(self, evaluator, sample_returns):
        """Test basic strategy evaluation."""
        metrics = evaluator.evaluate_strategy(sample_returns)
        
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
    
    def test_evaluate_with_benchmark(self, evaluator, sample_returns):
        """Test evaluation with benchmark."""
        np.random.seed(42)
        benchmark_returns = np.random.randn(252) * 0.01
        
        metrics = evaluator.evaluate_strategy(sample_returns, benchmark_returns)
        
        assert 'information_ratio' in metrics
        assert 'beta' in metrics
        assert 'alpha' in metrics
        assert 'downside_capture' in metrics
    
    def test_summary_report_creation(self, evaluator, sample_returns):
        """Test summary report generation."""
        metrics = evaluator.evaluate_strategy(sample_returns)
        
        report = evaluator.create_summary_report(metrics)
        
        assert isinstance(report, str)
        assert 'STRATEGY PERFORMANCE SUMMARY' in report
        assert 'Sharpe Ratio' in report
        assert 'Max Drawdown' in report
    
    def test_compare_strategies(self, evaluator):
        """Test strategy comparison."""
        np.random.seed(42)
        
        strategies = {
            'Strategy A': np.random.randn(252) * 0.01 + 0.0005,
            'Strategy B': np.random.randn(252) * 0.015 + 0.0003,
            'Strategy C': np.random.randn(252) * 0.008 + 0.0007
        }
        
        comparison = evaluator.compare_strategies(strategies)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert 'strategy' in comparison.columns
        assert 'sharpe_ratio' in comparison.columns


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_positive_strategy_metrics(self):
        """Test metrics for profitable strategy."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001
        
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_strategy(returns)
        
        assert metrics['total_return'] > 0
        assert metrics['sharpe_ratio'] > 0
        assert metrics['win_rate'] > 0.5
    
    def test_high_volatility_impact(self):
        """Test metrics for high volatility strategy."""
        np.random.seed(42)
        low_vol = np.random.randn(252) * 0.005 + 0.0005
        high_vol = np.random.randn(252) * 0.03 + 0.0005
        
        evaluator = PerformanceEvaluator()
        
        metrics_low = evaluator.evaluate_strategy(low_vol)
        metrics_high = evaluator.evaluate_strategy(high_vol)
        
        assert metrics_low['annual_volatility'] < metrics_high['annual_volatility']
        assert metrics_low['sharpe_ratio'] > metrics_high['sharpe_ratio']
    
    def test_drawdown_recovery(self):
        """Test drawdown metrics during recovery."""
        returns = np.array(
            [0.01] * 50 +
            [-0.02] * 30 +
            [0.015] * 50
        )
        
        dd_metrics = RiskMetrics.max_drawdown(returns)
        
        assert dd_metrics['max_drawdown'] > 0
        assert dd_metrics['current_drawdown'] < dd_metrics['max_drawdown']
    
    def test_realistic_market_returns(self):
        """Test with realistic market-like returns."""
        np.random.seed(42)
        
        drift = 0.0003
        volatility = 0.01
        returns = np.random.randn(252 * 3) * volatility + drift
        
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_strategy(returns)
        
        assert 0 < metrics['annual_return'] < 0.5
        assert 0 < metrics['annual_volatility'] < 0.5
        assert -2 < metrics['sharpe_ratio'] < 3
        assert 0 < metrics['max_drawdown'] < 0.5
    
    def test_benchmark_comparison(self):
        """Test strategy vs benchmark comparison."""
        np.random.seed(42)
        
        benchmark = np.random.randn(252) * 0.01 + 0.0003
        
        better_strategy = benchmark + np.random.randn(252) * 0.003 + 0.0002
        worse_strategy = benchmark + np.random.randn(252) * 0.015 - 0.0001
        
        evaluator = PerformanceEvaluator()
        
        better_metrics = evaluator.evaluate_strategy(better_strategy, benchmark)
        worse_metrics = evaluator.evaluate_strategy(worse_strategy, benchmark)
        
        assert better_metrics['information_ratio'] > worse_metrics['information_ratio']
        assert better_metrics['alpha'] > worse_metrics['alpha']
