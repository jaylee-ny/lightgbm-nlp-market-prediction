import pytest
import numpy as np
import pandas as pd

from src.backtest.costs import (
    TransactionCostModel,
    PortfolioRebalancer,
    CostAnalyzer
)


class TestTransactionCostModel:
    """Test transaction cost calculations."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = TransactionCostModel(
            commission_bps=1.0,
            spread_bps=5.0,
            market_impact_coef=0.1
        )
        
        assert model.commission_bps == 1.0
        assert model.spread_bps == 5.0
        assert model.market_impact_coef == 0.1
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        model = TransactionCostModel(commission_bps=10.0)
        
        cost = model._calculate_commission(10000)
        
        assert cost == 10.0
    
    def test_spread_calculation(self):
        """Test bid-ask spread cost."""
        model = TransactionCostModel(spread_bps=5.0)
        
        cost = model._calculate_spread_cost(10000)
        
        assert cost == 5.0
    
    def test_market_impact_calculation(self):
        """Test market impact with square-root model."""
        model = TransactionCostModel(market_impact_coef=0.1)
        
        position_value = 100000
        daily_volume = 10000000
        
        impact = model._calculate_market_impact(position_value, daily_volume)
        
        assert impact > 0
        assert impact < position_value * 0.01
    
    def test_total_cost_calculation(self):
        """Test total cost includes all components."""
        model = TransactionCostModel(
            commission_bps=1.0,
            spread_bps=5.0,
            market_impact_coef=0.1
        )
        
        costs = model.calculate_total_cost(
            position_value=100000,
            daily_volume=10000000
        )
        
        assert 'commission' in costs
        assert 'spread' in costs
        assert 'market_impact' in costs
        assert 'total' in costs
        assert 'total_bps' in costs
        
        assert costs['total'] > 0
        assert costs['total'] == (
            costs['commission'] + 
            costs['spread'] + 
            costs['market_impact']
        )
    
    def test_high_participation_rate(self):
        """Test market impact with high participation rate."""
        model = TransactionCostModel(market_impact_coef=0.1)
        
        position_value = 5000000
        daily_volume = 10000000
        
        impact = model._calculate_market_impact(position_value, daily_volume)
        
        participation = position_value / daily_volume
        assert participation == 0.5
        assert impact > 0
    
    def test_roundtrip_cost(self):
        """Test roundtrip cost calculation."""
        model = TransactionCostModel(
            commission_bps=1.0,
            spread_bps=5.0,
            market_impact_coef=0.1
        )
        
        roundtrip = model.calculate_roundtrip_cost(
            position_value=100000,
            daily_volume=10000000
        )
        
        single_trip = model.calculate_total_cost(
            position_value=100000,
            daily_volume=10000000
        )
        
        assert roundtrip == single_trip['total'] * 2
    
    def test_zero_volume_handling(self):
        """Test handling of zero daily volume."""
        model = TransactionCostModel()
        
        impact = model._calculate_market_impact(
            position_value=100000,
            daily_volume=0
        )
        
        assert impact > 0


class TestPortfolioRebalancer:
    """Test portfolio rebalancing with costs."""
    
    @pytest.fixture
    def cost_model(self):
        """Create cost model for testing."""
        return TransactionCostModel(
            commission_bps=1.0,
            spread_bps=5.0,
            market_impact_coef=0.1
        )
    
    def test_rebalancer_initialization(self, cost_model):
        """Test rebalancer can be initialized."""
        rebalancer = PortfolioRebalancer(cost_model)
        
        assert rebalancer.cost_model is not None
        assert rebalancer.current_positions == {}
    
    def test_calculate_rebalance_costs_from_zero(self, cost_model):
        """Test rebalancing from zero positions."""
        rebalancer = PortfolioRebalancer(cost_model)
        
        target_positions = {
            'AAPL': 100,
            'GOOGL': 50
        }
        
        prices = {
            'AAPL': 150.0,
            'GOOGL': 120.0
        }
        
        volumes = {
            'AAPL': 1e8,
            'GOOGL': 5e7
        }
        
        result = rebalancer.calculate_rebalance_costs(
            target_positions,
            prices,
            volumes
        )
        
        assert result['total_cost'] > 0
        assert result['n_trades'] == 2
        assert len(result['trades']) == 2
    
    def test_calculate_rebalance_costs_with_existing(self, cost_model):
        """Test rebalancing with existing positions."""
        rebalancer = PortfolioRebalancer(cost_model)
        rebalancer.current_positions = {'AAPL': 50}
        
        target_positions = {
            'AAPL': 100,
            'GOOGL': 50
        }
        
        prices = {
            'AAPL': 150.0,
            'GOOGL': 120.0
        }
        
        volumes = {
            'AAPL': 1e8,
            'GOOGL': 5e7
        }
        
        result = rebalancer.calculate_rebalance_costs(
            target_positions,
            prices,
            volumes
        )
        
        assert result['n_trades'] == 2
        
        aapl_trade = [t for t in result['trades'] if t['asset'] == 'AAPL'][0]
        assert aapl_trade['shares_traded'] == 50
    
    def test_update_positions(self, cost_model):
        """Test updating positions after rebalance."""
        rebalancer = PortfolioRebalancer(cost_model)
        
        new_positions = {'AAPL': 100, 'GOOGL': 50}
        rebalancer.update_positions(new_positions)
        
        assert rebalancer.current_positions == new_positions
    
    def test_no_trade_when_at_target(self, cost_model):
        """Test no trades when already at target."""
        rebalancer = PortfolioRebalancer(cost_model)
        rebalancer.current_positions = {'AAPL': 100}
        
        target_positions = {'AAPL': 100}
        prices = {'AAPL': 150.0}
        volumes = {'AAPL': 1e8}
        
        result = rebalancer.calculate_rebalance_costs(
            target_positions,
            prices,
            volumes
        )
        
        assert result['n_trades'] == 0
        assert result['total_cost'] == 0


class TestCostAnalyzer:
    """Test cost analysis functions."""
    
    def test_analyze_turnover(self):
        """Test turnover analysis."""
        predictions = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10).repeat(3),
            'asset': ['AAPL', 'GOOGL', 'MSFT'] * 10,
            'prediction': np.random.uniform(0, 1, 30)
        })
        
        turnover = CostAnalyzer.analyze_turnover(predictions, threshold=0.5)
        
        assert 'mean_turnover' in turnover
        assert 'median_turnover' in turnover
        assert 'max_turnover' in turnover
        assert 'total_trades' in turnover
        
        assert turnover['mean_turnover'] >= 0
    
    def test_estimate_total_costs(self):
        """Test total cost estimation."""
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        turnover_rate = 0.5
        
        cost_model = TransactionCostModel()
        
        costs = CostAnalyzer.estimate_total_costs(
            returns,
            turnover_rate,
            cost_model
        )
        
        assert 'total_costs' in costs
        assert 'gross_return' in costs
        assert 'net_return' in costs
        assert 'cost_drag' in costs
        
        assert costs['gross_return'] > costs['net_return']
        assert costs['cost_drag'] > 0
    
    def test_breakeven_analysis(self):
        """Test breakeven analysis."""
        cost_model = TransactionCostModel()
        
        holding_periods = np.array([1, 5, 10, 20, 60])
        
        analysis = CostAnalyzer.breakeven_analysis(
            cost_model,
            holding_periods
        )
        
        assert len(analysis) == len(holding_periods)
        assert 'holding_period_days' in analysis.columns
        assert 'breakeven_return_pct' in analysis.columns
        assert 'annualized_breakeven_pct' in analysis.columns
        
        assert (analysis['breakeven_return_pct'] > 0).all()


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_realistic_cost_scenario(self):
        """Test costs with realistic trading scenario."""
        cost_model = TransactionCostModel(
            commission_bps=1.0,
            spread_bps=5.0,
            market_impact_coef=0.1
        )
        
        rebalancer = PortfolioRebalancer(cost_model)
        
        initial_positions = {
            'AAPL': 1000,
            'GOOGL': 500,
            'MSFT': 750
        }
        rebalancer.update_positions(initial_positions)
        
        target_positions = {
            'AAPL': 800,
            'GOOGL': 600,
            'MSFT': 750,
            'TSLA': 200
        }
        
        prices = {
            'AAPL': 150.0,
            'GOOGL': 120.0,
            'MSFT': 300.0,
            'TSLA': 250.0
        }
        
        volumes = {
            'AAPL': 1e9,
            'GOOGL': 5e8,
            'MSFT': 8e8,
            'TSLA': 3e8
        }
        
        result = rebalancer.calculate_rebalance_costs(
            target_positions,
            prices,
            volumes
        )
        
        assert result['n_trades'] == 3
        assert result['total_cost'] > 0
        
        portfolio_value = sum(
            initial_positions[asset] * prices[asset]
            for asset in initial_positions
        )
        
        cost_pct = (result['total_cost'] / portfolio_value) * 100
        assert cost_pct < 1.0
    
    def test_high_frequency_cost_impact(self):
        """Test cost impact of high-frequency trading."""
        cost_model = TransactionCostModel(
            commission_bps=0.5,
            spread_bps=3.0,
            market_impact_coef=0.05
        )
        
        daily_returns = np.random.randn(252) * 0.01
        turnover_rate = 2.0
        
        costs = CostAnalyzer.estimate_total_costs(
            daily_returns,
            turnover_rate,
            cost_model,
            portfolio_value=1e6
        )
        
        assert costs['cost_drag'] > 0
        
        gross_sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        net_returns = daily_returns - (costs['cost_per_period'] / 1e6)
        net_sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
        
        assert net_sharpe < gross_sharpe
