"""Tests for backtesting engine module."""

import numpy as np
import pandas as pd
import pytest

from krx_portfolio.backtesting.engine import BacktestEngine
from krx_portfolio.models.rebalance import Rebalancer


class TestBacktestEngine:
    """Test suite for BacktestEngine class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Create 100 days of data for 5 assets
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        # Generate correlated returns
        base_returns = np.random.multivariate_normal(
            mean=[0.0005, 0.0008, 0.0006, 0.0012, 0.0010],
            cov=np.array([
                [0.0004, 0.0001, 0.0001, 0.0002, 0.0001],
                [0.0001, 0.0006, 0.0002, 0.0001, 0.0002],
                [0.0001, 0.0002, 0.0003, 0.0001, 0.0001],
                [0.0002, 0.0001, 0.0001, 0.0009, 0.0003],
                [0.0001, 0.0002, 0.0001, 0.0003, 0.0008],
            ]),
            size=len(dates)
        )
        
        returns = pd.DataFrame(base_returns, index=dates, columns=assets)
        
        # Generate prices starting from 100
        prices = pd.DataFrame(index=dates, columns=assets)
        prices.iloc[0] = [100, 150, 120, 200, 180]
        
        for i in range(1, len(dates)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns.iloc[i])
        
        # Generate monthly rebalancing weights
        weight_dates = pd.date_range('2023-01-01', periods=100, freq='D')[::22]  # Roughly monthly
        weights_data = []
        
        for date in weight_dates:
            if date <= dates[-1]:
                # Random weights that sum to 1
                random_weights = np.random.dirichlet([1, 1, 1, 1, 1])
                weights_data.append(pd.Series(random_weights, index=assets, name=date))
        
        weights = pd.DataFrame(weights_data).fillna(method='ffill')
        
        return {
            'returns': returns,
            'prices': prices,
            'weights': weights,
            'dates': dates,
            'assets': assets
        }

    @pytest.fixture
    def basic_engine(self):
        """Create basic BacktestEngine for testing."""
        return BacktestEngine(
            initial_capital=1_000_000,
            transaction_cost_bps=25.0,
            market_impact_model="linear"
        )

    def test_engine_initialization(self, basic_engine):
        """Test BacktestEngine initialization."""
        assert basic_engine.initial_capital == 1_000_000
        assert basic_engine.transaction_cost_bps == 0.0025  # 25 bps in decimal
        assert basic_engine.market_impact_model == "linear"
        assert basic_engine.results == {}
        assert basic_engine.portfolio_history == []

    def test_data_alignment(self, basic_engine, sample_data):
        """Test data alignment functionality."""
        weights = sample_data['weights']
        returns = sample_data['returns']
        prices = sample_data['prices']
        
        # Test with overlapping data
        aligned_data = basic_engine._align_data(weights, returns, prices, None, None)
        assert aligned_data is not None
        
        weights_aligned, returns_aligned, prices_aligned, dates = aligned_data
        assert len(weights_aligned) == len(returns_aligned)
        assert len(weights_aligned) == len(dates)
        assert list(weights_aligned.columns) == list(returns_aligned.columns)

    def test_data_alignment_date_filter(self, basic_engine, sample_data):
        """Test data alignment with date filtering."""
        weights = sample_data['weights']
        returns = sample_data['returns']
        prices = sample_data['prices']
        
        start_date = pd.Timestamp('2023-01-15')
        end_date = pd.Timestamp('2023-02-15')
        
        aligned_data = basic_engine._align_data(weights, returns, prices, start_date, end_date)
        assert aligned_data is not None
        
        _, _, _, dates = aligned_data
        assert dates[0] >= start_date
        assert dates[-1] <= end_date

    def test_portfolio_initialization(self, basic_engine, sample_data):
        """Test portfolio state initialization."""
        initial_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], 
                                  index=sample_data['assets'])
        start_date = sample_data['dates'][0]
        
        portfolio_state = basic_engine._initialize_portfolio(initial_weights, start_date)
        
        assert portfolio_state['date'] == start_date
        assert portfolio_state['total_value'] == 1_000_000
        assert portfolio_state['cash'] == 0.0
        assert np.isclose(portfolio_state['weights'].sum(), 1.0)
        assert np.isclose(portfolio_state['positions'].sum(), 1_000_000)

    def test_apply_market_returns(self, basic_engine, sample_data):
        """Test market returns application."""
        initial_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], 
                                  index=sample_data['assets'])
        start_date = sample_data['dates'][0]
        
        portfolio_state = basic_engine._initialize_portfolio(initial_weights, start_date)
        daily_returns = sample_data['returns'].iloc[1]
        
        # Store original values for comparison
        original_total_value = portfolio_state['total_value']
        original_positions = portfolio_state['positions'].copy()
        
        updated_state = basic_engine._apply_market_returns(portfolio_state, daily_returns)
        
        # Use tolerance-based comparison for floating point precision
        assert not np.isclose(updated_state['total_value'], original_total_value, rtol=1e-10)
        assert 'daily_return' in updated_state
        assert 'cumulative_return' in updated_state
        
        # Check that positions changed (unless all returns are exactly 0)
        if not daily_returns.abs().sum() == 0:
            assert not np.allclose(updated_state['positions'], original_positions, rtol=1e-10)

    def test_should_rebalance(self, basic_engine, sample_data):
        """Test rebalancing decision logic."""
        dates = sample_data['dates']
        weights = sample_data['weights']
        
        rebalance_dates = [dates[10], dates[30], dates[60]]
        
        # Should rebalance on scheduled date
        assert basic_engine._should_rebalance(dates[10], rebalance_dates, 10, weights)
        
        # Should not rebalance on non-scheduled date
        assert not basic_engine._should_rebalance(dates[5], rebalance_dates, 5, weights)

    def test_calculate_market_impact_linear(self, basic_engine):
        """Test linear market impact calculation."""
        current_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
        target_weights = pd.Series([0.3, 0.1, 0.2, 0.2, 0.2])
        portfolio_value = 1_000_000
        
        impact = basic_engine._calculate_market_impact(
            current_weights, target_weights, portfolio_value
        )
        
        assert impact >= 0
        assert isinstance(impact, float)

    def test_calculate_market_impact_none(self):
        """Test no market impact model."""
        engine = BacktestEngine(market_impact_model="none")
        
        current_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
        target_weights = pd.Series([0.3, 0.1, 0.2, 0.2, 0.2])
        portfolio_value = 1_000_000
        
        impact = engine._calculate_market_impact(
            current_weights, target_weights, portfolio_value
        )
        
        assert impact == 0.0

    def test_full_backtest_basic(self, basic_engine, sample_data):
        """Test basic full backtest execution."""
        weights = sample_data['weights']
        returns = sample_data['returns']
        prices = sample_data['prices']
        
        results = basic_engine.run_backtest(
            weights=weights,
            returns=returns,
            prices=prices
        )
        
        # Check result structure
        assert 'portfolio_history' in results
        assert 'total_return' in results
        assert 'annualized_return' in results
        assert 'volatility' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        # Check portfolio history
        portfolio_df = results['portfolio_history']
        assert len(portfolio_df) > 0
        assert 'total_value' in portfolio_df.columns
        assert 'daily_return' in portfolio_df.columns
        assert 'cumulative_return' in portfolio_df.columns

    def test_backtest_with_custom_rebalancer(self, sample_data):
        """Test backtesting with custom rebalancer."""
        custom_rebalancer = Rebalancer(
            schedule="month_end",
            turnover_budget=0.3,
            tc_bps=30.0
        )
        
        engine = BacktestEngine(
            initial_capital=500_000,
            rebalancer=custom_rebalancer
        )
        
        results = engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        assert results['initial_capital'] == 500_000
        assert len(results['portfolio_history']) > 0

    def test_backtest_with_benchmark(self, basic_engine, sample_data):
        """Test backtesting with benchmark comparison."""
        # Create benchmark returns (simple equal-weight portfolio)
        benchmark_returns = sample_data['returns'].mean(axis=1)
        basic_engine.benchmark_returns = benchmark_returns
        
        results = basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        assert 'benchmark_total_return' in results
        assert 'excess_return' in results
        assert 'tracking_error' in results
        assert 'information_ratio' in results

    def test_backtest_date_range_filtering(self, basic_engine, sample_data):
        """Test backtesting with date range filtering."""
        start_date = pd.Timestamp('2023-01-15')
        end_date = pd.Timestamp('2023-02-15')
        
        results = basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices'],
            start_date=start_date,
            end_date=end_date
        )
        
        portfolio_history = results['portfolio_history']
        assert portfolio_history.index[0] >= start_date
        assert portfolio_history.index[-1] <= end_date

    def test_portfolio_weights_extraction(self, basic_engine, sample_data):
        """Test portfolio weights extraction after backtest."""
        basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        weights_df = basic_engine.get_portfolio_weights_over_time()
        assert isinstance(weights_df, pd.DataFrame)
        assert len(weights_df.columns) == len(sample_data['assets'])
        assert len(weights_df) > 0

    def test_results_compilation_metrics(self, basic_engine, sample_data):
        """Test that compiled results contain expected metrics."""
        basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        results = basic_engine.results
        
        # Basic performance metrics
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio'
        ]
        
        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))
            assert not np.isnan(results[metric])

    def test_transaction_cost_tracking(self, basic_engine, sample_data):
        """Test transaction cost tracking."""
        results = basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        assert 'total_transaction_costs' in results
        assert 'cost_ratio' in results
        assert 'average_turnover' in results
        assert results['total_transaction_costs'] >= 0
        assert results['cost_ratio'] >= 0

    def test_rebalance_history_tracking(self, basic_engine, sample_data):
        """Test rebalancing history tracking."""
        basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        rebalance_df = basic_engine.results['rebalance_history']
        
        if len(rebalance_df) > 0:
            required_cols = ['date', 'target_weights', 'executed_weights', 
                           'transaction_cost', 'turnover']
            
            for col in required_cols:
                assert col in rebalance_df.columns

    def test_edge_case_no_overlapping_data(self, basic_engine):
        """Test handling of no overlapping data."""
        # Create non-overlapping data
        weights = pd.DataFrame(
            [[0.5, 0.5]], 
            index=[pd.Timestamp('2023-01-01')],
            columns=['A', 'B']
        )
        
        returns = pd.DataFrame(
            [[0.01, -0.01]], 
            index=[pd.Timestamp('2024-01-01')],  # Different year
            columns=['A', 'B']
        )
        
        with pytest.raises(ValueError, match="No overlapping data found"):
            basic_engine.run_backtest(weights=weights, returns=returns)

    def test_edge_case_single_asset(self, basic_engine):
        """Test backtesting with single asset."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, 50),
            index=dates,
            columns=['SINGLE_ASSET']
        )
        
        weights = pd.DataFrame(
            [[1.0]] * len(dates),  # Daily weights
            index=dates,
            columns=['SINGLE_ASSET']
        )
        
        results = basic_engine.run_backtest(weights=weights, returns=returns)
        
        assert 'total_return' in results
        assert len(results['portfolio_history']) == len(dates)

    def test_zero_returns_handling(self, basic_engine):
        """Test handling of zero returns."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        assets = ['A', 'B']
        
        # All zero returns
        returns = pd.DataFrame(
            np.zeros((30, 2)),
            index=dates,
            columns=assets
        )
        
        weights = pd.DataFrame(
            [[0.5, 0.5]] * 3,
            index=dates[::10],
            columns=assets
        )
        
        results = basic_engine.run_backtest(weights=weights, returns=returns)
        
        assert results['total_return'] == 0.0
        assert results['volatility'] == 0.0
        assert results['final_value'] == results['initial_capital']

    def test_export_results(self, basic_engine, sample_data, tmp_path):
        """Test results export functionality."""
        basic_engine.run_backtest(
            weights=sample_data['weights'],
            returns=sample_data['returns'],
            prices=sample_data['prices']
        )
        
        # Export to temporary directory
        basic_engine.export_results(str(tmp_path))
        
        # Check that files were created
        expected_files = ['portfolio_history.csv', 'summary_metrics.yaml']
        
        for filename in expected_files:
            assert (tmp_path / filename).exists()

    def test_numerical_stability(self, basic_engine):
        """Test numerical stability with extreme values."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Create returns with some extreme values
        extreme_returns = pd.DataFrame({
            'A': [0.5, -0.9, 0.01] + [0.001] * 17,  # Large gain, then large loss
            'B': [0.001] * 20,  # Stable asset
        }, index=dates)
        
        weights = pd.DataFrame({
            'A': [0.5] * 2,
            'B': [0.5] * 2,
        }, index=[dates[0], dates[10]])
        
        results = basic_engine.run_backtest(weights=weights, returns=extreme_returns)
        
        # Check that results are finite and reasonable
        assert np.isfinite(results['total_return'])
        assert np.isfinite(results['volatility'])
        assert results['final_value'] > 0