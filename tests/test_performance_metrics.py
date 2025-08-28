"""Tests for performance metrics module."""

import numpy as np
import pandas as pd
import pytest

from krx_portfolio.backtesting.metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Generate returns with some known statistical properties
        returns = pd.Series(
            np.random.normal(0.0008, 0.015, 252),  # ~20% annual return, 15% daily vol
            index=dates,
            name='portfolio'
        )
        
        return returns

    @pytest.fixture
    def sample_benchmark(self):
        """Create sample benchmark data."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        benchmark = pd.Series(
            np.random.normal(0.0004, 0.012, 252),  # ~10% annual return, 12% daily vol
            index=dates,
            name='benchmark'
        )
        
        return benchmark

    @pytest.fixture
    def basic_metrics(self):
        """Create basic PerformanceMetrics instance."""
        return PerformanceMetrics(risk_free_rate=0.02, confidence_level=0.05)

    def test_metrics_initialization(self, basic_metrics):
        """Test PerformanceMetrics initialization."""
        assert basic_metrics.risk_free_rate == 0.02
        assert basic_metrics.confidence_level == 0.05

    def test_calculate_return_metrics(self, basic_metrics, sample_returns):
        """Test basic return metrics calculation."""
        metrics = basic_metrics._calculate_return_metrics(sample_returns)
        
        expected_keys = [
            'total_return', 'annualized_return', 'arithmetic_mean', 
            'geometric_mean', 'best_day', 'worst_day', 'positive_days', 'negative_days'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert np.isfinite(metrics[key])
        
        # Sanity checks
        assert 0 <= metrics['positive_days'] <= 1
        assert 0 <= metrics['negative_days'] <= 1
        assert np.isclose(metrics['positive_days'] + metrics['negative_days'], 1.0, atol=0.01)

    def test_calculate_risk_metrics(self, basic_metrics, sample_returns):
        """Test risk metrics calculation."""
        metrics = basic_metrics._calculate_risk_metrics(sample_returns)
        
        expected_keys = [
            'volatility', 'downside_volatility', 'semi_volatility', 'volatility_daily'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert metrics[key] >= 0
        
        # Volatility should be reasonable (daily vol * sqrt(252))
        expected_annual_vol = sample_returns.std() * np.sqrt(252)
        assert np.isclose(metrics['volatility'], expected_annual_vol, rtol=0.01)

    def test_calculate_risk_adjusted_metrics(self, basic_metrics, sample_returns):
        """Test risk-adjusted metrics calculation."""
        metrics = basic_metrics._calculate_risk_adjusted_metrics(sample_returns)
        
        expected_keys = ['sharpe_ratio', 'sortino_ratio']
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert np.isfinite(metrics[key])

    def test_calculate_drawdown_metrics(self, basic_metrics):
        """Test drawdown metrics calculation."""
        # Create portfolio values with known drawdown
        values = pd.Series([100, 110, 105, 95, 90, 100, 115, 120])
        
        metrics = basic_metrics._calculate_drawdown_metrics(values)
        
        expected_keys = [
            'max_drawdown', 'avg_drawdown', 'max_drawdown_duration', 
            'avg_drawdown_duration', 'calmar_ratio', 'recovery_factor'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Max drawdown should be negative
        assert metrics['max_drawdown'] <= 0
        
        # Duration should be non-negative integers
        assert metrics['max_drawdown_duration'] >= 0
        assert metrics['avg_drawdown_duration'] >= 0

    def test_calculate_benchmark_metrics(self, basic_metrics, sample_returns, sample_benchmark):
        """Test benchmark comparison metrics."""
        metrics = basic_metrics._calculate_benchmark_metrics(sample_returns, sample_benchmark)
        
        expected_keys = [
            'information_ratio', 'tracking_error', 'beta', 'alpha',
            'up_capture_ratio', 'down_capture_ratio'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert np.isfinite(metrics[key])
        
        # Beta should be reasonable
        assert -5 <= metrics['beta'] <= 5  # Extreme but reasonable bounds
        
        # Tracking error should be positive
        assert metrics['tracking_error'] >= 0

    def test_calculate_moment_metrics(self, basic_metrics, sample_returns):
        """Test higher moment metrics calculation."""
        metrics = basic_metrics._calculate_moment_metrics(sample_returns)
        
        expected_keys = ['skewness', 'kurtosis', 'jarque_bera_stat']
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_calculate_tail_risk_metrics(self, basic_metrics, sample_returns):
        """Test tail risk metrics calculation."""
        metrics = basic_metrics._calculate_tail_risk_metrics(sample_returns)
        
        expected_keys = [
            'var_95', 'var_99', 'cvar_95', 'cvar_99', 'max_loss', 'gain_to_pain_ratio'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # VaR should be negative (representing losses)
        assert metrics['var_95'] <= 0
        assert metrics['var_99'] <= 0
        
        # CVaR should be more negative than VaR
        assert metrics['cvar_95'] <= metrics['var_95']
        assert metrics['cvar_99'] <= metrics['var_99']
        
        # Max loss should be the minimum return
        assert metrics['max_loss'] == sample_returns.min()

    def test_calculate_all_metrics_basic(self, basic_metrics, sample_returns):
        """Test comprehensive metrics calculation."""
        all_metrics = basic_metrics.calculate_all_metrics(sample_returns)
        
        # Should contain metrics from all categories
        expected_categories = [
            'total_return', 'volatility', 'sharpe_ratio', 
            'max_drawdown', 'skewness', 'var_95'
        ]
        
        for key in expected_categories:
            assert key in all_metrics
        
        # Check that we have a reasonable number of metrics
        assert len(all_metrics) >= 20

    def test_calculate_all_metrics_with_benchmark(self, basic_metrics, sample_returns, sample_benchmark):
        """Test comprehensive metrics with benchmark."""
        all_metrics = basic_metrics.calculate_all_metrics(
            sample_returns, benchmark_returns=sample_benchmark
        )
        
        # Should contain benchmark-specific metrics
        benchmark_metrics = ['information_ratio', 'tracking_error', 'beta', 'alpha']
        
        for key in benchmark_metrics:
            assert key in all_metrics

    def test_calculate_all_metrics_with_portfolio_values(self, basic_metrics, sample_returns):
        """Test metrics calculation with portfolio values."""
        # Generate portfolio values from returns
        portfolio_values = (1 + sample_returns).cumprod() * 100000
        
        all_metrics = basic_metrics.calculate_all_metrics(
            sample_returns, portfolio_values=portfolio_values
        )
        
        # Should contain drawdown metrics
        drawdown_metrics = ['max_drawdown', 'calmar_ratio']
        
        for key in drawdown_metrics:
            assert key in all_metrics

    def test_frequency_inference_daily(self, basic_metrics):
        """Test frequency inference for daily data."""
        daily_returns = pd.Series(
            [0.01] * 252,
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        freq = basic_metrics._infer_frequency(daily_returns)
        assert freq == 252

    def test_frequency_inference_monthly(self, basic_metrics):
        """Test frequency inference for monthly data."""
        monthly_returns = pd.Series(
            [0.02] * 12,
            index=pd.date_range('2023-01-01', periods=12, freq='M')
        )
        
        freq = basic_metrics._infer_frequency(monthly_returns)
        assert freq == 12

    def test_rolling_metrics(self, basic_metrics, sample_returns):
        """Test rolling metrics calculation."""
        window = 30
        rolling_results = basic_metrics.rolling_metrics(
            sample_returns, 
            window=window,
            metrics=['sharpe_ratio', 'volatility']
        )
        
        assert isinstance(rolling_results, pd.DataFrame)
        assert 'sharpe_ratio' in rolling_results.columns
        assert 'volatility' in rolling_results.columns
        assert len(rolling_results) == len(sample_returns)
        
        # First (window-1) values should be NaN
        assert rolling_results.iloc[:window-1].isna().all().all()
        
        # Later values should be non-NaN
        assert not rolling_results.iloc[window:].isna().all().any()

    def test_performance_attribution(self, basic_metrics):
        """Test performance attribution analysis."""
        # Create sample portfolio and asset data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['A', 'B', 'C']
        
        # Portfolio returns
        portfolio_returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=dates
        )
        
        # Asset weights
        weights = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], 100),
            index=dates,
            columns=assets
        )
        
        # Asset returns
        asset_returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            index=dates,
            columns=assets
        )
        
        attribution = basic_metrics.performance_attribution(
            portfolio_returns, weights, asset_returns
        )
        
        assert isinstance(attribution, pd.DataFrame)
        assert 'avg_weight' in attribution.columns
        assert 'total_contribution' in attribution.columns
        assert len(attribution) == len(assets)

    def test_edge_case_zero_volatility(self, basic_metrics):
        """Test handling of zero volatility."""
        # Constant returns (zero volatility)
        constant_returns = pd.Series([0.001] * 100)
        
        metrics = basic_metrics._calculate_risk_adjusted_metrics(constant_returns)
        
        # Should handle zero volatility gracefully
        assert metrics['sharpe_ratio'] == 0.0

    def test_edge_case_all_positive_returns(self, basic_metrics):
        """Test handling of all positive returns."""
        positive_returns = pd.Series(np.abs(np.random.normal(0.01, 0.005, 100)))
        
        metrics = basic_metrics._calculate_risk_adjusted_metrics(positive_returns)
        
        # Should handle case with no negative returns
        assert np.isfinite(metrics['sortino_ratio'])
        assert metrics['sortino_ratio'] >= 0

    def test_edge_case_all_negative_returns(self, basic_metrics):
        """Test handling of all negative returns."""
        negative_returns = pd.Series(-np.abs(np.random.normal(0.01, 0.005, 100)))
        
        metrics = basic_metrics.calculate_all_metrics(negative_returns)
        
        # Should handle all negative returns
        assert metrics['total_return'] < 0
        assert metrics['positive_days'] == 0
        assert metrics['negative_days'] == 1.0

    def test_edge_case_single_observation(self, basic_metrics):
        """Test handling of single observation."""
        single_return = pd.Series([0.05])
        
        # Should handle gracefully (though some metrics will be undefined)
        metrics = basic_metrics._calculate_return_metrics(single_return)
        
        assert metrics['total_return'] == 0.05
        assert metrics['best_day'] == 0.05
        assert metrics['worst_day'] == 0.05

    def test_edge_case_empty_series(self, basic_metrics):
        """Test handling of empty series."""
        empty_returns = pd.Series([], dtype=float)
        
        # Should handle empty series gracefully
        with pytest.raises((ValueError, IndexError)):
            basic_metrics.calculate_all_metrics(empty_returns)

    def test_missing_data_handling(self, basic_metrics):
        """Test handling of missing data."""
        # Returns with some NaN values
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, -0.01, np.nan, 0.005])
        
        metrics = basic_metrics.calculate_all_metrics(returns_with_nan)
        
        # Should handle NaN values by dropping them
        assert np.isfinite(metrics['total_return'])
        assert np.isfinite(metrics['volatility'])

    def test_extreme_outliers(self, basic_metrics):
        """Test handling of extreme outliers."""
        # Returns with extreme outliers
        outlier_returns = pd.Series([0.001] * 98 + [5.0, -3.0])  # 500% and -300% days
        
        metrics = basic_metrics.calculate_all_metrics(outlier_returns)
        
        # Should handle extreme values without crashing
        assert np.isfinite(metrics['total_return'])
        assert np.isfinite(metrics['volatility'])
        assert metrics['best_day'] == 5.0
        assert metrics['worst_day'] == -3.0

    def test_different_confidence_levels(self):
        """Test metrics calculation with different confidence levels."""
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        # Test different confidence levels
        for confidence_level in [0.01, 0.05, 0.10]:
            metrics_calc = PerformanceMetrics(confidence_level=confidence_level)
            metrics = metrics_calc.calculate_all_metrics(returns)
            
            # VaR should be different for different confidence levels
            assert 'var_95' in metrics or 'var_99' in metrics or 'var_90' in metrics

    def test_numerical_precision(self, basic_metrics):
        """Test numerical precision with very small numbers."""
        # Very small returns
        tiny_returns = pd.Series(np.random.normal(0, 1e-8, 100))
        
        metrics = basic_metrics.calculate_all_metrics(tiny_returns)
        
        # Should handle small numbers without numerical issues
        assert np.isfinite(metrics['volatility'])
        assert np.isfinite(metrics['sharpe_ratio'])

    def test_correlation_with_known_formulas(self, basic_metrics):
        """Test that calculated metrics match known formulas."""
        # Create returns with known properties
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        metrics = basic_metrics.calculate_all_metrics(returns)
        
        # Verify total return calculation
        expected_total_return = (1 + returns).prod() - 1
        assert np.isclose(metrics['total_return'], expected_total_return)
        
        # Verify volatility calculation
        expected_volatility = returns.std() * np.sqrt(252)
        assert np.isclose(metrics['volatility'], expected_volatility)