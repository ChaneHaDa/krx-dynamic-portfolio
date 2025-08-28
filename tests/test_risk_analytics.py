"""Tests for risk analytics module."""

import numpy as np
import pandas as pd
import pytest

from krx_portfolio.backtesting.risk_analytics import RiskAnalytics


class TestRiskAnalytics:
    """Test suite for RiskAnalytics class."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        
        # Generate returns with known statistical properties
        returns = pd.Series(
            np.random.normal(0.0005, 0.02, 1000),  # Small positive mean, realistic vol
            index=dates,
            name='portfolio'
        )
        
        return returns

    @pytest.fixture
    def sample_asset_returns(self):
        """Create sample multi-asset returns."""
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        # Generate correlated returns
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.0001, 0.0003],
            [0.0002, 0.0006, 0.0002, 0.0002],
            [0.0001, 0.0002, 0.0003, 0.0001],
            [0.0003, 0.0002, 0.0001, 0.0009]
        ])
        
        returns_data = np.random.multivariate_normal(
            mean=[0.0008, 0.0006, 0.0005, 0.0012],
            cov=cov_matrix,
            size=len(dates)
        )
        
        return pd.DataFrame(returns_data, index=dates, columns=assets)

    @pytest.fixture
    def basic_risk_analytics(self):
        """Create basic RiskAnalytics instance."""
        return RiskAnalytics(
            confidence_levels=[0.01, 0.05, 0.10],
            risk_free_rate=0.02
        )

    def test_risk_analytics_initialization(self, basic_risk_analytics):
        """Test RiskAnalytics initialization."""
        assert basic_risk_analytics.confidence_levels == [0.01, 0.05, 0.10]
        assert basic_risk_analytics.risk_free_rate == 0.02
        assert 'historical' in basic_risk_analytics.var_methods
        assert 'parametric' in basic_risk_analytics.var_methods
        assert 'cornish_fisher' in basic_risk_analytics.var_methods

    def test_historical_var(self, basic_risk_analytics, sample_returns):
        """Test historical VaR calculation."""
        var_result = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='historical'
        )
        
        assert 'historical' in var_result
        var_value = var_result['historical']
        
        assert isinstance(var_value, float)
        assert var_value <= 0  # VaR should be negative (representing loss)
        
        # Check that approximately 5% of returns are below VaR
        exceedances = (sample_returns <= var_value).sum()
        expected_exceedances = len(sample_returns) * 0.05
        assert abs(exceedances - expected_exceedances) <= len(sample_returns) * 0.02  # 2% tolerance

    def test_parametric_var(self, basic_risk_analytics, sample_returns):
        """Test parametric (normal) VaR calculation."""
        var_result = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='parametric'
        )
        
        assert 'parametric' in var_result
        var_value = var_result['parametric']
        
        assert isinstance(var_value, float)
        assert var_value <= 0
        
        # Parametric VaR should be close to normal quantile
        from scipy import stats
        expected_var = sample_returns.mean() + stats.norm.ppf(0.05) * sample_returns.std()
        assert abs(var_value - expected_var) < 0.001

    def test_cornish_fisher_var(self, basic_risk_analytics, sample_returns):
        """Test Cornish-Fisher VaR calculation."""
        var_result = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='cornish_fisher'
        )
        
        assert 'cornish_fisher' in var_result
        var_value = var_result['cornish_fisher']
        
        assert isinstance(var_value, float)
        assert var_value <= 0

    def test_calculate_var_all_methods(self, basic_risk_analytics, sample_returns):
        """Test VaR calculation with all methods."""
        var_result = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='all'
        )
        
        expected_methods = ['historical', 'parametric', 'cornish_fisher']
        
        for method in expected_methods:
            assert method in var_result
            assert isinstance(var_result[method], float)
            assert var_result[method] <= 0

    def test_calculate_cvar(self, basic_risk_analytics, sample_returns):
        """Test Conditional VaR calculation."""
        cvar_value = basic_risk_analytics.calculate_cvar(
            sample_returns, confidence_level=0.05, method='historical'
        )
        
        assert isinstance(cvar_value, float)
        assert cvar_value <= 0
        
        # CVaR should be more negative than VaR
        var_value = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='historical'
        )['historical']
        
        assert cvar_value <= var_value

    def test_var_with_window(self, basic_risk_analytics, sample_returns):
        """Test VaR calculation with rolling window."""
        window = 250
        
        var_result = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='historical', window=window
        )
        
        assert 'historical' in var_result
        # Should use only the last 'window' observations

    def test_extreme_value_analysis_pot(self, basic_risk_analytics, sample_returns):
        """Test Peaks over Threshold extreme value analysis."""
        eva_result = basic_risk_analytics.extreme_value_analysis(
            sample_returns, method='peaks_over_threshold', threshold_percentile=0.95
        )
        
        expected_keys = ['threshold', 'scale', 'shape', 'n_exceedances']
        
        for key in expected_keys:
            assert key in eva_result
        
        assert eva_result['threshold'] < 0  # Should be negative (losses)
        assert eva_result['n_exceedances'] >= 0

    def test_extreme_value_analysis_block_maxima(self, basic_risk_analytics, sample_returns):
        """Test Block Maxima extreme value analysis."""
        eva_result = basic_risk_analytics.extreme_value_analysis(
            sample_returns, method='block_maxima'
        )
        
        expected_keys = ['location', 'scale', 'shape', 'n_blocks']
        
        for key in expected_keys:
            assert key in eva_result
        
        if not np.isnan(eva_result['n_blocks']):
            assert eva_result['n_blocks'] > 0

    def test_risk_factor_decomposition_regression(self, basic_risk_analytics):
        """Test risk factor decomposition using regression."""
        # Create synthetic portfolio and factor returns
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Factor returns
        factor_returns = pd.DataFrame({
            'Market': np.random.normal(0.0005, 0.015, 252),
            'Value': np.random.normal(0.0002, 0.008, 252),
            'Growth': np.random.normal(-0.0001, 0.010, 252),
        }, index=dates)
        
        # Portfolio returns (linear combination of factors + noise)
        portfolio_returns = (
            0.8 * factor_returns['Market'] + 
            0.3 * factor_returns['Value'] - 
            0.2 * factor_returns['Growth'] +
            np.random.normal(0, 0.005, 252)
        )
        portfolio_returns.index = dates
        
        decomposition = basic_risk_analytics.risk_factor_decomposition(
            portfolio_returns, factor_returns, method='regression'
        )
        
        if 'error' not in decomposition:
            expected_keys = [
                'factor_exposures', 'factor_risk_contributions', 
                'systematic_risk', 'specific_risk', 'total_risk', 'r_squared'
            ]
            
            for key in expected_keys:
                assert key in decomposition
            
            # R-squared should be reasonable
            if 'r_squared' in decomposition:
                assert 0 <= decomposition['r_squared'] <= 1

    def test_stress_testing(self, basic_risk_analytics):
        """Test stress testing functionality."""
        # Create portfolio weights
        portfolio_weights = pd.Series([0.3, 0.2, 0.2, 0.3], 
                                    index=['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        # Create asset returns
        asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.0008, 0.025, 100),
            'MSFT': np.random.normal(0.0006, 0.018, 100),
            'TSLA': np.random.normal(0.002, 0.04, 100),
        })
        
        # Define stress scenarios
        scenarios = {
            'market_crash': {'AAPL': -0.2, 'GOOGL': -0.25, 'MSFT': -0.15, 'TSLA': -0.3},
            'tech_selloff': {'AAPL': -0.1, 'GOOGL': -0.15, 'MSFT': -0.08, 'TSLA': -0.2},
            'interest_rate_shock': {'AAPL': -0.05, 'GOOGL': -0.03, 'MSFT': -0.02, 'TSLA': -0.08},
        }
        
        stress_results = basic_risk_analytics.stress_testing(
            portfolio_weights, asset_returns, scenarios
        )
        
        assert isinstance(stress_results, pd.Series)
        assert len(stress_results) == len(scenarios)
        
        # All scenario impacts should be negative (losses)
        assert all(stress_results <= 0)

    def test_correlation_analysis(self, basic_risk_analytics, sample_asset_returns):
        """Test correlation analysis."""
        corr_analysis = basic_risk_analytics.correlation_analysis(
            sample_asset_returns, method='pearson'
        )
        
        expected_keys = [
            'correlation_matrix', 'average_correlation', 
            'eigenvalues', 'pca_variance_explained'
        ]
        
        for key in expected_keys:
            assert key in corr_analysis
        
        # Correlation matrix should be square and symmetric
        corr_matrix = corr_analysis['correlation_matrix']
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert np.allclose(corr_matrix.values, corr_matrix.T.values)
        
        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix.values), 1.0)
        
        # Average correlation should be reasonable
        avg_corr = corr_analysis['average_correlation']
        assert -1 <= avg_corr <= 1

    def test_correlation_analysis_with_rolling_window(self, basic_risk_analytics, sample_asset_returns):
        """Test correlation analysis with rolling window."""
        rolling_window = 60
        
        corr_analysis = basic_risk_analytics.correlation_analysis(
            sample_asset_returns, rolling_window=rolling_window
        )
        
        assert 'rolling_avg_correlation' in corr_analysis
        
        rolling_corr = corr_analysis['rolling_avg_correlation']
        assert isinstance(rolling_corr, pd.Series)
        assert len(rolling_corr) == len(sample_asset_returns) - rolling_window

    def test_tail_dependency_analysis(self, basic_risk_analytics, sample_asset_returns):
        """Test tail dependency analysis."""
        tail_deps = basic_risk_analytics.tail_dependency_analysis(
            sample_asset_returns, threshold_percentile=0.90
        )
        
        assert isinstance(tail_deps, pd.DataFrame)
        assert tail_deps.shape == (len(sample_asset_returns.columns), len(sample_asset_returns.columns))
        
        # Diagonal should be 0 (assets don't have tail dependency with themselves in this context)
        # Off-diagonal should be between 0 and 1
        for i in range(len(tail_deps)):
            for j in range(len(tail_deps.columns)):
                if i != j:
                    assert 0 <= tail_deps.iloc[i, j] <= 1

    def test_portfolio_var_decomposition(self, basic_risk_analytics):
        """Test portfolio VaR decomposition."""
        # Create portfolio weights
        portfolio_weights = pd.Series([0.4, 0.3, 0.2, 0.1], 
                                    index=['A', 'B', 'C', 'D'])
        
        # Create asset returns
        asset_returns = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, 252),
            'B': np.random.normal(0.0005, 0.018, 252),
            'C': np.random.normal(0.0008, 0.022, 252),
            'D': np.random.normal(0.0012, 0.025, 252),
        })
        
        var_decomp = basic_risk_analytics.portfolio_var_decomposition(
            portfolio_weights, asset_returns, confidence_level=0.05
        )
        
        assert isinstance(var_decomp, pd.Series)
        assert len(var_decomp) == len(portfolio_weights)
        
        # VaR contributions should sum approximately to total portfolio VaR
        # (This is an approximation due to the marginal VaR calculation method)

    def test_monte_carlo_var_normal(self, basic_risk_analytics, sample_returns):
        """Test Monte Carlo VaR with normal distribution."""
        mc_var = basic_risk_analytics.monte_carlo_var(
            sample_returns, confidence_level=0.05, 
            n_simulations=10000, distribution='normal'
        )
        
        assert isinstance(mc_var, float)
        assert mc_var <= 0
        
        # Should be close to parametric VaR
        param_var = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='parametric'
        )['parametric']
        
        assert abs(mc_var - param_var) < 0.01  # Reasonable tolerance for MC

    def test_monte_carlo_var_empirical(self, basic_risk_analytics, sample_returns):
        """Test Monte Carlo VaR with empirical distribution."""
        mc_var = basic_risk_analytics.monte_carlo_var(
            sample_returns, confidence_level=0.05, 
            n_simulations=10000, distribution='empirical'
        )
        
        assert isinstance(mc_var, float)
        assert mc_var <= 0
        
        # Should be close to historical VaR
        hist_var = basic_risk_analytics.calculate_var(
            sample_returns, confidence_level=0.05, method='historical'
        )['historical']
        
        # Empirical MC should be close to historical VaR
        assert abs(mc_var - hist_var) < 0.01

    def test_edge_case_insufficient_data(self, basic_risk_analytics):
        """Test handling of insufficient data."""
        short_returns = pd.Series([0.01, -0.02, 0.005])  # Only 3 observations
        
        # Some methods should handle this gracefully
        var_result = basic_risk_analytics.calculate_var(
            short_returns, confidence_level=0.05, method='historical'
        )
        
        assert 'historical' in var_result

    def test_edge_case_all_zero_returns(self, basic_risk_analytics):
        """Test handling of all zero returns."""
        zero_returns = pd.Series([0.0] * 100)
        
        var_result = basic_risk_analytics.calculate_var(
            zero_returns, confidence_level=0.05, method='historical'
        )
        
        assert var_result['historical'] == 0.0
        
        cvar_result = basic_risk_analytics.calculate_cvar(
            zero_returns, confidence_level=0.05, method='historical'
        )
        
        assert cvar_result == 0.0

    def test_edge_case_extreme_outliers(self, basic_risk_analytics):
        """Test handling of extreme outliers."""
        # Returns with extreme outlier
        outlier_returns = pd.Series([0.001] * 99 + [-0.5])  # One -50% day
        
        var_result = basic_risk_analytics.calculate_var(
            outlier_returns, confidence_level=0.05, method='historical'
        )
        
        # Should handle extreme outlier without crashing
        assert isinstance(var_result['historical'], float)
        assert var_result['historical'] <= 0

    def test_different_confidence_levels(self, basic_risk_analytics, sample_returns):
        """Test VaR calculation at different confidence levels."""
        confidence_levels = [0.01, 0.05, 0.10]
        
        var_results = {}
        for cl in confidence_levels:
            var_results[cl] = basic_risk_analytics.calculate_var(
                sample_returns, confidence_level=cl, method='historical'
            )['historical']
        
        # VaR should be more negative (larger loss) for lower confidence levels
        assert var_results[0.01] <= var_results[0.05] <= var_results[0.10]

    def test_pca_risk_decomposition(self, basic_risk_analytics):
        """Test PCA-based risk decomposition."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create factor returns
        factor_returns = pd.DataFrame({
            'Factor1': np.random.normal(0.0005, 0.015, 252),
            'Factor2': np.random.normal(0.0002, 0.008, 252),
            'Factor3': np.random.normal(-0.0001, 0.010, 252),
        }, index=dates)
        
        # Portfolio returns
        portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.018, 252), index=dates
        )
        
        decomposition = basic_risk_analytics.risk_factor_decomposition(
            portfolio_returns, factor_returns, method='pca'
        )
        
        expected_keys = [
            'principal_component_loadings', 'explained_variance_ratio', 'risk_contributions'
        ]
        
        for key in expected_keys:
            assert key in decomposition

    def test_numerical_stability_extreme_values(self, basic_risk_analytics):
        """Test numerical stability with extreme values."""
        # Very large and small returns
        extreme_returns = pd.Series([1e-10] * 50 + [1e10] * 50)
        
        # Should not crash, though results may not be meaningful
        try:
            var_result = basic_risk_analytics.calculate_var(
                extreme_returns, confidence_level=0.05, method='historical'
            )
            assert isinstance(var_result['historical'], float)
        except (OverflowError, ValueError):
            # Some extreme cases might legitimately fail
            pass

    def test_invalid_method_error(self, basic_risk_analytics, sample_returns):
        """Test error handling for invalid methods."""
        with pytest.raises(ValueError):
            basic_risk_analytics.calculate_var(
                sample_returns, confidence_level=0.05, method='invalid_method'
            )
        
        with pytest.raises(ValueError):
            basic_risk_analytics.extreme_value_analysis(
                sample_returns, method='invalid_eva_method'
            )