"""Test cases for risk modeling module."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import eigvalsh

from krx_portfolio.models.risk import RiskModel


class TestRiskModel:
    """Test cases for RiskModel class."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        # Generate correlated returns
        returns = np.random.multivariate_normal(
            mean=[0.001, 0.0008, 0.0012, 0.002, 0.0015],
            cov=np.array(
                [
                    [0.0004, 0.0001, 0.0002, 0.0001, 0.0002],
                    [0.0001, 0.0003, 0.0001, 0.0001, 0.0001],
                    [0.0002, 0.0001, 0.0005, 0.0002, 0.0002],
                    [0.0001, 0.0001, 0.0002, 0.0008, 0.0003],
                    [0.0002, 0.0001, 0.0002, 0.0003, 0.0006],
                ]
            ),
            size=len(dates),
        )

        return pd.DataFrame(returns, index=dates, columns=assets)

    @pytest.fixture
    def risk_model(self):
        """Create risk model instance."""
        return RiskModel(method="sample", ridge=1e-6)

    def test_initialization(self):
        """Test risk model initialization."""
        model = RiskModel(
            method="ledoit_wolf", ewma_lambda=0.95, ridge=1e-5, min_periods=100
        )

        assert model.method == "ledoit_wolf"
        assert model.ewma_lambda == 0.95
        assert model.ridge == 1e-5
        assert model.min_periods == 100
        assert not model._fitted

    def test_sample_covariance_method(self, sample_returns):
        """Test sample covariance method."""
        model = RiskModel(method="sample")
        model.fit(sample_returns)

        cov_matrix = model.cov()
        expected_cov = sample_returns.cov().values

        # Should be close to pandas cov (with ridge regularization)
        assert cov_matrix.shape == (
            len(sample_returns.columns),
            len(sample_returns.columns),
        )
        assert np.allclose(
            cov_matrix,
            expected_cov + model.ridge * np.eye(len(sample_returns.columns)),
            atol=1e-8,
        )

    def test_ledoit_wolf_method(self, sample_returns):
        """Test Ledoit-Wolf shrinkage method."""
        model = RiskModel(method="ledoit_wolf")
        model.fit(sample_returns)

        cov_matrix = model.cov()
        assert cov_matrix.shape == (
            len(sample_returns.columns),
            len(sample_returns.columns),
        )
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric

    def test_oas_method(self, sample_returns):
        """Test Oracle Approximating Shrinkage method."""
        model = RiskModel(method="oas")
        model.fit(sample_returns)

        cov_matrix = model.cov()
        assert cov_matrix.shape == (
            len(sample_returns.columns),
            len(sample_returns.columns),
        )
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric

    def test_ewma_method(self, sample_returns):
        """Test EWMA method."""
        model = RiskModel(method="ewma", ewma_lambda=0.94)
        model.fit(sample_returns)

        cov_matrix = model.cov()
        assert cov_matrix.shape == (
            len(sample_returns.columns),
            len(sample_returns.columns),
        )
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric

    def test_unknown_method_raises_error(self, sample_returns):
        """Test that unknown method raises error."""
        model = RiskModel(method="unknown_method")
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(sample_returns)

    def test_covariance_is_psd(self, sample_returns):
        """Test that covariance matrix is positive semi-definite."""
        for method in ["sample", "ledoit_wolf", "oas", "ewma"]:
            model = RiskModel(method=method)
            model.fit(sample_returns)

            cov_matrix = model.cov()
            eigenvals = eigvalsh(cov_matrix)

            # All eigenvalues should be non-negative (within tolerance)
            assert (
                np.min(eigenvals) >= -1e-8
            ), f"Method {method} produced non-PSD matrix"

    def test_correlation_matrix(self, sample_returns):
        """Test correlation matrix calculation."""
        model = RiskModel()
        model.fit(sample_returns)

        corr_matrix = model.corr()

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)

        # Off-diagonal elements should be between -1 and 1
        off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        assert np.all(off_diag >= -1.0)
        assert np.all(off_diag <= 1.0)

    def test_volatility_calculation(self, sample_returns):
        """Test volatility calculation."""
        model = RiskModel()
        model.fit(sample_returns)

        vols = model.vol()

        assert len(vols) == len(sample_returns.columns)
        assert isinstance(vols, pd.Series)
        assert all(vols.index == sample_returns.columns)
        assert np.all(vols >= 0)  # Volatilities should be non-negative

    def test_nearest_psd_correction(self):
        """Test nearest PSD matrix correction."""
        model = RiskModel()

        # Create a non-PSD matrix
        non_psd = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.95], [0.8, 0.95, 1.0]])
        # Make it slightly non-PSD
        non_psd[2, 2] = 0.1

        psd_matrix = model.nearest_psd(non_psd)

        # Check that result is PSD
        eigenvals = eigvalsh(psd_matrix)
        assert np.min(eigenvals) >= -1e-8

    def test_ridge_regularization_effect(self, sample_returns):
        """Test effect of ridge regularization."""
        model_no_ridge = RiskModel(ridge=0.0)
        model_with_ridge = RiskModel(ridge=1e-3)

        model_no_ridge.fit(sample_returns)
        model_with_ridge.fit(sample_returns)

        cov_no_ridge = model_no_ridge.cov()
        cov_with_ridge = model_with_ridge.cov()

        # Diagonal elements should be larger with ridge
        diag_diff = np.diag(cov_with_ridge) - np.diag(cov_no_ridge)
        assert np.all(diag_diff >= 0)
        assert np.max(diag_diff) >= 1e-3 * 0.9  # Should be close to ridge value

    def test_ewma_lambda_sensitivity(self, sample_returns):
        """Test EWMA lambda parameter sensitivity."""
        model_low_lambda = RiskModel(method="ewma", ewma_lambda=0.5)
        model_high_lambda = RiskModel(method="ewma", ewma_lambda=0.99)

        model_low_lambda.fit(sample_returns)
        model_high_lambda.fit(sample_returns)

        # Different lambdas should produce different covariances
        cov_low = model_low_lambda.cov()
        cov_high = model_high_lambda.cov()

        assert not np.allclose(cov_low, cov_high, rtol=0.1)

    def test_minimum_periods_requirement(self, sample_returns):
        """Test minimum periods requirement."""
        short_returns = sample_returns.iloc[:10]  # Only 10 periods

        model = RiskModel(min_periods=50)
        with pytest.raises(ValueError, match="Need at least 50 periods"):
            model.fit(short_returns)

    def test_must_fit_before_using(self):
        """Test that model must be fitted before using."""
        model = RiskModel()

        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.cov()

        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.corr()

        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.vol()

    def test_risk_budget_calculation(self, sample_returns):
        """Test risk budgeting calculation."""
        model = RiskModel()
        model.fit(sample_returns)

        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
        risk_budget = model.risk_budget(weights)

        expected_keys = {
            "portfolio_vol",
            "marginal_contrib",
            "risk_contrib",
            "risk_contrib_pct",
            "target_budgets",
            "budget_diff",
        }
        assert set(risk_budget.keys()) == expected_keys

        # Risk contributions should sum to portfolio volatility
        assert (
            abs(np.sum(risk_budget["risk_contrib"]) - risk_budget["portfolio_vol"])
            < 1e-10
        )

        # Risk contribution percentages should sum to 1
        assert abs(np.sum(risk_budget["risk_contrib_pct"]) - 1.0) < 1e-10

    def test_factor_exposure_placeholder(self, sample_returns):
        """Test factor exposure method (placeholder implementation)."""
        model = RiskModel()
        model.fit(sample_returns)

        # Create dummy factor data
        factor_data = pd.DataFrame(
            {
                "market": sample_returns.mean(axis=1),
                "size": np.random.randn(len(sample_returns)),
            },
            index=sample_returns.index,
        )

        exposure = model.factor_exposure(factor_data)

        # Should return structure with expected keys (even if values are None)
        expected_keys = {"factor_loadings", "specific_risk", "factor_cov", "r_squared"}
        assert set(exposure.keys()) == expected_keys

    @pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "oas", "ewma"])
    def test_method_produces_valid_covariance(self, method, sample_returns):
        """Test that all methods produce valid covariance matrices."""
        model = RiskModel(method=method)
        model.fit(sample_returns)

        cov_matrix = model.cov()

        # Basic properties
        assert cov_matrix.shape == (5, 5)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
        assert np.min(eigvalsh(cov_matrix)) >= -1e-8  # PSD
        assert np.all(np.diag(cov_matrix) > 0)  # Positive diagonal

    def test_model_chaining(self, sample_returns):
        """Test that fit returns self for method chaining."""
        model = RiskModel()
        result = model.fit(sample_returns)

        assert result is model
        assert model._fitted

    def test_copy_behavior(self, sample_returns):
        """Test that cov() returns a copy, not reference."""
        model = RiskModel()
        model.fit(sample_returns)

        cov1 = model.cov()
        cov2 = model.cov()

        # Should be equal but not the same object
        assert np.array_equal(cov1, cov2)
        cov1[0, 0] = 999  # Modify first matrix
        assert cov1[0, 0] != cov2[0, 0]  # Second should be unchanged
