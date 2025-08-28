"""Risk modeling module for covariance estimation and portfolio risk management."""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from sklearn.covariance import OAS, LedoitWolf


class RiskModel:
    """Risk model for covariance matrix estimation with multiple methods."""

    def __init__(
        self,
        method: Literal["sample", "ledoit_wolf", "oas", "ewma"] = "ledoit_wolf",
        ewma_lambda: float = 0.94,
        ridge: float = 1e-6,
        factor_model: Optional[str] = None,
        min_periods: int = 252,
    ):
        """
        Initialize risk model.

        Parameters
        ----------
        method : str
            Covariance estimation method
            - 'sample': Sample covariance
            - 'ledoit_wolf': Ledoit-Wolf shrinkage
            - 'oas': Oracle Approximating Shrinkage
            - 'ewma': Exponentially Weighted Moving Average
        ewma_lambda : float
            Decay parameter for EWMA (closer to 1 = more recent weight)
        ridge : float
            Ridge regularization parameter (added to diagonal)
        factor_model : str, optional
            Factor model type ('market', 'fama_french', etc.)
        min_periods : int
            Minimum periods required for estimation
        """
        self.method = method
        self.ewma_lambda = ewma_lambda
        self.ridge = ridge
        self.factor_model = factor_model
        self.min_periods = min_periods

        # Will be set after fitting
        self._cov_matrix = None
        self._returns = None
        self._fitted = False

    def fit(self, returns: pd.DataFrame) -> "RiskModel":
        """
        Fit the risk model to return data.

        Parameters
        ----------
        returns : pd.DataFrame
            Return data with shape (T, N) where T=time, N=assets

        Returns
        -------
        RiskModel
            Self for method chaining
        """
        if len(returns) < self.min_periods:
            raise ValueError(
                f"Need at least {self.min_periods} periods, got {len(returns)}"
            )

        self._returns = returns.copy()

        if self.method == "sample":
            self._cov_matrix = self._sample_cov(returns)
        elif self.method == "ledoit_wolf":
            self._cov_matrix = self._ledoit_wolf_cov(returns)
        elif self.method == "oas":
            self._cov_matrix = self._oas_cov(returns)
        elif self.method == "ewma":
            self._cov_matrix = self._ewma_cov(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply ridge regularization and ensure PSD
        self._cov_matrix = self._regularize_cov(self._cov_matrix)
        self._cov_matrix = self.nearest_psd(self._cov_matrix)

        self._fitted = True
        return self

    def cov(self) -> np.ndarray:
        """
        Get covariance matrix.

        Returns
        -------
        np.ndarray
            Covariance matrix (N, N)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        return self._cov_matrix.copy()

    def corr(self) -> np.ndarray:
        """
        Get correlation matrix.

        Returns
        -------
        np.ndarray
            Correlation matrix (N, N)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        std = np.sqrt(np.diag(self._cov_matrix))
        return self._cov_matrix / np.outer(std, std)

    def vol(self) -> pd.Series:
        """
        Get asset volatilities.

        Returns
        -------
        pd.Series
            Asset volatilities
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        return pd.Series(
            np.sqrt(np.diag(self._cov_matrix)), index=self._returns.columns
        )

    def _sample_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute sample covariance matrix."""
        return returns.cov().values

    def _ledoit_wolf_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute Ledoit-Wolf shrinkage covariance matrix."""
        lw = LedoitWolf()
        return lw.fit(returns.values).covariance_

    def _oas_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute Oracle Approximating Shrinkage covariance matrix."""
        oas = OAS()
        return oas.fit(returns.values).covariance_

    def _ewma_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute Exponentially Weighted Moving Average covariance matrix."""
        T, N = returns.shape
        returns_array = returns.values
        
        # Initialize with first observation
        cov_matrix = np.outer(returns_array[0], returns_array[0])
        
        # Exponentially weighted covariance update
        for t in range(1, T):
            r_t = returns_array[t]
            cov_matrix = (
                self.ewma_lambda * cov_matrix + 
                (1 - self.ewma_lambda) * np.outer(r_t, r_t)
            )
        
        # Bias correction (similar to pandas ewm)
        sum_weights = (1 - self.ewma_lambda ** T) / (1 - self.ewma_lambda)
        cov_matrix = cov_matrix / (sum_weights / T)
        
        return cov_matrix

    def _regularize_cov(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Apply ridge regularization to covariance matrix."""
        return cov_matrix + self.ridge * np.eye(cov_matrix.shape[0])

    def nearest_psd(self, matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Find the nearest positive semi-definite matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix
        eps : float
            Minimum eigenvalue threshold

        Returns
        -------
        np.ndarray
            Nearest PSD matrix
        """
        # Check if already PSD
        if np.min(eigvalsh(matrix)) >= -eps:
            return matrix

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)

        # Clip negative eigenvalues
        eigenvals = np.maximum(eigenvals, eps)

        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def risk_budget(
        self, weights: np.ndarray, risk_budgets: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """
        Calculate risk budgeting metrics.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        risk_budgets : np.ndarray, optional
            Target risk budgets (if None, equal budgets)

        Returns
        -------
        dict
            Risk budgeting metrics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        # Calculate marginal risk contributions
        portfolio_vol = np.sqrt(weights @ self._cov_matrix @ weights)
        marginal_contrib = (self._cov_matrix @ weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib

        # Risk contribution percentages
        risk_contrib_pct = risk_contrib / portfolio_vol

        if risk_budgets is None:
            risk_budgets = np.ones(len(weights)) / len(weights)

        return {
            "portfolio_vol": portfolio_vol,
            "marginal_contrib": marginal_contrib,
            "risk_contrib": risk_contrib,
            "risk_contrib_pct": risk_contrib_pct,
            "target_budgets": risk_budgets,
            "budget_diff": risk_contrib_pct - risk_budgets,
        }

    def factor_exposure(
        self, factors: pd.DataFrame, method: str = "ols"
    ) -> dict[str, Any]:
        """
        Calculate factor exposures and specific risk.

        Parameters
        ----------
        factors : pd.DataFrame
            Factor return data (T, F)
        method : str
            Estimation method ('ols', 'ridge', 'lasso')

        Returns
        -------
        dict
            Factor model results
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        # Align dates between returns and factors
        common_dates = self._returns.index.intersection(factors.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and factors")
        
        returns_aligned = self._returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        n_assets = returns_aligned.shape[1]
        n_factors = factors_aligned.shape[1]
        
        # Add intercept
        X = np.column_stack([np.ones(len(factors_aligned)), factors_aligned.values])
        Y = returns_aligned.values
        
        if method == "ols":
            # Ordinary Least Squares
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ Y)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                beta = np.linalg.pinv(X) @ Y
        elif method == "ridge":
            # Ridge regression with small regularization
            ridge_param = 1e-4
            XTX_ridge = X.T @ X + ridge_param * np.eye(X.shape[1])
            beta = np.linalg.solve(XTX_ridge, X.T @ Y)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ols' or 'ridge'")
        
        # Extract intercepts and factor loadings
        intercepts = beta[0, :]
        factor_loadings = beta[1:, :].T  # (n_assets, n_factors)
        
        # Calculate residuals and specific risk
        Y_pred = X @ beta
        residuals = Y - Y_pred
        specific_var = np.var(residuals, axis=0, ddof=X.shape[1])
        specific_risk = np.sqrt(specific_var)
        
        # Factor covariance matrix
        factor_cov = np.cov(factors_aligned.T)
        
        # R-squared calculation
        tss = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        rss = np.sum(residuals ** 2, axis=0)
        r_squared = 1 - rss / np.maximum(tss, 1e-10)
        
        return {
            "factor_loadings": pd.DataFrame(
                factor_loadings,
                index=returns_aligned.columns,
                columns=factors_aligned.columns
            ),
            "intercepts": pd.Series(intercepts, index=returns_aligned.columns),
            "specific_risk": pd.Series(specific_risk, index=returns_aligned.columns),
            "factor_cov": pd.DataFrame(
                factor_cov,
                index=factors_aligned.columns,
                columns=factors_aligned.columns
            ),
            "r_squared": pd.Series(r_squared, index=returns_aligned.columns),
            "residuals": pd.DataFrame(
                residuals,
                index=common_dates,
                columns=returns_aligned.columns
            )
        }
