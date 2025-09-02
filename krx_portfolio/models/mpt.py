"""Modern Portfolio Theory (MPT) optimization module."""

from typing import Optional

import numpy as np
from scipy.optimize import minimize


class MPTOptimizer:
    """Modern Portfolio Theory optimizer with multiple objective functions."""

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 0.1),
        rf: float = 0.0,
        sector_caps: Optional[dict[str, float]] = None,
        turnover_budget: Optional[float] = None,
        penalty: Optional[str] = None,
    ):
        """
        Initialize MPT optimizer.

        Parameters
        ----------
        bounds : tuple
            (lower, upper) bounds for individual asset weights
        rf : float
            Risk-free rate for Sharpe ratio calculation
        sector_caps : dict, optional
            Sector-wise weight caps {sector: max_weight}
        turnover_budget : float, optional
            Maximum turnover allowed (sum of absolute weight changes)
        penalty : str, optional
            Penalty type for turnover ('l1', 'l2')
        """
        self.bounds = bounds
        self.rf = rf
        self.sector_caps = sector_caps or {}
        self.turnover_budget = turnover_budget
        self.penalty = penalty

    def max_sharpe(
        self, mu: np.ndarray, sigma: np.ndarray, w_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Maximize Sharpe ratio: argmax_w (μᵀw - rf) / sqrt(wᵀΣw).

        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector (N,)
        sigma : np.ndarray
            Covariance matrix (N, N)
        w_prev : np.ndarray, optional
            Previous weights for turnover penalty

        Returns
        -------
        np.ndarray
            Optimal weights (N,)
        """
        n_assets = len(mu)
        
        def neg_sharpe(w: np.ndarray) -> float:
            """Negative Sharpe ratio (for minimization)."""
            portfolio_return = np.dot(w, mu)
            portfolio_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            if portfolio_vol == 0:
                return -np.inf
            sharpe = (portfolio_return - self.rf) / portfolio_vol
            
            # Add turnover penalty if specified
            penalty = 0.0
            if w_prev is not None and self.turnover_budget is not None:
                turnover = np.sum(np.abs(w - w_prev))
                if self.penalty == "l1":
                    penalty = 0.1 * max(0, turnover - self.turnover_budget)
                elif self.penalty == "l2":
                    penalty = 0.1 * max(0, turnover - self.turnover_budget) ** 2
            
            return -sharpe + penalty
        
        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        # Bounds
        bounds = [(self.bounds[0], self.bounds[1]) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False}
        )
        
        if not result.success:
            # Fallback to equal weights if optimization fails
            return self._apply_constraints(w0, w_prev)
        
        return self._apply_constraints(result.x, w_prev)

    def min_variance(
        self, mu: np.ndarray, sigma: np.ndarray, w_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Minimize portfolio variance: argmin_w wᵀΣw.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector (N,)
        sigma : np.ndarray
            Covariance matrix (N, N)
        w_prev : np.ndarray, optional
            Previous weights for turnover penalty

        Returns
        -------
        np.ndarray
            Optimal weights (N,)
        """
        n_assets = len(mu)
        
        def portfolio_variance(w):
            """Portfolio variance with optional turnover penalty."""
            variance = np.dot(w, np.dot(sigma, w))
            
            # Add turnover penalty if specified
            penalty = 0.0
            if w_prev is not None and self.turnover_budget is not None:
                turnover = np.sum(np.abs(w - w_prev))
                if self.penalty == "l1":
                    penalty = 0.1 * max(0, turnover - self.turnover_budget)
                elif self.penalty == "l2":
                    penalty = 0.1 * max(0, turnover - self.turnover_budget) ** 2
            
            return variance + penalty
        
        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        # Bounds
        bounds = [(self.bounds[0], self.bounds[1]) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False}
        )
        
        if not result.success:
            return self._apply_constraints(w0, w_prev)
        
        return self._apply_constraints(result.x, w_prev)

    def mean_variance(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: Optional[float] = None,
        target_return: Optional[float] = None,
        w_prev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Mean-variance optimization: argmin_w λ·wᵀΣw - (1-λ)·μᵀw.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector (N,)
        sigma : np.ndarray
            Covariance matrix (N, N)
        risk_aversion : float, optional
            Risk aversion parameter λ ∈ [0, 1]
        target_return : float, optional
            Target return constraint (alternative to risk_aversion)
        w_prev : np.ndarray, optional
            Previous weights for turnover penalty

        Returns
        -------
        np.ndarray
            Optimal weights (N,)
        """
        if risk_aversion is None and target_return is None:
            raise ValueError("Either risk_aversion or target_return must be specified")

        n_assets = len(mu)
        
        if risk_aversion is not None:
            # Risk aversion approach: minimize λ·variance - (1-λ)·return
            def objective(w):
                """Mean-variance objective with risk aversion parameter."""
                portfolio_return = np.dot(w, mu)
                portfolio_var = np.dot(w, np.dot(sigma, w))
                
                # Add turnover penalty if specified
                penalty = 0.0
                if w_prev is not None and self.turnover_budget is not None:
                    turnover = np.sum(np.abs(w - w_prev))
                    if self.penalty == "l1":
                        penalty = 0.1 * max(0, turnover - self.turnover_budget)
                    elif self.penalty == "l2":
                        penalty = 0.1 * max(0, turnover - self.turnover_budget) ** 2
                
                return risk_aversion * portfolio_var - (1 - risk_aversion) * portfolio_return + penalty
            
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            
        else:
            # Target return approach: minimize variance subject to return constraint
            def objective(w):
                """Minimize variance with turnover penalty."""
                portfolio_var = np.dot(w, np.dot(sigma, w))
                
                penalty = 0.0
                if w_prev is not None and self.turnover_budget is not None:
                    turnover = np.sum(np.abs(w - w_prev))
                    if self.penalty == "l1":
                        penalty = 0.1 * max(0, turnover - self.turnover_budget)
                    elif self.penalty == "l2":
                        penalty = 0.1 * max(0, turnover - self.turnover_budget) ** 2
                
                return portfolio_var + penalty
            
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: np.dot(w, mu) - target_return}
            ]
        
        # Bounds
        bounds = [(self.bounds[0], self.bounds[1]) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False}
        )
        
        if not result.success:
            return self._apply_constraints(w0, w_prev)
        
        return self._apply_constraints(result.x, w_prev)

    def _apply_constraints(
        self, w: np.ndarray, w_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply position bounds, sector caps, and turnover constraints.

        Parameters
        ----------
        w : np.ndarray
            Raw optimization weights
        w_prev : np.ndarray, optional
            Previous weights for turnover calculation

        Returns
        -------
        np.ndarray
            Constrained weights that sum to 1
        """
        # 1. Apply bounds
        w = np.clip(w, self.bounds[0], self.bounds[1])

        # 2. Apply sector caps (requires asset-to-sector mapping)
        if self.sector_caps:
            # Note: Sector constraint implementation requires external sector mapping
            # This is a constraint validation rather than full sector optimization
            max_individual_weight = min(self.bounds[1], min(self.sector_caps.values()))
            w = np.clip(w, self.bounds[0], max_individual_weight)

        # 3. Apply turnover budget constraint (post-optimization check)
        if w_prev is not None and self.turnover_budget is not None:
            turnover = np.sum(np.abs(w - w_prev))
            if turnover > self.turnover_budget:
                # Scale back the changes proportionally
                w_diff = w - w_prev
                scale_factor = self.turnover_budget / turnover
                w = w_prev + w_diff * scale_factor

        # 4. Normalize to sum to 1
        total_weight = np.sum(w)
        if total_weight > 0:
            w = w / total_weight
        else:
            # Fallback to equal weights
            w = np.ones(len(w)) / len(w)

        return w

    def _calculate_portfolio_stats(
        self, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate portfolio statistics.

        Parameters
        ----------
        w : np.ndarray
            Portfolio weights
        mu : np.ndarray
            Expected returns
        sigma : np.ndarray
            Covariance matrix

        Returns
        -------
        dict
            Portfolio statistics (return, volatility, sharpe)
        """
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = (port_return - self.rf) / port_vol if port_vol > 0 else 0.0

        return {"return": port_return, "volatility": port_vol, "sharpe": sharpe}
