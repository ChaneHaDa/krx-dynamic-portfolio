"""Advanced risk analytics for portfolio backtesting."""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA


class RiskAnalytics:
    """
    Advanced risk analytics for portfolio backtesting and risk management.
    
    This class provides sophisticated risk metrics and analysis tools including:
    - Value at Risk (VaR) and Conditional VaR calculations
    - Extreme value theory applications
    - Risk factor decomposition
    - Stress testing and scenario analysis
    - Correlation and tail dependency analysis
    """

    def __init__(
        self,
        confidence_levels: list[float] = None,
        var_methods: list[str] = None,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize risk analytics engine.

        Parameters
        ----------
        confidence_levels : list of float, optional
            VaR confidence levels (default: [0.01, 0.05, 0.10])
        var_methods : list of str, optional
            VaR calculation methods (default: ['historical', 'parametric', 'cornish_fisher'])
        risk_free_rate : float
            Risk-free rate for calculations
        """
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]
        self.var_methods = var_methods or ['historical', 'parametric', 'cornish_fisher']
        self.risk_free_rate = risk_free_rate

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.05,
        method: str = 'historical',
        window: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Calculate Value at Risk using multiple methods.

        Parameters
        ----------
        returns : pd.Series
            Return series
        confidence_level : float
            VaR confidence level (e.g., 0.05 for 95% VaR)
        method : str
            Calculation method ('historical', 'parametric', 'cornish_fisher')
        window : int, optional
            Rolling window for calculation (uses all data if None)

        Returns
        -------
        dict
            VaR estimates by method
        """
        if window is not None:
            returns = returns.iloc[-window:]

        var_results = {}

        if method in ['historical', 'all']:
            var_results['historical'] = self._historical_var(returns, confidence_level)

        if method in ['parametric', 'all']:
            var_results['parametric'] = self._parametric_var(returns, confidence_level)

        if method in ['cornish_fisher', 'all']:
            var_results['cornish_fisher'] = self._cornish_fisher_var(returns, confidence_level)

        if method == 'all':
            return var_results
        else:
            return {method: var_results[method]}

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.05,
        method: str = 'historical',
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Parameters
        ----------
        returns : pd.Series
            Return series
        confidence_level : float
            CVaR confidence level
        method : str
            Calculation method

        Returns
        -------
        float
            CVaR estimate
        """
        var_estimate = self.calculate_var(returns, confidence_level, method)
        
        if method in var_estimate:
            var_value = var_estimate[method]
        else:
            var_value = list(var_estimate.values())[0]

        # Expected shortfall: mean of returns below VaR
        tail_losses = returns[returns <= var_value]
        
        if len(tail_losses) > 0:
            return tail_losses.mean()
        else:
            # If no observations below VaR, return VaR itself
            return var_value

    def extreme_value_analysis(
        self,
        returns: pd.Series,
        method: str = 'peaks_over_threshold',
        threshold_percentile: float = 0.95,
    ) -> dict[str, float]:
        """
        Perform extreme value analysis for tail risk estimation.

        Parameters
        ----------
        returns : pd.Series
            Return series
        method : str
            EVA method ('peaks_over_threshold', 'block_maxima')
        threshold_percentile : float
            Threshold percentile for POT method

        Returns
        -------
        dict
            Extreme value analysis results
        """
        if method == 'peaks_over_threshold':
            return self._peaks_over_threshold(returns, threshold_percentile)
        elif method == 'block_maxima':
            return self._block_maxima(returns)
        else:
            raise ValueError(f"Unknown EVA method: {method}")

    def risk_factor_decomposition(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        method: str = 'regression',
    ) -> dict[str, Union[float, pd.Series]]:
        """
        Decompose portfolio risk into factor components.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Portfolio return series
        factor_returns : pd.DataFrame
            Factor return data
        method : str
            Decomposition method ('regression', 'pca')

        Returns
        -------
        dict
            Risk decomposition results
        """
        if method == 'regression':
            return self._regression_risk_decomposition(portfolio_returns, factor_returns)
        elif method == 'pca':
            return self._pca_risk_decomposition(portfolio_returns, factor_returns)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

    def stress_testing(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        scenarios: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """
        Perform stress testing under various market scenarios.

        Parameters
        ----------
        portfolio_weights : pd.Series
            Current portfolio weights
        asset_returns : pd.DataFrame
            Historical asset returns
        scenarios : dict
            Stress scenarios {scenario_name: {asset: shock_pct}}

        Returns
        -------
        pd.DataFrame
            Stress test results
        """
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            scenario_returns = {}
            
            for asset in portfolio_weights.index:
                if asset in asset_returns.columns:
                    base_return = asset_returns[asset].mean()
                    shock = shocks.get(asset, 0.0)
                    scenario_returns[asset] = base_return + shock
                else:
                    scenario_returns[asset] = 0.0
            
            # Calculate portfolio impact
            scenario_return = sum(
                portfolio_weights[asset] * scenario_returns[asset]
                for asset in portfolio_weights.index
                if asset in scenario_returns
            )
            
            results[scenario_name] = scenario_return

        return pd.Series(results, name='portfolio_impact')

    def correlation_analysis(
        self,
        returns: pd.DataFrame,
        method: str = 'pearson',
        rolling_window: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze correlation structure of returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        rolling_window : int, optional
            Rolling window for time-varying correlations

        Returns
        -------
        dict
            Correlation analysis results
        """
        results = {}
        
        # Static correlation
        results['correlation_matrix'] = returns.corr(method=method)
        
        # Average correlation
        corr_matrix = results['correlation_matrix']
        # Remove diagonal and get upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = corr_matrix.values[mask].mean()
        results['average_correlation'] = avg_correlation
        
        # Rolling correlation if requested
        if rolling_window is not None and rolling_window < len(returns):
            rolling_corrs = {}
            for i in range(rolling_window, len(returns)):
                window_data = returns.iloc[i-rolling_window:i]
                window_corr = window_data.corr(method=method)
                date = returns.index[i]
                rolling_corrs[date] = window_corr.values[mask].mean()
            
            results['rolling_avg_correlation'] = pd.Series(rolling_corrs)
        
        # Eigenvalue decomposition for factor structure
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix.values)
        results['eigenvalues'] = pd.Series(eigenvals[::-1], name='eigenvalues')  # Descending order
        
        # Principal component variance explained
        total_variance = eigenvals.sum()
        results['pca_variance_explained'] = eigenvals[::-1] / total_variance

        return results

    def tail_dependency_analysis(
        self,
        returns: pd.DataFrame,
        threshold_percentile: float = 0.95,
    ) -> pd.DataFrame:
        """
        Analyze tail dependencies between assets.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns
        threshold_percentile : float
            Threshold for tail analysis

        Returns
        -------
        pd.DataFrame
            Tail dependency coefficients
        """
        n_assets = len(returns.columns)
        tail_deps = np.zeros((n_assets, n_assets))
        
        # Calculate thresholds
        thresholds = returns.quantile([1 - threshold_percentile, threshold_percentile])
        
        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                if i != j:
                    # Upper tail dependency
                    upper_threshold_i = thresholds.loc[1 - threshold_percentile, asset_i]
                    upper_threshold_j = thresholds.loc[1 - threshold_percentile, asset_j]
                    
                    # Count joint exceedances
                    joint_upper = (
                        (returns[asset_i] > upper_threshold_i) & 
                        (returns[asset_j] > upper_threshold_j)
                    ).sum()
                    
                    marginal_upper_i = (returns[asset_i] > upper_threshold_i).sum()
                    
                    if marginal_upper_i > 0:
                        tail_deps[i, j] = joint_upper / marginal_upper_i
                    else:
                        tail_deps[i, j] = 0
        
        return pd.DataFrame(
            tail_deps, 
            index=returns.columns, 
            columns=returns.columns
        )

    def portfolio_var_decomposition(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        confidence_level: float = 0.05,
    ) -> pd.Series:
        """
        Decompose portfolio VaR into individual asset contributions.

        Parameters
        ----------
        portfolio_weights : pd.Series
            Portfolio weights
        asset_returns : pd.DataFrame
            Asset returns
        confidence_level : float
            VaR confidence level

        Returns
        -------
        pd.Series
            VaR contributions by asset
        """
        # Align weights and returns
        common_assets = portfolio_weights.index.intersection(asset_returns.columns)
        weights_aligned = portfolio_weights[common_assets]
        returns_aligned = asset_returns[common_assets]
        
        # Calculate portfolio returns
        portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
        
        # Portfolio VaR
        portfolio_var = self._historical_var(portfolio_returns, confidence_level)
        
        # Component VaR using marginal VaR approach
        var_contributions = {}
        
        for asset in common_assets:
            # Calculate marginal VaR by perturbing weight slightly
            epsilon = 0.0001
            perturbed_weights = weights_aligned.copy()
            perturbed_weights[asset] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
            
            perturbed_returns = (returns_aligned * perturbed_weights).sum(axis=1)
            perturbed_var = self._historical_var(perturbed_returns, confidence_level)
            
            marginal_var = (perturbed_var - portfolio_var) / epsilon
            component_var = weights_aligned[asset] * marginal_var
            
            var_contributions[asset] = component_var
        
        return pd.Series(var_contributions)

    # Private helper methods

    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return np.percentile(returns.dropna(), confidence_level * 100)

    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric (normal) VaR."""
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(confidence_level)
        return mean_return + z_score * std_return

    def _cornish_fisher_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)."""
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = stats.skew(returns, nan_policy='omit')
        kurtosis = stats.kurtosis(returns, nan_policy='omit')  # Excess kurtosis
        
        z_score = stats.norm.ppf(confidence_level)
        
        # Cornish-Fisher adjustment
        adjusted_z = (
            z_score +
            (z_score**2 - 1) * skewness / 6 +
            (z_score**3 - 3*z_score) * kurtosis / 24 -
            (2*z_score**3 - 5*z_score) * skewness**2 / 36
        )
        
        return mean_return + adjusted_z * std_return

    def _peaks_over_threshold(
        self, 
        returns: pd.Series, 
        threshold_percentile: float
    ) -> dict[str, float]:
        """Peaks over threshold extreme value analysis."""
        threshold = np.percentile(returns, (1 - threshold_percentile) * 100)
        exceedances = returns[returns < threshold] - threshold
        
        if len(exceedances) == 0:
            return {"threshold": threshold, "scale": np.nan, "shape": np.nan}
        
        # Fit Generalized Pareto Distribution
        try:
            # Simple method-of-moments estimation
            mean_excess = exceedances.mean()
            var_excess = exceedances.var()
            
            # Method of moments estimates
            shape = -0.5 * ((mean_excess**2) / var_excess - 1)
            scale = 0.5 * mean_excess * ((mean_excess**2) / var_excess + 1)
            
            return {
                "threshold": threshold,
                "scale": scale,
                "shape": shape,
                "n_exceedances": len(exceedances),
            }
        except:
            return {"threshold": threshold, "scale": np.nan, "shape": np.nan}

    def _block_maxima(self, returns: pd.Series, block_size: int = 22) -> dict[str, float]:
        """Block maxima extreme value analysis."""
        # Split returns into blocks and find minima (for losses)
        blocks = []
        for i in range(0, len(returns), block_size):
            block = returns.iloc[i:i+block_size]
            if len(block) >= block_size // 2:  # Only use blocks with sufficient data
                blocks.append(block.min())
        
        if len(blocks) < 3:
            return {"location": np.nan, "scale": np.nan, "shape": np.nan}
        
        block_minima = pd.Series(blocks)
        
        # Fit Generalized Extreme Value distribution (simple moments method)
        try:
            mean_min = block_minima.mean()
            std_min = block_minima.std()
            skew_min = stats.skew(block_minima)
            
            # Rough moment estimates for GEV
            shape = -skew_min * 0.1  # Simplified estimate
            scale = std_min * (1 + shape)**(-1) if shape > -1 else std_min
            location = mean_min - scale * (1 - 1.0**(1+shape)) / shape if shape != 0 else mean_min
            
            return {
                "location": location,
                "scale": scale,
                "shape": shape,
                "n_blocks": len(blocks),
            }
        except:
            return {"location": np.nan, "scale": np.nan, "shape": np.nan}

    def _regression_risk_decomposition(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> dict[str, Union[float, pd.Series]]:
        """Risk decomposition using factor regression."""
        # Align data
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        fact_ret = factor_returns.loc[common_dates]
        
        if len(common_dates) < max(10, len(factor_returns.columns) * 2):
            return {"error": "Insufficient overlapping data"}
        
        # Multiple regression
        X = fact_ret.values
        y = port_ret.values
        
        try:
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # OLS estimation
            beta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
            
            # Predictions and residuals
            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            
            # Risk decomposition
            factor_exposures = pd.Series(beta[1:], index=factor_returns.columns)
            factor_var_contrib = factor_exposures**2 * fact_ret.var()
            specific_risk = np.var(residuals)
            
            total_risk = port_ret.var()
            systematic_risk = factor_var_contrib.sum()
            
            return {
                "factor_exposures": factor_exposures,
                "factor_risk_contributions": factor_var_contrib,
                "systematic_risk": systematic_risk,
                "specific_risk": specific_risk,
                "total_risk": total_risk,
                "r_squared": 1 - specific_risk / total_risk,
            }
        except np.linalg.LinAlgError:
            return {"error": "Singular matrix in regression"}

    def _pca_risk_decomposition(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> dict[str, Union[float, pd.Series]]:
        """Risk decomposition using PCA."""
        # Combine portfolio and factor returns
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        combined_data = pd.concat([
            portfolio_returns.loc[common_dates].to_frame('portfolio'),
            factor_returns.loc[common_dates]
        ], axis=1)
        
        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(combined_data.fillna(0))
        
        # Extract portfolio loadings on principal components
        portfolio_loadings = pca.components_[:, 0]  # First column is portfolio
        
        # Risk contribution by principal component
        pc_variances = pca.explained_variance_
        risk_contributions = portfolio_loadings**2 * pc_variances
        
        return {
            "principal_component_loadings": pd.Series(
                portfolio_loadings, 
                index=[f"PC{i+1}" for i in range(len(portfolio_loadings))]
            ),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "risk_contributions": pd.Series(
                risk_contributions,
                index=[f"PC{i+1}" for i in range(len(risk_contributions))]
            ),
        }

    def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.05,
        n_simulations: int = 10000,
        distribution: str = 'normal',
    ) -> float:
        """
        Calculate VaR using Monte Carlo simulation.

        Parameters
        ----------
        returns : pd.Series
            Historical returns
        confidence_level : float
            VaR confidence level
        n_simulations : int
            Number of Monte Carlo simulations
        distribution : str
            Distribution for simulation ('normal', 'empirical')

        Returns
        -------
        float
            Monte Carlo VaR estimate
        """
        if distribution == 'normal':
            mean_return = returns.mean()
            std_return = returns.std()
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        elif distribution == 'empirical':
            # Bootstrap from empirical distribution
            simulated_returns = np.random.choice(returns.values, n_simulations, replace=True)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return np.percentile(simulated_returns, confidence_level * 100)