"""Performance metrics calculation for portfolio backtesting."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculation for portfolio backtesting.
    
    This class provides a wide range of performance and risk metrics commonly
    used in quantitative finance and portfolio management.
    """

    def __init__(self, risk_free_rate: float = 0.0, confidence_level: float = 0.05):
        """
        Initialize performance metrics calculator.

        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default: 0.0)
        confidence_level : float
            Confidence level for VaR calculations (default: 0.05 for 95% VaR)
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        portfolio_values: Optional[pd.Series] = None,
    ) -> dict[str, float]:
        """
        Calculate comprehensive set of performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Portfolio returns (daily)
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        portfolio_values : pd.Series, optional
            Portfolio values over time (for drawdown calculation)

        Returns
        -------
        dict
            Dictionary containing all calculated metrics
        """
        metrics = {}

        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown metrics
        if portfolio_values is not None:
            metrics.update(self._calculate_drawdown_metrics(portfolio_values))
        elif returns is not None:
            # Calculate from cumulative returns if portfolio values not available
            cum_returns = (1 + returns).cumprod()
            metrics.update(self._calculate_drawdown_metrics(cum_returns))
        
        # Benchmark comparison metrics
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # Higher moment metrics
        metrics.update(self._calculate_moment_metrics(returns))
        
        # Tail risk metrics
        metrics.update(self._calculate_tail_risk_metrics(returns))

        return metrics

    def _calculate_return_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate basic return metrics."""
        total_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        periods_per_year = self._infer_frequency(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        
        # Geometric mean
        geometric_mean = (1 + returns).prod() ** (1 / len(returns)) - 1
        arithmetic_mean = returns.mean()
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "arithmetic_mean": arithmetic_mean,
            "geometric_mean": geometric_mean,
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "positive_days": (returns > 0).sum() / len(returns),
            "negative_days": (returns < 0).sum() / len(returns),
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate risk metrics."""
        periods_per_year = self._infer_frequency(returns)
        
        volatility_daily = returns.std()
        volatility_annualized = volatility_daily * np.sqrt(periods_per_year)
        
        # Downside volatility
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(periods_per_year)
        
        # Semi-volatility (below mean)
        mean_return = returns.mean()
        below_mean_returns = returns[returns < mean_return]
        semi_volatility = below_mean_returns.std() * np.sqrt(periods_per_year)

        return {
            "volatility": volatility_annualized,
            "downside_volatility": downside_volatility,
            "semi_volatility": semi_volatility,
            "volatility_daily": volatility_daily,
        }

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        periods_per_year = self._infer_frequency(returns)
        annual_rf_rate = self.risk_free_rate
        daily_rf_rate = annual_rf_rate / periods_per_year
        
        mean_return = returns.mean()
        volatility = returns.std()
        annualized_return = mean_return * periods_per_year
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - annual_rf_rate) / annualized_volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (using downside volatility)
        negative_returns = returns[returns < daily_rf_rate]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = (annualized_return - annual_rf_rate) / downside_deviation if downside_deviation > 0 else 0.0
        else:
            sortino_ratio = np.inf if annualized_return > annual_rf_rate else 0.0
        
        # Calmar ratio (calculated later with max drawdown)
        
        # Information ratio (requires benchmark)
        # Will be calculated in benchmark metrics if benchmark is provided
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
        }

    def _calculate_drawdown_metrics(self, values: pd.Series) -> dict[str, float]:
        """Calculate drawdown metrics from portfolio values or cumulative returns."""
        # Calculate running maximum
        running_max = values.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Average drawdown
        negative_dd = drawdowns[drawdowns < 0]
        avg_drawdown = negative_dd.mean() if len(negative_dd) > 0 else 0.0
        
        # Drawdown duration analysis
        drawdown_periods = []
        current_dd_length = 0
        in_drawdown = False
        
        for dd in drawdowns:
            if dd < -0.001:  # In drawdown (threshold to avoid noise)
                if not in_drawdown:
                    in_drawdown = True
                    current_dd_length = 1
                else:
                    current_dd_length += 1
            else:
                if in_drawdown:
                    drawdown_periods.append(current_dd_length)
                    in_drawdown = False
                    current_dd_length = 0
        
        # Add final drawdown if still in one
        if in_drawdown:
            drawdown_periods.append(current_dd_length)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio
        periods_per_year = self._infer_frequency(values)
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / len(values)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown_duration": avg_drawdown_duration,
            "calmar_ratio": calmar_ratio,
            "recovery_factor": total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0,
        }

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict[str, float]:
        """Calculate benchmark comparison metrics."""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {"information_ratio": 0.0, "tracking_error": 0.0, "beta": 0.0, "alpha": 0.0}
        
        # Excess returns
        excess_returns = aligned_returns - aligned_benchmark
        
        # Tracking error
        periods_per_year = self._infer_frequency(aligned_returns)
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        
        # Information ratio
        mean_excess_return = excess_returns.mean() * periods_per_year
        information_ratio = mean_excess_return / tracking_error if tracking_error > 0 else 0.0
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Alpha calculation (Jensen's Alpha)
        daily_rf_rate = self.risk_free_rate / periods_per_year
        portfolio_mean = aligned_returns.mean()
        benchmark_mean = aligned_benchmark.mean()
        alpha = (portfolio_mean - daily_rf_rate) - beta * (benchmark_mean - daily_rf_rate)
        alpha_annualized = alpha * periods_per_year
        
        # Up/Down capture ratios
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0
        
        up_capture = (
            aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean()
            if up_periods.sum() > 0 and aligned_benchmark[up_periods].mean() != 0 
            else 0.0
        )
        
        down_capture = (
            aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean()
            if down_periods.sum() > 0 and aligned_benchmark[down_periods].mean() != 0 
            else 0.0
        )

        return {
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "beta": beta,
            "alpha": alpha_annualized,
            "up_capture_ratio": up_capture,
            "down_capture_ratio": down_capture,
        }

    def _calculate_moment_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate higher moment metrics (skewness, kurtosis)."""
        skewness = stats.skew(returns, nan_policy='omit')
        kurtosis = stats.kurtosis(returns, nan_policy='omit')  # Excess kurtosis
        
        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "jarque_bera_stat": self._jarque_bera_test(returns),
        }

    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate tail risk metrics (VaR, CVaR, etc.)."""
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, self.confidence_level * 100)
        var_99 = np.percentile(returns, 1.0)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum loss (worst single-day return)
        max_loss = returns.min()
        
        # Gain-to-Pain ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        gain_to_pain = (
            positive_returns.sum() / abs(negative_returns.sum())
            if len(negative_returns) > 0 and negative_returns.sum() != 0
            else 0.0
        )

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "max_loss": max_loss,
            "gain_to_pain_ratio": gain_to_pain,
        }

    def _jarque_bera_test(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera test statistic for normality."""
        try:
            statistic, _ = stats.jarque_bera(returns.dropna())
            return statistic
        except:
            return np.nan

    def _infer_frequency(self, returns: pd.Series) -> int:
        """Infer the frequency of returns data and return periods per year."""
        if hasattr(returns.index, 'freq') and returns.index.freq is not None:
            # Use pandas frequency if available
            freq = returns.index.freq
            if hasattr(freq, 'n'):
                if 'D' in str(freq):
                    return 252
                elif 'W' in str(freq):
                    return 52
                elif 'M' in str(freq):
                    return 12
                elif 'Q' in str(freq):
                    return 4
        
        # Fallback: infer from data length and date range
        if hasattr(returns.index, 'to_pydatetime'):
            try:
                total_days = (returns.index[-1] - returns.index[0]).days
                avg_days_per_period = total_days / len(returns)
                
                if avg_days_per_period <= 1.5:
                    return 252  # Daily
                elif avg_days_per_period <= 8:
                    return 52   # Weekly
                elif avg_days_per_period <= 32:
                    return 12   # Monthly
                elif avg_days_per_period <= 95:
                    return 4    # Quarterly
                else:
                    return 1    # Annual
            except:
                pass
        
        # Default assumption: daily data
        return 252

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: list[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int
            Rolling window size (default: 252 for 1-year rolling)
        metrics : list, optional
            List of metrics to calculate (default: common metrics)

        Returns
        -------
        pd.DataFrame
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'volatility', 'max_drawdown', 'var_95']
        
        rolling_results = {}
        
        for metric in metrics:
            if metric == 'sharpe_ratio':
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                periods_per_year = self._infer_frequency(returns)
                rolling_results[metric] = (
                    (rolling_mean * periods_per_year - self.risk_free_rate) /
                    (rolling_std * np.sqrt(periods_per_year))
                )
            elif metric == 'volatility':
                periods_per_year = self._infer_frequency(returns)
                rolling_results[metric] = returns.rolling(window).std() * np.sqrt(periods_per_year)
            elif metric == 'max_drawdown':
                rolling_results[metric] = returns.rolling(window).apply(
                    lambda x: self._calculate_rolling_max_drawdown(x), raw=False
                )
            elif metric == 'var_95':
                rolling_results[metric] = returns.rolling(window).quantile(self.confidence_level)
        
        return pd.DataFrame(rolling_results, index=returns.index)

    def _calculate_rolling_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate max drawdown for a rolling window."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()

    def performance_attribution(
        self,
        portfolio_returns: pd.Series,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate performance attribution by asset.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Portfolio returns
        weights : pd.DataFrame
            Asset weights over time
        asset_returns : pd.DataFrame
            Individual asset returns

        Returns
        -------
        pd.DataFrame
            Attribution analysis results
        """
        # Align all data
        common_dates = portfolio_returns.index.intersection(
            weights.index.intersection(asset_returns.index)
        )
        
        weights_aligned = weights.loc[common_dates]
        returns_aligned = asset_returns.loc[common_dates]
        portfolio_aligned = portfolio_returns.loc[common_dates]
        
        # Calculate contribution by asset
        contributions = weights_aligned.shift(1) * returns_aligned
        contributions = contributions.fillna(0)
        
        # Summary attribution
        total_contribution = contributions.sum()
        avg_weights = weights_aligned.mean()
        
        attribution_df = pd.DataFrame({
            'avg_weight': avg_weights,
            'total_contribution': total_contribution,
            'contribution_pct': total_contribution / portfolio_aligned.sum() * 100,
        })
        
        return attribution_df.sort_values('total_contribution', ascending=False)