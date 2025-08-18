"""Integrated portfolio optimization pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .mpt import MPTOptimizer
from .rebalance import Rebalancer
from .risk import RiskModel

logger = logging.getLogger(__name__)


class PortfolioOptimizationPipeline:
    """End-to-end portfolio optimization pipeline."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize optimization pipeline.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to YAML configuration file
        config : dict, optional
            Configuration dictionary (alternative to config_path)
        """
        if config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config is not None:
            self.config = config.copy()
        else:
            self.config = self._get_default_config()

        # Initialize components
        self.risk_model = self._create_risk_model()
        self.optimizer = self._create_optimizer()
        self.rebalancer = self._create_rebalancer()

        # Cache for fitted models
        self._fitted_risk_model = None
        self._last_weights = None
        self._rebalance_dates = None

    def build_weights(
        self,
        mu: np.ndarray,
        returns: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
        prices: Optional[pd.Series] = None,
        current_weights: Optional[np.ndarray] = None,
        date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Build portfolio weights through complete optimization pipeline.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector
        returns : pd.DataFrame
            Historical return data for risk model
        sector_map : dict, optional
            Asset to sector mapping
        prices : pd.Series, optional
            Current asset prices for rebalancing
        current_weights : np.ndarray, optional
            Current portfolio weights
        date : pd.Timestamp, optional
            Current date for rebalancing logic

        Returns
        -------
        dict
            Complete optimization results
        """
        results = {}

        # Step 1: Fit risk model
        logger.info("Fitting risk model...")
        self.risk_model.fit(returns)
        covariance_matrix = self.risk_model.cov()

        # Step 2: Portfolio optimization
        logger.info("Running portfolio optimization...")
        objective = self.config["objective"]

        if objective == "max_sharpe":
            target_weights = self.optimizer.max_sharpe(
                mu, covariance_matrix, current_weights
            )
        elif objective == "min_variance":
            target_weights = self.optimizer.min_variance(
                mu, covariance_matrix, current_weights
            )
        elif objective == "mean_variance":
            risk_aversion = self.config.get("risk_aversion")
            target_return = self.config.get("target_return")
            target_weights = self.optimizer.mean_variance(
                mu, covariance_matrix, risk_aversion, target_return, current_weights
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        results["target_weights"] = target_weights
        results["risk_metrics"] = self._calculate_risk_metrics(
            target_weights, mu, covariance_matrix
        )

        # Step 3: Apply rebalancing logic
        if current_weights is not None:
            logger.info("Applying rebalancing logic...")

            # Check if rebalancing is needed
            if self._rebalance_dates is None and date is not None:
                date_range = pd.date_range(
                    start=date, end=date + pd.DateOffset(years=1), freq="D"
                )
                self._rebalance_dates = self.rebalancer.next_rebalance_dates(
                    pd.DatetimeIndex(date_range)
                )

            should_rebalance = date is None or self.rebalancer.should_rebalance(
                target_weights, current_weights, date, self._rebalance_dates or []
            )

            if should_rebalance and prices is not None:
                rebalance_results = self.rebalancer.apply(
                    target_weights,
                    current_weights,
                    prices,
                    asset_names=returns.columns.tolist(),
                )
                results.update(rebalance_results)
                results["rebalanced"] = True
            else:
                results["w_executed"] = current_weights
                results["rebalanced"] = False
        else:
            results["w_executed"] = target_weights
            results["rebalanced"] = False

        # Step 4: Add sector analysis if available
        if sector_map is not None:
            results["sector_analysis"] = self._analyze_sector_weights(
                results["w_executed"], sector_map, returns.columns
            )

        # Step 5: Log summary
        self._log_optimization_summary(results)

        return results

    def generate_weight_series(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.DataFrame,
        rebalance_dates: Optional[List[pd.Timestamp]] = None,
        initial_weights: Optional[np.ndarray] = None,
        prices: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate time series of portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (dates x assets)
        expected_returns : pd.DataFrame
            Expected returns for each date
        rebalance_dates : list, optional
            Specific rebalancing dates
        initial_weights : np.ndarray, optional
            Initial portfolio weights
        prices : pd.DataFrame, optional
            Price data for rebalancing

        Returns
        -------
        pd.DataFrame
            Time series of portfolio weights
        """
        if rebalance_dates is None:
            rebalance_dates = self.rebalancer.next_rebalance_dates(returns.index)

        weight_series = []
        current_weights = initial_weights

        for date in returns.index:
            if date in rebalance_dates or current_weights is None:
                # Get expected returns for this date
                if date in expected_returns.index:
                    mu = expected_returns.loc[date].values

                    # Get historical window for risk model
                    lookback_window = self.config.get("lookback_window", 252)
                    start_idx = max(0, returns.index.get_loc(date) - lookback_window)
                    hist_returns = returns.iloc[
                        start_idx : returns.index.get_loc(date) + 1
                    ]

                    if len(hist_returns) >= 30:  # Minimum periods for risk model
                        current_prices = (
                            prices.loc[date] if prices is not None else None
                        )

                        results = self.build_weights(
                            mu=mu,
                            returns=hist_returns,
                            prices=current_prices,
                            current_weights=current_weights,
                            date=date,
                        )

                        current_weights = results["w_executed"]

            # Record weights for this date
            if current_weights is not None:
                weight_series.append(
                    pd.Series(current_weights, index=returns.columns, name=date)
                )

        return pd.DataFrame(weight_series).fillna(0)

    def save_results(
        self, results: Dict[str, Any], output_path: Union[str, Path]
    ) -> None:
        """Save optimization results to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save weights
        if "w_executed" in results:
            weights_df = pd.DataFrame(
                {"weight": results["w_executed"]},
                index=range(len(results["w_executed"])),
            )
            weights_df.to_parquet(output_path / "weights.parquet")

        # Save orders if available
        if "orders" in results and not results["orders"].empty:
            results["orders"].to_parquet(output_path / "orders.parquet")

        # Save summary metrics
        summary = {
            "optimization_date": datetime.now().isoformat(),
            "objective": self.config["objective"],
            "rebalanced": results.get("rebalanced", False),
            "turnover": results.get("turnover", 0.0),
            "transaction_cost": results.get("tc_cost", 0.0),
        }

        if "risk_metrics" in results:
            summary.update(results["risk_metrics"])

        with open(output_path / "summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _create_risk_model(self) -> RiskModel:
        """Create risk model from configuration."""
        risk_config = self.config.get("risk_model", {})
        return RiskModel(
            method=risk_config.get("method", "ledoit_wolf"),
            ewma_lambda=risk_config.get("ewma_lambda", 0.94),
            ridge=risk_config.get("ridge", 1e-6),
            min_periods=risk_config.get("min_periods", 252),
        )

    def _create_optimizer(self) -> MPTOptimizer:
        """Create optimizer from configuration."""
        constraints = self.config.get("constraints", {})
        return MPTOptimizer(
            bounds=tuple(constraints.get("w_bounds", [0.0, 0.1])),
            rf=self.config.get("risk_free_rate", 0.0),
            sector_caps=constraints.get("sector_caps"),
            turnover_budget=constraints.get("turnover_budget"),
            penalty=constraints.get("turnover_penalty"),
        )

    def _create_rebalancer(self) -> Rebalancer:
        """Create rebalancer from configuration."""
        rebalance_config = self.config.get("rebalance", {})
        return Rebalancer(
            schedule=rebalance_config.get("schedule", "month_end"),
            turnover_budget=rebalance_config.get("turnover_budget", 0.25),
            rebalance_threshold=rebalance_config.get("threshold", 0.05),
            tc_bps=rebalance_config.get("tc_bps", 25.0),
        )

    def _calculate_risk_metrics(
        self, weights: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (
            portfolio_return - self.config.get("risk_free_rate", 0.0)
        ) / portfolio_vol

        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
            "effective_positions": int(np.sum(weights > 0.001)),
        }

    def _analyze_sector_weights(
        self, weights: np.ndarray, sector_map: Dict[str, str], asset_names: pd.Index
    ) -> Dict[str, float]:
        """Analyze sector weight distribution."""
        sector_weights = {}
        for asset, weight in zip(asset_names, weights):
            sector = sector_map.get(asset, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        return sector_weights

    def _log_optimization_summary(self, results: Dict[str, Any]) -> None:
        """Log optimization summary."""
        logger.info("Portfolio optimization completed:")
        logger.info(f"  Objective: {self.config['objective']}")
        logger.info(
            f"  Expected return: {results.get('risk_metrics', {}).get('expected_return', 'N/A'):.4f}"
        )
        logger.info(
            f"  Volatility: {results.get('risk_metrics', {}).get('volatility', 'N/A'):.4f}"
        )
        logger.info(
            f"  Sharpe ratio: {results.get('risk_metrics', {}).get('sharpe_ratio', 'N/A'):.4f}"
        )
        logger.info(f"  Rebalanced: {results.get('rebalanced', False)}")
        if results.get("rebalanced"):
            logger.info(f"  Turnover: {results.get('turnover', 0.0):.4f}")
            logger.info(f"  Transaction cost: {results.get('tc_cost', 0.0):.4f}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "objective": "max_sharpe",
            "risk_free_rate": 0.0,
            "risk_model": {"method": "ledoit_wolf", "ewma_lambda": 0.94, "ridge": 1e-6},
            "constraints": {
                "long_only": True,
                "w_bounds": [0.0, 0.1],
                "sector_caps": {},
                "min_positions": 10,
            },
            "rebalance": {
                "schedule": "month_end",
                "threshold": 0.05,
                "turnover_budget": 0.25,
                "tc_bps": 25.0,
            },
            "vol_target": {"enable": False, "annual_vol": 0.10},
            "lookback_window": 252,
        }


def build_weights(
    mu: np.ndarray,
    returns: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for single-shot portfolio optimization.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector
    returns : pd.DataFrame
        Historical return data
    sector_map : dict, optional
        Asset to sector mapping
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    dict
        Optimization results with weights, risk metrics, and execution details
    """
    pipeline = PortfolioOptimizationPipeline(config=config)
    return pipeline.build_weights(mu, returns, sector_map)


def create_monthly_weights(
    returns_data: pd.DataFrame,
    expected_returns: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Create monthly portfolio weights and save results.

    Parameters
    ----------
    returns_data : pd.DataFrame
        Historical returns data
    expected_returns : pd.DataFrame
        Expected returns by date
    config : dict, optional
        Configuration dictionary
    output_dir : str or Path, optional
        Output directory for saving results

    Returns
    -------
    pd.DataFrame
        Monthly portfolio weights
    """
    pipeline = PortfolioOptimizationPipeline(config=config)

    # Generate monthly rebalance dates
    monthly_dates = pipeline.rebalancer.next_rebalance_dates(returns_data.index)

    # Generate weight series
    weight_series = pipeline.generate_weight_series(
        returns_data, expected_returns, monthly_dates
    )

    # Save results if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save monthly weights
        for date in weight_series.index:
            month_str = date.strftime("%Y%m")
            weight_series.loc[[date]].to_parquet(
                output_dir / f"weights_{month_str}.parquet"
            )

    return weight_series
