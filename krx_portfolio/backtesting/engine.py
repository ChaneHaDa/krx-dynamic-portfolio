"""Backtesting engine for portfolio performance simulation."""

import logging
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..models.rebalance import Rebalancer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Portfolio backtesting engine with realistic execution modeling.
    
    This engine simulates portfolio performance over time, including:
    - Portfolio value tracking
    - Rebalancing execution with transaction costs
    - Market impact and slippage modeling
    - Performance attribution analysis
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        rebalancer: Optional[Rebalancer] = None,
        transaction_cost_bps: float = 25.0,
        market_impact_model: str = "linear",
        benchmark_returns: Optional[pd.Series] = None,
        cash_rate: float = 0.0,
    ):
        """
        Initialize backtesting engine.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value
        rebalancer : Rebalancer, optional
            Rebalancing strategy (if None, uses default monthly rebalancer)
        transaction_cost_bps : float
            Transaction costs in basis points (round-trip)
        market_impact_model : str
            Market impact model type ('linear', 'sqrt', 'none')
        benchmark_returns : pd.Series, optional
            Benchmark return series for comparison
        cash_rate : float
            Risk-free rate for cash holdings
        """
        self.initial_capital = initial_capital
        self.rebalancer = rebalancer or Rebalancer(
            schedule="month_end", tc_bps=transaction_cost_bps
        )
        self.transaction_cost_bps = transaction_cost_bps / 10000.0  # Convert to decimal
        self.market_impact_model = market_impact_model
        self.benchmark_returns = benchmark_returns
        self.cash_rate = cash_rate / 252.0 if cash_rate > 0 else 0.0  # Daily rate

        # Results storage
        self.results = {}
        self.portfolio_history = []
        self.rebalance_history = []
        self.transaction_history = []

    def run_backtest(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, Any]:
        """
        Run complete backtest simulation.

        Parameters
        ----------
        weights : pd.DataFrame
            Target portfolio weights over time (dates x assets)
        returns : pd.DataFrame
            Asset returns (dates x assets)
        prices : pd.DataFrame, optional
            Asset prices for transaction cost calculation
        start_date : pd.Timestamp, optional
            Backtest start date
        end_date : pd.Timestamp, optional
            Backtest end date

        Returns
        -------
        dict
            Complete backtest results including performance metrics
        """
        logger.info("Starting portfolio backtest simulation...")

        # Align data and date ranges
        aligned_data = self._align_data(weights, returns, prices, start_date, end_date)
        
        if aligned_data is None:
            raise ValueError("No overlapping data found for backtest period")

        weights_aligned, returns_aligned, prices_aligned, dates = aligned_data

        # Initialize portfolio state
        portfolio_state = self._initialize_portfolio(
            weights_aligned.iloc[0], dates[0]
        )

        # Generate rebalancing dates
        rebalance_dates = self.rebalancer.next_rebalance_dates(dates)
        logger.info(f"Generated {len(rebalance_dates)} rebalancing dates")

        # Main simulation loop
        for i, date in enumerate(dates):
            if i == 0:
                # Record initial state
                self._record_portfolio_state(portfolio_state, date)
                continue

            # 1. Apply market returns to current holdings
            portfolio_state = self._apply_market_returns(
                portfolio_state, returns_aligned.iloc[i]
            )

            # 2. Check if rebalancing is needed
            if self._should_rebalance(date, rebalance_dates, i, weights_aligned):
                # Find the most recent weights available
                if i < len(weights_aligned):
                    target_weights = weights_aligned.iloc[i]
                else:
                    # Use forward fill for missing weights
                    target_weights = weights_aligned.iloc[-1]
                    
                current_prices = (
                    prices_aligned.iloc[i] if prices_aligned is not None else None
                )

                portfolio_state = self._execute_rebalancing(
                    portfolio_state, target_weights, current_prices, date
                )

            # 3. Record portfolio state
            self._record_portfolio_state(portfolio_state, date)

        # Compile and return results
        self.results = self._compile_results()
        logger.info("Backtest simulation completed")
        
        return self.results

    def _align_data(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame],
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
    ) -> Optional[tuple]:
        """Align all data sources to common date range and assets."""
        # Find common date range
        common_dates = weights.index.intersection(returns.index)
        
        if start_date is not None:
            common_dates = common_dates[common_dates >= start_date]
        if end_date is not None:
            common_dates = common_dates[common_dates <= end_date]

        if len(common_dates) < 2:
            return None

        # Find common assets
        common_assets = weights.columns.intersection(returns.columns)
        if len(common_assets) == 0:
            return None

        # Align data
        weights_aligned = weights.loc[common_dates, common_assets].fillna(0)
        returns_aligned = returns.loc[common_dates, common_assets].fillna(0)
        
        prices_aligned = None
        if prices is not None:
            if len(prices.index.intersection(common_dates)) > 0:
                prices_aligned = prices.loc[common_dates, common_assets].fillna(method='ffill')

        return weights_aligned, returns_aligned, prices_aligned, common_dates

    def _initialize_portfolio(
        self, initial_weights: pd.Series, start_date: pd.Timestamp
    ) -> dict[str, Any]:
        """Initialize portfolio state."""
        # Normalize weights to sum to 1
        if initial_weights.sum() > 0:
            initial_weights = initial_weights / initial_weights.sum()
        else:
            # Equal weight if no valid weights
            initial_weights = pd.Series(
                1.0 / len(initial_weights), index=initial_weights.index
            )

        portfolio_state = {
            "date": start_date,
            "total_value": self.initial_capital,
            "cash": 0.0,
            "weights": initial_weights.copy(),
            "positions": initial_weights * self.initial_capital,
            "cumulative_return": 0.0,
            "daily_return": 0.0,
            "transaction_costs": 0.0,
            "cumulative_costs": 0.0,
            "turnover": 0.0,
        }

        return portfolio_state

    def _apply_market_returns(
        self, portfolio_state: dict[str, Any], daily_returns: pd.Series
    ) -> dict[str, Any]:
        """Apply daily market returns to portfolio positions."""
        # Get aligned returns for portfolio positions
        positions = portfolio_state["positions"]
        aligned_returns = daily_returns.reindex(positions.index, fill_value=0.0)
        
        # Calculate new position values after market returns
        new_positions = positions * (1 + aligned_returns)
        
        # Apply daily cash rate (convert annual rate to daily)
        daily_cash_rate = self.cash_rate / 252.0
        new_cash = portfolio_state["cash"] * (1 + daily_cash_rate)
        
        # Calculate new total value
        old_value = portfolio_state["total_value"]
        new_total_value = new_positions.sum() + new_cash
        
        # Calculate returns
        daily_return = (new_total_value / old_value) - 1 if old_value > 0 else 0
        cumulative_return = ((1 + portfolio_state["cumulative_return"]) * (1 + daily_return)) - 1
        
        # Update weights
        new_weights = new_positions / new_total_value if new_total_value > 0 else portfolio_state["weights"]
        
        portfolio_state.update({
            "total_value": new_total_value,
            "cash": new_cash,
            "positions": new_positions,
            "weights": new_weights,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
        })

        return portfolio_state

    def _should_rebalance(
        self,
        date: pd.Timestamp,
        rebalance_dates: list[pd.Timestamp],
        index: int,
        weights: pd.DataFrame,
    ) -> bool:
        """Determine if rebalancing should occur."""
        # Check scheduled rebalancing first (no need for weights data)
        for rebal_date in rebalance_dates:
            if date == rebal_date:
                return True
        
        # For other checks, ensure we have valid weights data
        if index >= len(weights):
            return False

        # For threshold-based rebalancing, we need current and target weights
        # This simplified version always returns False for non-scheduled dates
        # Real implementation would compare current vs target weights
        return False

    def _execute_rebalancing(
        self,
        portfolio_state: dict[str, Any],
        target_weights: pd.Series,
        prices: Optional[pd.Series],
        date: pd.Timestamp,
    ) -> dict[str, Any]:
        """Execute portfolio rebalancing with transaction costs."""
        current_weights = portfolio_state["weights"]
        total_value = portfolio_state["total_value"]

        # Normalize target weights
        if target_weights.sum() > 0:
            target_weights = target_weights / target_weights.sum()
        else:
            # No change if invalid target weights
            return portfolio_state

        # Use rebalancer to get optimal execution
        if prices is not None:
            rebalance_result = self.rebalancer.apply(
                w_target=target_weights.values,
                w_current=current_weights.values,
                prices=prices,
                portfolio_value=total_value,
                asset_names=target_weights.index.tolist(),
            )
            
            executed_weights = pd.Series(
                rebalance_result["w_executed"], index=target_weights.index
            )
            transaction_cost = rebalance_result["tc_cost"] * total_value
            turnover = rebalance_result["turnover"]
            
        else:
            # Simple rebalancing without price-based optimization
            executed_weights = target_weights.copy()
            turnover = np.sum(np.abs(target_weights.values - current_weights.values))
            transaction_cost = turnover * self.transaction_cost_bps * total_value

        # Apply market impact
        market_impact_cost = self._calculate_market_impact(
            current_weights, executed_weights, total_value
        )
        
        total_transaction_cost = transaction_cost + market_impact_cost

        # Update portfolio state
        new_total_value = total_value - total_transaction_cost
        new_positions = executed_weights * new_total_value
        
        portfolio_state.update({
            "total_value": new_total_value,
            "positions": new_positions,
            "weights": executed_weights,
            "cash": 0.0,  # Assume fully invested after rebalancing
            "transaction_costs": total_transaction_cost,
            "cumulative_costs": portfolio_state["cumulative_costs"] + total_transaction_cost,
            "turnover": turnover,
        })

        # Record rebalancing event
        self.rebalance_history.append({
            "date": date,
            "target_weights": target_weights,
            "executed_weights": executed_weights,
            "transaction_cost": total_transaction_cost,
            "turnover": turnover,
            "portfolio_value_before": total_value,
            "portfolio_value_after": new_total_value,
        })

        return portfolio_state

    def _calculate_market_impact(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
    ) -> float:
        """Calculate market impact costs based on trade size."""
        if self.market_impact_model == "none":
            return 0.0

        weight_changes = np.abs(target_weights.values - current_weights.values)
        trade_sizes = weight_changes * portfolio_value

        if self.market_impact_model == "linear":
            # Linear impact: cost proportional to trade size
            impact_rate = 0.0005  # 5 bps per $1M traded
            impact_cost = np.sum(trade_sizes * impact_rate / 1_000_000)
        elif self.market_impact_model == "sqrt":
            # Square root impact: cost proportional to sqrt of trade size
            impact_rate = 0.001  # Base rate
            impact_cost = np.sum(np.sqrt(trade_sizes / 1_000_000) * impact_rate * trade_sizes)
        else:
            impact_cost = 0.0

        return impact_cost

    def _record_portfolio_state(
        self, portfolio_state: dict[str, Any], date: pd.Timestamp
    ) -> None:
        """Record current portfolio state for analysis."""
        state_record = {
            "date": date,
            "total_value": portfolio_state["total_value"],
            "cash": portfolio_state["cash"],
            "daily_return": portfolio_state["daily_return"],
            "cumulative_return": portfolio_state["cumulative_return"],
            "transaction_costs": portfolio_state.get("transaction_costs", 0.0),
            "cumulative_costs": portfolio_state["cumulative_costs"],
            "turnover": portfolio_state.get("turnover", 0.0),
        }
        
        # Add individual position weights
        for asset, weight in portfolio_state["weights"].items():
            state_record[f"weight_{asset}"] = weight

        self.portfolio_history.append(state_record)

    def _compile_results(self) -> dict[str, Any]:
        """Compile all backtest results into structured format."""
        # Convert history to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index("date")
        
        # Calculate summary statistics
        total_return = portfolio_df["cumulative_return"].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = portfolio_df["daily_return"].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.cash_rate * 252) / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        portfolio_values = portfolio_df["total_value"]
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calculate other metrics
        total_costs = portfolio_df["cumulative_costs"].iloc[-1]
        cost_ratio = total_costs / self.initial_capital
        avg_turnover = np.mean([r.get("turnover", 0) for r in self.rebalance_history])

        results = {
            # Time series data
            "portfolio_history": portfolio_df,
            "rebalance_history": pd.DataFrame(self.rebalance_history),
            
            # Summary statistics
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            
            # Cost analysis
            "total_transaction_costs": total_costs,
            "cost_ratio": cost_ratio,
            "average_turnover": avg_turnover,
            "num_rebalances": len(self.rebalance_history),
            
            # Portfolio characteristics
            "initial_capital": self.initial_capital,
            "final_value": portfolio_df["total_value"].iloc[-1],
            "backtest_start": portfolio_df.index[0],
            "backtest_end": portfolio_df.index[-1],
            "trading_days": len(portfolio_df),
        }

        # Add benchmark comparison if available
        if self.benchmark_returns is not None:
            benchmark_aligned = self.benchmark_returns.loc[portfolio_df.index]
            benchmark_cumret = (1 + benchmark_aligned).cumprod() - 1
            benchmark_total_return = benchmark_cumret.iloc[-1]
            
            results.update({
                "benchmark_total_return": benchmark_total_return,
                "excess_return": total_return - benchmark_total_return,
                "tracking_error": (portfolio_df["daily_return"] - benchmark_aligned).std() * np.sqrt(252),
                "information_ratio": (annualized_return - benchmark_total_return * 252 / len(portfolio_df)) / 
                                   (portfolio_df["daily_return"] - benchmark_aligned).std() * np.sqrt(252)
                                   if (portfolio_df["daily_return"] - benchmark_aligned).std() > 0 else 0,
            })

        return results

    def get_portfolio_weights_over_time(self) -> pd.DataFrame:
        """Extract portfolio weights time series from backtest results."""
        if not self.portfolio_history:
            raise ValueError("No backtest results available. Run backtest first.")
        
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index("date")
        
        # Extract weight columns
        weight_cols = [col for col in portfolio_df.columns if col.startswith("weight_")]
        weights_df = portfolio_df[weight_cols].copy()
        
        # Clean column names (remove 'weight_' prefix)
        weights_df.columns = [col.replace("weight_", "") for col in weights_df.columns]
        
        return weights_df

    def export_results(self, output_path: str) -> None:
        """Export backtest results to files."""
        if not self.results:
            raise ValueError("No backtest results to export. Run backtest first.")
        
        from pathlib import Path
        import yaml
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export portfolio history to CSV (safer than parquet for complex data)
        self.results["portfolio_history"].to_csv(output_path / "portfolio_history.csv")
        
        # Export rebalance history to CSV (contains Series objects, not parquet-safe)
        if not self.results["rebalance_history"].empty:
            # Convert Series columns to string representation for CSV export
            rebalance_df = self.results["rebalance_history"].copy()
            for col in rebalance_df.columns:
                if rebalance_df[col].dtype == 'object':
                    rebalance_df[col] = rebalance_df[col].astype(str)
            rebalance_df.to_csv(output_path / "rebalance_history.csv")
        
        # Export summary metrics
        summary_metrics = {k: v for k, v in self.results.items() 
                          if not isinstance(v, pd.DataFrame)}
        
        # Convert numpy types to native Python types for YAML serialization
        for key, value in summary_metrics.items():
            if hasattr(value, 'item'):
                summary_metrics[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                summary_metrics[key] = value.isoformat()
        
        with open(output_path / "summary_metrics.yaml", "w") as f:
            yaml.dump(summary_metrics, f, default_flow_style=False)
        
        logger.info(f"Backtest results exported to {output_path}")