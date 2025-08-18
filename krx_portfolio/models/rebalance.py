"""Dynamic portfolio rebalancing module."""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd


class Rebalancer:
    """Dynamic portfolio rebalancing with cost-aware execution."""

    def __init__(
        self,
        schedule: Literal["month_end", "quarter_end", "weekly", "daily"] = "month_end",
        turnover_budget: float = 0.2,
        rebalance_threshold: float = 0.05,
        tc_bps: float = 25.0,
        min_trade_size: float = 0.001,
        max_positions: Optional[int] = None,
    ):
        """
        Initialize rebalancer.

        Parameters
        ----------
        schedule : str
            Rebalancing schedule ('month_end', 'quarter_end', 'weekly', 'daily')
        turnover_budget : float
            Maximum monthly turnover as fraction of portfolio value
        rebalance_threshold : float
            L1 distance threshold to trigger rebalancing
        tc_bps : float
            Transaction cost in basis points (round-trip)
        min_trade_size : float
            Minimum trade size as fraction of portfolio
        max_positions : int, optional
            Maximum number of positions
        """
        self.schedule = schedule
        self.turnover_budget = turnover_budget
        self.rebalance_threshold = rebalance_threshold
        self.tc_bps = tc_bps / 10000.0  # Convert bps to decimal
        self.min_trade_size = min_trade_size
        self.max_positions = max_positions

    def next_rebalance_dates(
        self,
        dates: pd.DatetimeIndex,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> List[pd.Timestamp]:
        """
        Generate rebalancing dates with business day adjustments.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Available trading dates
        start_date : pd.Timestamp, optional
            Start date for rebalancing schedule
        end_date : pd.Timestamp, optional
            End date for rebalancing schedule

        Returns
        -------
        list
            List of rebalancing dates
        """
        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = dates[-1]

        rebalance_dates = []

        if self.schedule == "month_end":
            rebalance_dates = self._get_month_end_dates(dates, start_date, end_date)
        elif self.schedule == "quarter_end":
            rebalance_dates = self._get_quarter_end_dates(dates, start_date, end_date)
        elif self.schedule == "weekly":
            rebalance_dates = self._get_weekly_dates(dates, start_date, end_date)
        elif self.schedule == "daily":
            rebalance_dates = dates[
                (dates >= start_date) & (dates <= end_date)
            ].tolist()
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return rebalance_dates

    def should_rebalance(
        self,
        w_target: np.ndarray,
        w_current: np.ndarray,
        date: pd.Timestamp,
        rebalance_dates: List[pd.Timestamp],
    ) -> bool:
        """
        Determine if rebalancing should occur.

        Parameters
        ----------
        w_target : np.ndarray
            Target weights
        w_current : np.ndarray
            Current weights
        date : pd.Timestamp
            Current date
        rebalance_dates : list
            Scheduled rebalance dates

        Returns
        -------
        bool
            Whether to rebalance
        """
        # Check if it's a scheduled rebalance date
        if date in rebalance_dates:
            return True

        # Check threshold-based rebalancing
        l1_distance = np.sum(np.abs(w_target - w_current))
        if l1_distance >= self.rebalance_threshold:
            return True

        return False

    def apply(
        self,
        w_target: np.ndarray,
        w_current: np.ndarray,
        prices: pd.Series,
        portfolio_value: float = 1.0,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply rebalancing with transaction cost optimization.

        Parameters
        ----------
        w_target : np.ndarray
            Target portfolio weights
        w_current : np.ndarray
            Current portfolio weights
        prices : pd.Series
            Asset prices for execution
        portfolio_value : float
            Total portfolio value
        asset_names : list, optional
            Asset names for order tracking

        Returns
        -------
        dict
            Rebalancing results with execution details
        """
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(w_target))]

        # Calculate raw trade sizes
        w_diff = w_target - w_current

        # Apply minimum trade size filter
        w_diff = self._apply_min_trade_filter(w_diff)

        # Apply turnover budget constraint
        w_diff = self._apply_turnover_constraint(w_diff)

        # Apply position limits
        if self.max_positions is not None:
            w_diff = self._apply_position_limits(w_target, w_current, w_diff)

        # Calculate final executed weights
        w_executed = w_current + w_diff

        # Renormalize if needed
        if np.abs(np.sum(w_executed) - 1.0) > 1e-6:
            w_executed = w_executed / np.sum(w_executed)
            w_diff = w_executed - w_current

        # Calculate order details
        turnover = np.sum(np.abs(w_diff))
        tc_cost = turnover * self.tc_bps

        # Create orders DataFrame
        orders = self._create_orders(w_diff, prices, portfolio_value, asset_names)

        return {
            "w_executed": w_executed,
            "w_diff": w_diff,
            "orders": orders,
            "turnover": turnover,
            "tc_cost": tc_cost,
            "portfolio_value": portfolio_value,
        }

    def _get_month_end_dates(
        self, dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Get month-end business dates."""
        # Generate month ends
        monthly_dates = pd.date_range(
            start=start_date, end=end_date, freq="ME"  # Use ME instead of deprecated M
        )

        # Adjust to business days
        rebalance_dates = []
        for date in monthly_dates:
            # Find closest business day (prefer earlier)
            business_date = self._find_closest_business_day(
                date, dates, prefer_earlier=True
            )
            if business_date is not None:
                rebalance_dates.append(business_date)

        return rebalance_dates

    def _get_quarter_end_dates(
        self, dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Get quarter-end business dates."""
        quarterly_dates = pd.date_range(
            start=start_date, end=end_date, freq="QE"  # Use QE instead of deprecated Q
        )

        rebalance_dates = []
        for date in quarterly_dates:
            business_date = self._find_closest_business_day(
                date, dates, prefer_earlier=True
            )
            if business_date is not None:
                rebalance_dates.append(business_date)

        return rebalance_dates

    def _get_weekly_dates(
        self, dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Get weekly business dates (Fridays)."""
        weekly_dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")

        rebalance_dates = []
        for date in weekly_dates:
            business_date = self._find_closest_business_day(date, dates)
            if business_date is not None:
                rebalance_dates.append(business_date)

        return rebalance_dates

    def _find_closest_business_day(
        self,
        date: pd.Timestamp,
        business_dates: pd.DatetimeIndex,
        prefer_earlier: bool = False,
    ) -> Optional[pd.Timestamp]:
        """Find closest business day to given date."""
        if date in business_dates:
            return date

        # Find nearest business days
        earlier_dates = business_dates[business_dates <= date]
        later_dates = business_dates[business_dates > date]

        if prefer_earlier and len(earlier_dates) > 0:
            return earlier_dates[-1]
        elif len(later_dates) > 0 and len(earlier_dates) > 0:
            # Choose closest
            days_back = (date - earlier_dates[-1]).days
            days_forward = (later_dates[0] - date).days
            return earlier_dates[-1] if days_back <= days_forward else later_dates[0]
        elif len(earlier_dates) > 0:
            return earlier_dates[-1]
        elif len(later_dates) > 0:
            return later_dates[0]
        else:
            return None

    def _apply_min_trade_filter(self, w_diff: np.ndarray) -> np.ndarray:
        """Filter out trades smaller than minimum size."""
        mask = np.abs(w_diff) >= self.min_trade_size
        return w_diff * mask

    def _apply_turnover_constraint(self, w_diff: np.ndarray) -> np.ndarray:
        """Apply turnover budget constraint."""
        total_turnover = np.sum(np.abs(w_diff))
        if total_turnover <= self.turnover_budget:
            return w_diff

        # Scale down proportionally
        scale_factor = self.turnover_budget / total_turnover
        return w_diff * scale_factor

    def _apply_position_limits(
        self, w_target: np.ndarray, w_current: np.ndarray, w_diff: np.ndarray
    ) -> np.ndarray:
        """Apply maximum position limits."""
        w_new = w_current + w_diff

        # If we have too many positions, keep only the largest
        if np.sum(w_new > 0) > self.max_positions:
            # Get indices of largest positions
            top_indices = np.argsort(w_new)[-self.max_positions :]
            mask = np.zeros_like(w_new, dtype=bool)
            mask[top_indices] = True

            # Zero out smaller positions
            w_new = w_new * mask
            w_diff = w_new - w_current

        return w_diff

    def _create_orders(
        self,
        w_diff: np.ndarray,
        prices: pd.Series,
        portfolio_value: float,
        asset_names: List[str],
    ) -> pd.DataFrame:
        """Create detailed order information."""
        orders = []
        for i, (asset, weight_change) in enumerate(zip(asset_names, w_diff)):
            if abs(weight_change) < 1e-8:  # Skip tiny trades
                continue

            dollar_amount = weight_change * portfolio_value
            shares = dollar_amount / prices.iloc[i] if prices.iloc[i] > 0 else 0

            orders.append(
                {
                    "asset": asset,
                    "weight_change": weight_change,
                    "dollar_amount": dollar_amount,
                    "shares": shares,
                    "price": prices.iloc[i],
                    "side": "buy" if weight_change > 0 else "sell",
                }
            )

        return pd.DataFrame(orders)
