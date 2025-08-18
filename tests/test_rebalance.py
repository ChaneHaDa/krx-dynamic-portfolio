"""Test cases for dynamic rebalancing module."""


import numpy as np
import pandas as pd
import pytest

from krx_portfolio.models.rebalance import Rebalancer


class TestRebalancer:
    """Test cases for Rebalancer class."""

    @pytest.fixture
    def business_dates(self):
        """Create sample business dates."""
        return pd.bdate_range(start="2023-01-01", end="2023-12-31", freq="B")

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        return pd.Series([150.0, 2800.0, 350.0, 200.0, 450.0], index=assets)

    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer instance."""
        return Rebalancer(
            schedule="month_end",
            turnover_budget=0.2,
            rebalance_threshold=0.05,
            tc_bps=25.0,
        )

    def test_initialization(self):
        """Test rebalancer initialization."""
        rebalancer = Rebalancer(
            schedule="quarter_end",
            turnover_budget=0.3,
            rebalance_threshold=0.08,
            tc_bps=30.0,
            min_trade_size=0.002,
        )

        assert rebalancer.schedule == "quarter_end"
        assert rebalancer.turnover_budget == 0.3
        assert rebalancer.rebalance_threshold == 0.08
        assert rebalancer.tc_bps == 30.0 / 10000.0
        assert rebalancer.min_trade_size == 0.002

    def test_month_end_dates_generation(self, rebalancer, business_dates):
        """Test month-end rebalancing dates generation."""
        start_date = pd.Timestamp("2023-03-01")
        end_date = pd.Timestamp("2023-06-30")

        rebalance_dates = rebalancer.next_rebalance_dates(
            business_dates, start_date, end_date
        )

        # Should have approximately 4 month-end dates
        assert len(rebalance_dates) >= 3
        assert len(rebalance_dates) <= 5

        # All dates should be business days
        for date in rebalance_dates:
            assert date in business_dates

    def test_quarter_end_dates_generation(self, business_dates):
        """Test quarter-end rebalancing dates generation."""
        rebalancer = Rebalancer(schedule="quarter_end")
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-12-31")

        rebalance_dates = rebalancer.next_rebalance_dates(
            business_dates, start_date, end_date
        )

        # Should have 4 quarter-end dates
        assert len(rebalance_dates) == 4

    def test_weekly_dates_generation(self, business_dates):
        """Test weekly rebalancing dates generation."""
        rebalancer = Rebalancer(schedule="weekly")
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-31")

        rebalance_dates = rebalancer.next_rebalance_dates(
            business_dates, start_date, end_date
        )

        # Should have approximately 4-5 weekly dates
        assert len(rebalance_dates) >= 4
        assert len(rebalance_dates) <= 6

    def test_daily_dates_generation(self, business_dates):
        """Test daily rebalancing dates generation."""
        rebalancer = Rebalancer(schedule="daily")
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-10")

        rebalance_dates = rebalancer.next_rebalance_dates(
            business_dates, start_date, end_date
        )

        # Should include all business days in range
        expected_dates = business_dates[
            (business_dates >= start_date) & (business_dates <= end_date)
        ]
        assert len(rebalance_dates) == len(expected_dates)

    def test_unknown_schedule_raises_error(self, business_dates):
        """Test that unknown schedule raises error."""
        rebalancer = Rebalancer(schedule="unknown")

        with pytest.raises(ValueError, match="Unknown schedule"):
            rebalancer.next_rebalance_dates(business_dates)

    def test_should_rebalance_scheduled_date(self, rebalancer):
        """Test rebalancing on scheduled dates."""
        w_target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        w_current = np.array([0.21, 0.19, 0.21, 0.19, 0.2])  # Small differences

        rebalance_date = pd.Timestamp("2023-01-31")
        rebalance_dates = [rebalance_date]

        should_rebalance = rebalancer.should_rebalance(
            w_target, w_current, rebalance_date, rebalance_dates
        )

        assert should_rebalance is True

    def test_should_rebalance_threshold_triggered(self, rebalancer):
        """Test rebalancing when threshold is exceeded."""
        w_target = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # L1 distance = 0.2 > 0.05

        non_rebalance_date = pd.Timestamp("2023-01-15")
        rebalance_dates = []

        should_rebalance = rebalancer.should_rebalance(
            w_target, w_current, non_rebalance_date, rebalance_dates
        )

        assert should_rebalance is True

    def test_should_not_rebalance_below_threshold(self, rebalancer):
        """Test no rebalancing when below threshold."""
        w_target = np.array([0.21, 0.20, 0.20, 0.20, 0.19])
        w_current = np.array(
            [0.20, 0.20, 0.20, 0.20, 0.20]
        )  # L1 distance = 0.04 < 0.05

        non_rebalance_date = pd.Timestamp("2023-01-15")
        rebalance_dates = []

        should_rebalance = rebalancer.should_rebalance(
            w_target, w_current, non_rebalance_date, rebalance_dates
        )

        assert should_rebalance is False

    def test_apply_rebalancing_basic(self, rebalancer, sample_prices):
        """Test basic rebalancing application."""
        w_target = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        portfolio_value = 1000000.0

        result = rebalancer.apply(w_target, w_current, sample_prices, portfolio_value)

        # Check result structure
        expected_keys = {
            "w_executed",
            "w_diff",
            "orders",
            "turnover",
            "tc_cost",
            "portfolio_value",
        }
        assert set(result.keys()) == expected_keys

        # Check weight constraints
        assert abs(np.sum(result["w_executed"]) - 1.0) < 1e-6
        assert np.all(result["w_executed"] >= 0)

        # Check turnover calculation
        assert result["turnover"] >= 0
        assert result["tc_cost"] >= 0

    def test_apply_with_asset_names(self, rebalancer, sample_prices):
        """Test rebalancing with explicit asset names."""
        w_target = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        asset_names = sample_prices.index.tolist()

        result = rebalancer.apply(
            w_target, w_current, sample_prices, asset_names=asset_names
        )

        # Orders should have correct asset names
        if not result["orders"].empty:
            assert set(result["orders"]["asset"]).issubset(set(asset_names))

    def test_turnover_budget_constraint(self, sample_prices):
        """Test turnover budget constraint."""
        rebalancer = Rebalancer(turnover_budget=0.1)  # Low budget

        w_target = np.array([0.5, 0.1, 0.1, 0.1, 0.2])  # High turnover
        w_current = np.array([0.1, 0.3, 0.2, 0.2, 0.2])

        result = rebalancer.apply(w_target, w_current, sample_prices)

        # Turnover should not exceed budget
        assert result["turnover"] <= rebalancer.turnover_budget + 1e-6

    def test_minimum_trade_size_filter(self, sample_prices):
        """Test minimum trade size filtering."""
        rebalancer = Rebalancer(min_trade_size=0.05)  # 5% minimum

        w_target = np.array([0.21, 0.20, 0.19, 0.20, 0.20])  # Small changes
        w_current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

        result = rebalancer.apply(w_target, w_current, sample_prices)

        # Most trades should be filtered out
        significant_trades = np.abs(result["w_diff"]) >= rebalancer.min_trade_size
        assert np.sum(significant_trades) <= 1  # At most one significant trade

    def test_position_limits(self, sample_prices):
        """Test maximum position limits."""
        rebalancer = Rebalancer(max_positions=3)

        w_target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 5 positions
        w_current = np.zeros(5)  # Starting from zero

        result = rebalancer.apply(w_target, w_current, sample_prices)

        # Should have at most 3 positions
        positions = np.sum(result["w_executed"] > 1e-6)
        assert positions <= 3

    def test_orders_dataframe_structure(self, rebalancer, sample_prices):
        """Test orders DataFrame structure."""
        w_target = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = rebalancer.apply(w_target, w_current, sample_prices)
        orders = result["orders"]

        if not orders.empty:
            expected_columns = {
                "asset",
                "weight_change",
                "dollar_amount",
                "shares",
                "price",
                "side",
            }
            assert set(orders.columns) == expected_columns

            # Check side assignment
            buy_orders = orders[orders["side"] == "buy"]
            sell_orders = orders[orders["side"] == "sell"]

            assert np.all(buy_orders["weight_change"] > 0)
            assert np.all(sell_orders["weight_change"] < 0)

    def test_transaction_cost_calculation(self, sample_prices):
        """Test transaction cost calculation."""
        rebalancer1 = Rebalancer(tc_bps=10)  # Low cost
        rebalancer2 = Rebalancer(tc_bps=50)  # High cost

        w_target = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result1 = rebalancer1.apply(w_target, w_current, sample_prices)
        result2 = rebalancer2.apply(w_target, w_current, sample_prices)

        # Higher tc_bps should result in higher costs
        assert result2["tc_cost"] > result1["tc_cost"]

    def test_find_closest_business_day(self, rebalancer, business_dates):
        """Test finding closest business day."""
        # Test weekend date
        weekend_date = pd.Timestamp("2023-07-01")  # Saturday

        closest = rebalancer._find_closest_business_day(weekend_date, business_dates)

        # Should find a nearby business day
        assert closest in business_dates
        assert abs((closest - weekend_date).days) <= 3

    def test_find_closest_business_day_prefer_earlier(self, rebalancer, business_dates):
        """Test finding closest business day with preference for earlier dates."""
        # Test holiday that falls on weekday
        holiday_date = pd.Timestamp("2023-07-04")  # Tuesday (Independence Day)

        closest = rebalancer._find_closest_business_day(
            holiday_date, business_dates, prefer_earlier=True
        )

        assert closest in business_dates
        # Should prefer earlier date when prefer_earlier=True
        if closest != holiday_date:
            assert closest < holiday_date

    def test_no_trades_when_weights_identical(self, rebalancer, sample_prices):
        """Test no trades when target and current weights are identical."""
        w_same = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = rebalancer.apply(w_same, w_same, sample_prices)

        assert result["turnover"] == 0.0
        assert result["tc_cost"] == 0.0
        assert result["orders"].empty
        np.testing.assert_array_equal(result["w_executed"], w_same)

    def test_weight_normalization_after_constraints(self, rebalancer, sample_prices):
        """Test that weights are properly normalized after applying constraints."""
        # Create scenario where constraints might affect normalization
        w_target = np.array([0.4, 0.3, 0.2, 0.08, 0.02])  # Sum = 1.0
        w_current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = rebalancer.apply(w_target, w_current, sample_prices)

        # Final weights should still sum to 1
        assert abs(np.sum(result["w_executed"]) - 1.0) < 1e-6

    @pytest.mark.parametrize("schedule", ["month_end", "quarter_end", "weekly"])
    def test_schedule_types_produce_dates(self, business_dates, schedule):
        """Test that all schedule types produce valid dates."""
        rebalancer = Rebalancer(schedule=schedule)

        dates = rebalancer.next_rebalance_dates(business_dates)

        # Should produce some dates
        assert len(dates) > 0

        # All dates should be business days
        for date in dates:
            assert date in business_dates

    def test_rebalancing_with_zero_prices_handled(self, rebalancer):
        """Test handling of zero prices in rebalancing."""
        prices_with_zero = pd.Series([150.0, 0.0, 350.0, 200.0, 450.0])
        w_target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        w_current = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

        # Should not crash with zero price
        result = rebalancer.apply(w_target, w_current, prices_with_zero)

        # Result should be valid
        assert abs(np.sum(result["w_executed"]) - 1.0) < 1e-6
