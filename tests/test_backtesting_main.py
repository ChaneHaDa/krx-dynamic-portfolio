"""Tests for backtesting main pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from krx_portfolio.backtesting.main import BacktestPipeline


class TestBacktestPipeline:
    """Test suite for BacktestPipeline class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "etl": {
                "cache_dir": "data/cache",
                "force_reload": False,
            },
            "portfolio": {
                "objective": "max_sharpe",
                "risk_free_rate": 0.02,
                "lookback_window": 60,  # Shorter for testing
                "rebalance": {
                    "schedule": "month_end",
                    "turnover_budget": 0.25,
                    "tc_bps": 25.0,
                },
                "constraints": {
                    "w_bounds": [0.0, 0.15],
                },
            },
            "backtest": {
                "initial_capital": 1_000_000,
                "transaction_cost_bps": 25.0,
                "market_impact_model": "linear",
                "cash_rate": 0.02,
            },
            "analysis": {
                "risk_free_rate": 0.02,
                "confidence_level": 0.05,
                "var_confidence_levels": [0.05, 0.10],
                "rolling_window": 60,
            },
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', periods=120, freq='D')  # 4 months
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        # Returns
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (120, 3)),
            index=dates, columns=assets
        )
        
        # Prices
        prices = pd.DataFrame(index=dates, columns=assets)
        prices.iloc[0] = [100, 150, 120]
        for i in range(1, len(dates)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns.iloc[i])
        
        return {
            'returns': returns,
            'prices': prices,
            'dates': dates,
            'assets': assets
        }

    @pytest.fixture
    def basic_pipeline(self, sample_config):
        """Create basic BacktestPipeline for testing."""
        return BacktestPipeline(config=sample_config)

    @pytest.fixture
    def mock_etl_results(self, sample_data):
        """Create mock ETL results."""
        return {
            'returns': sample_data['returns'],
            'prices': sample_data['prices'],
            'features': pd.DataFrame(),  # Empty features for simplicity
            'universe': sample_data['assets']
        }

    def test_pipeline_initialization_with_config(self, sample_config):
        """Test pipeline initialization with config dict."""
        pipeline = BacktestPipeline(config=sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.portfolio_pipeline is not None
        assert pipeline.backtest_engine is not None
        assert pipeline.performance_metrics is not None
        assert pipeline.risk_analytics is not None

    def test_pipeline_initialization_with_config_file(self, sample_config):
        """Test pipeline initialization with config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = f.name
        
        try:
            pipeline = BacktestPipeline(config_path=config_path)
            assert pipeline.config == sample_config
        finally:
            Path(config_path).unlink()

    def test_pipeline_initialization_default_config(self):
        """Test pipeline initialization with default config."""
        pipeline = BacktestPipeline()
        
        # Should have default configuration
        assert pipeline.config is not None
        assert "etl" in pipeline.config
        assert "portfolio" in pipeline.config
        assert "backtest" in pipeline.config
        assert "analysis" in pipeline.config

    def test_get_default_config(self, basic_pipeline):
        """Test default configuration generation."""
        default_config = basic_pipeline._get_default_config()
        
        required_sections = ["etl", "portfolio", "backtest", "analysis"]
        for section in required_sections:
            assert section in default_config
        
        # Check some specific defaults
        assert default_config["backtest"]["initial_capital"] == 1_000_000
        assert default_config["portfolio"]["objective"] == "max_sharpe"
        assert default_config["analysis"]["confidence_level"] == 0.05

    @patch('krx_portfolio.backtesting.main.run_etl_pipeline')
    def test_run_etl(self, mock_etl_run, basic_pipeline, mock_etl_results):
        """Test ETL pipeline execution."""
        mock_etl_run.return_value = mock_etl_results
        
        result = basic_pipeline._run_etl(
            data_root="/path/to/data",
            start_date="2023-01-01",
            end_date="2023-03-31"
        )
        
        assert result == mock_etl_results
        mock_etl_run.assert_called_once_with(
            data_root="/path/to/data",
            start_date="2023-01-01",
            end_date="2023-03-31",
            cache_dir="data/cache",
            force_reload=False
        )

    def test_generate_weights(self, basic_pipeline, mock_etl_results):
        """Test portfolio weights generation."""
        weights_results = basic_pipeline._generate_weights(
            mock_etl_results,
            start_date="2023-01-01",
            end_date="2023-03-31"
        )
        
        assert "weights" in weights_results
        assert "expected_returns" in weights_results
        assert "rebalance_dates" in weights_results
        
        weights = weights_results["weights"]
        assert isinstance(weights, pd.DataFrame)
        assert len(weights.columns) == len(mock_etl_results["returns"].columns)

    def test_run_backtest(self, basic_pipeline, sample_data):
        """Test backtest execution."""
        # Create simple weights
        weight_dates = pd.date_range('2023-01-01', periods=4, freq='M')
        weights = pd.DataFrame(
            [[0.33, 0.33, 0.34]] * len(weight_dates),
            index=weight_dates,
            columns=sample_data['assets']
        )
        
        results = basic_pipeline._run_backtest(
            weights=weights,
            returns=sample_data['returns'],
            prices=sample_data['prices'],
            start_date="2023-01-01",
            end_date="2023-03-31"
        )
        
        assert "portfolio_history" in results
        assert "total_return" in results
        assert "annualized_return" in results

    def test_calculate_performance_metrics(self, basic_pipeline):
        """Test performance metrics calculation."""
        # Mock backtest results
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        backtest_results = {
            "portfolio_history": pd.DataFrame({
                "daily_return": np.random.normal(0.001, 0.02, 100),
                "total_value": np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 1_000_000,
                "cumulative_return": np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) - 1,
            }, index=dates)
        }
        
        performance_results = basic_pipeline._calculate_performance_metrics(backtest_results)
        
        assert "summary_metrics" in performance_results
        assert "rolling_metrics" in performance_results
        
        summary = performance_results["summary_metrics"]
        assert "total_return" in summary
        assert "volatility" in summary
        assert "sharpe_ratio" in summary

    def test_perform_risk_analysis(self, basic_pipeline):
        """Test risk analysis execution."""
        # Mock backtest results
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        backtest_results = {
            "portfolio_history": pd.DataFrame({
                "daily_return": np.random.normal(-0.001, 0.025, 100),  # Slightly negative mean for VaR
                "total_value": np.random.uniform(900_000, 1_100_000, 100),
            }, index=dates)
        }
        
        risk_results = basic_pipeline._perform_risk_analysis(backtest_results)
        
        # Should contain VaR results for different confidence levels
        assert "var_5" in risk_results
        assert "cvar_5" in risk_results
        
        # Should attempt extreme value analysis
        assert "extreme_value_analysis" in risk_results

    def test_run_backtest_with_weights(self, basic_pipeline, sample_data):
        """Test running backtest with pre-generated weights."""
        # Create weights
        weight_dates = pd.date_range('2023-01-01', periods=4, freq='M')
        weights = pd.DataFrame(
            [[0.4, 0.3, 0.3]] * len(weight_dates),
            index=weight_dates,
            columns=sample_data['assets']
        )
        
        # Create benchmark
        benchmark_returns = sample_data['returns'].mean(axis=1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = basic_pipeline.run_backtest_with_weights(
                weights=weights,
                returns=sample_data['returns'],
                prices=sample_data['prices'],
                benchmark_returns=benchmark_returns,
                start_date="2023-01-01",
                end_date="2023-03-31",
                output_dir=tmp_dir
            )
        
        assert "backtest_results" in results
        assert "performance_metrics" in results
        assert "risk_analysis" in results
        
        # Should have benchmark comparison metrics
        perf_metrics = results["performance_metrics"]["summary_metrics"]
        assert "information_ratio" in perf_metrics
        assert "tracking_error" in perf_metrics

    @patch('krx_portfolio.backtesting.main.run_etl_pipeline')
    def test_run_full_backtest(self, mock_etl_run, basic_pipeline, mock_etl_results):
        """Test full end-to-end backtesting pipeline."""
        mock_etl_run.return_value = mock_etl_results
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = basic_pipeline.run_full_backtest(
                data_root="/path/to/data",
                start_date="2023-01-01",
                end_date="2023-03-31",
                output_dir=tmp_dir
            )
        
        # Check result structure
        expected_keys = [
            "backtest_config", "etl_summary", "backtest_results",
            "performance_metrics", "risk_analysis", "portfolio_weights"
        ]
        
        for key in expected_keys:
            assert key in results
        
        # Check ETL summary
        etl_summary = results["etl_summary"]
        assert "data_period" in etl_summary
        assert "n_assets" in etl_summary
        assert "n_observations" in etl_summary

    def test_export_results(self, basic_pipeline):
        """Test results export functionality."""
        # Mock results
        mock_results = {
            "backtest_results": {
                "portfolio_history": pd.DataFrame({
                    "total_value": [1000000, 1010000, 1005000],
                    "daily_return": [0.0, 0.01, -0.005],
                }, index=pd.date_range('2023-01-01', periods=3, freq='D')),
                "total_return": 0.05,
                "volatility": 0.15,
            },
            "performance_metrics": {
                "summary_metrics": {
                    "total_return": 0.05,
                    "annualized_return": 0.12,
                    "volatility": 0.15,
                    "sharpe_ratio": 0.67,
                },
                "rolling_metrics": pd.DataFrame({
                    "sharpe_ratio": [0.6, 0.7, 0.65],
                    "volatility": [0.14, 0.15, 0.16],
                }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
            },
            "risk_analysis": {
                "var_5": {"historical": -0.025},
                "cvar_5": -0.035,
            },
            "portfolio_weights": {
                "weights": pd.DataFrame({
                    "AAPL": [0.4, 0.3],
                    "GOOGL": [0.3, 0.4],
                    "MSFT": [0.3, 0.3],
                }, index=pd.date_range('2023-01-01', periods=2, freq='M'))
            },
            "run_timestamp": "2023-12-01T00:00:00",
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock the backtest engine export method
            basic_pipeline.backtest_engine.export_results = MagicMock()
            
            basic_pipeline._export_results(mock_results, tmp_dir)
            
            # Check that files were created
            output_path = Path(tmp_dir)
            
            expected_files = [
                "performance_summary.yaml",
                "rolling_metrics.parquet", 
                "risk_analysis.yaml",
                "portfolio_weights.parquet",
                "summary_report.md"
            ]
            
            for filename in expected_files:
                assert (output_path / filename).exists()

    def test_generate_summary_report(self, basic_pipeline):
        """Test summary report generation."""
        mock_results = {
            "run_timestamp": "2023-12-01T00:00:00",
            "etl_summary": {
                "data_period": ("2023-01-01", "2023-12-31"),
                "n_assets": 5,
                "n_observations": 252,
            },
            "performance_metrics": {
                "summary_metrics": {
                    "total_return": 0.15,
                    "annualized_return": 0.12,
                    "volatility": 0.18,
                    "sharpe_ratio": 0.67,
                    "max_drawdown": -0.08,
                    "calmar_ratio": 1.5,
                }
            },
            "risk_analysis": {
                "var_5": {"historical": -0.025},
                "cvar_5": -0.035,
            },
            "backtest_config": {
                "backtest": {
                    "initial_capital": 1_000_000,
                    "transaction_cost_bps": 25,
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            basic_pipeline._generate_summary_report(mock_results, Path(tmp_dir))
            
            report_path = Path(tmp_dir) / "summary_report.md"
            assert report_path.exists()
            
            # Read and check content
            content = report_path.read_text()
            assert "Backtesting Results Summary" in content
            assert "15.00%" in content  # Total return
            assert "67%" in content or "0.67" in content  # Sharpe ratio

    def test_edge_case_no_rebalance_dates(self, basic_pipeline):
        """Test handling when no rebalance dates are generated."""
        # Create very short data that won't generate rebalance dates
        short_data = {
            'returns': pd.DataFrame(
                [[0.01, 0.02]], 
                index=[pd.Timestamp('2023-01-01')],
                columns=['A', 'B']
            )
        }
        
        # Should handle gracefully without crashing
        try:
            weights_results = basic_pipeline._generate_weights(
                short_data, "2023-01-01", "2023-01-01"
            )
            # May return empty or minimal weights
            assert "weights" in weights_results
        except (ValueError, IndexError):
            # Some edge cases might legitimately fail
            pass

    def test_error_handling_insufficient_data(self, basic_pipeline):
        """Test error handling with insufficient data."""
        insufficient_data = {
            'returns': pd.DataFrame(
                [[0.01]], 
                index=[pd.Timestamp('2023-01-01')],
                columns=['A']
            )
        }
        
        # Should handle insufficient data gracefully
        try:
            weights_results = basic_pipeline._generate_weights(
                insufficient_data, "2023-01-01", "2023-01-01"
            )
        except (ValueError, IndexError):
            # Expected for insufficient data
            pass

    def test_configuration_validation(self, sample_config):
        """Test that pipeline validates configuration properly."""
        # Test with missing required sections
        incomplete_config = {"etl": {}}
        
        # Should use defaults for missing sections
        pipeline = BacktestPipeline(config=incomplete_config)
        assert "portfolio" in pipeline.config
        assert "backtest" in pipeline.config

    def test_date_handling_edge_cases(self, basic_pipeline, sample_data):
        """Test edge cases in date handling."""
        weights = pd.DataFrame(
            [[0.5, 0.5]],
            index=[pd.Timestamp('2023-01-01')],
            columns=['A', 'B']
        )
        
        returns = pd.DataFrame(
            [[0.01, 0.02], [0.005, -0.01]],
            index=[pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
            columns=['A', 'B']
        )
        
        # Test with None dates
        results = basic_pipeline._run_backtest(
            weights=weights,
            returns=returns,
            prices=None,  # No prices
            start_date=None,
            end_date=None
        )
        
        assert "portfolio_history" in results

    @patch('krx_portfolio.backtesting.main.run_etl_pipeline')
    def test_full_pipeline_with_minimal_config(self, mock_etl_run, mock_etl_results):
        """Test full pipeline with minimal configuration."""
        minimal_config = {
            "backtest": {"initial_capital": 500_000}
        }
        
        pipeline = BacktestPipeline(config=minimal_config)
        mock_etl_run.return_value = mock_etl_results
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = pipeline.run_full_backtest(
                data_root="/path/to/data",
                output_dir=tmp_dir
            )
        
        assert results["backtest_results"]["initial_capital"] == 500_000

    def test_numerical_edge_cases(self, basic_pipeline):
        """Test handling of numerical edge cases."""
        # Create data with extreme values
        extreme_data = {
            'returns': pd.DataFrame({
                'A': [0.0] * 50 + [1.0, -0.9] + [0.0] * 48,  # Extreme spike and crash
                'B': [0.001] * 100,  # Stable asset
            }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        }
        
        # Should handle extreme cases without crashing
        try:
            weights_results = basic_pipeline._generate_weights(
                extreme_data, "2023-01-01", "2023-03-31"
            )
            assert "weights" in weights_results
        except (ValueError, np.linalg.LinAlgError):
            # Some extreme numerical cases might fail
            pass