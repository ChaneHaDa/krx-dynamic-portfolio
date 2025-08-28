"""Main backtesting pipeline integration module."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import yaml

from ..etl.main import run_etl_pipeline
from ..models.pipeline import PortfolioOptimizationPipeline
from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .risk_analytics import RiskAnalytics

logger = logging.getLogger(__name__)


class BacktestPipeline:
    """
    Complete backtesting pipeline integrating ETL, optimization, and backtesting.
    
    This pipeline provides end-to-end backtesting workflow:
    1. Data loading and preprocessing via ETL
    2. Portfolio weight generation via optimization
    3. Performance simulation via backtesting engine
    4. Comprehensive analysis and reporting
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize backtesting pipeline.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to YAML configuration file
        config : dict, optional
            Configuration dictionary
        """
        if config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config is not None:
            self.config = config.copy()
        else:
            self.config = self._get_default_config()

        # Initialize components
        self.portfolio_pipeline = PortfolioOptimizationPipeline(
            config=self.config.get("portfolio", {})
        )
        
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get("backtest", {}).get("initial_capital", 1_000_000),
            transaction_cost_bps=self.config.get("backtest", {}).get("transaction_cost_bps", 25.0),
            market_impact_model=self.config.get("backtest", {}).get("market_impact_model", "linear"),
            cash_rate=self.config.get("backtest", {}).get("cash_rate", 0.02),
        )
        
        self.performance_metrics = PerformanceMetrics(
            risk_free_rate=self.config.get("analysis", {}).get("risk_free_rate", 0.02),
            confidence_level=self.config.get("analysis", {}).get("confidence_level", 0.05),
        )
        
        self.risk_analytics = RiskAnalytics(
            confidence_levels=self.config.get("analysis", {}).get("var_confidence_levels", [0.01, 0.05, 0.10]),
            risk_free_rate=self.config.get("analysis", {}).get("risk_free_rate", 0.02),
        )

    def run_full_backtest(
        self,
        data_root: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run complete end-to-end backtesting pipeline.

        Parameters
        ----------
        data_root : str
            Path to raw data directory
        start_date : str, optional
            Backtest start date (YYYY-MM-DD)
        end_date : str, optional
            Backtest end date (YYYY-MM-DD)
        output_dir : str, optional
            Output directory for results

        Returns
        -------
        dict
            Complete backtesting results
        """
        logger.info("Starting full backtesting pipeline...")
        
        # Step 1: Run ETL pipeline
        logger.info("Step 1: Running ETL pipeline...")
        etl_results = self._run_etl(data_root, start_date, end_date)
        
        # Step 2: Generate portfolio weights
        logger.info("Step 2: Generating portfolio weights...")
        weights_results = self._generate_weights(etl_results, start_date, end_date)
        
        # Step 3: Run backtest simulation
        logger.info("Step 3: Running backtest simulation...")
        backtest_results = self._run_backtest(
            weights_results["weights"],
            etl_results["returns"],
            etl_results.get("prices"),
            start_date,
            end_date,
        )
        
        # Step 4: Calculate performance metrics
        logger.info("Step 4: Calculating performance metrics...")
        performance_results = self._calculate_performance_metrics(backtest_results)
        
        # Step 5: Risk analysis
        logger.info("Step 5: Performing risk analysis...")
        risk_results = self._perform_risk_analysis(backtest_results)
        
        # Step 6: Compile final results
        final_results = {
            "backtest_config": self.config,
            "etl_summary": {
                "data_period": (etl_results["returns"].index[0], etl_results["returns"].index[-1]),
                "n_assets": len(etl_results["returns"].columns),
                "n_observations": len(etl_results["returns"]),
            },
            "backtest_results": backtest_results,
            "performance_metrics": performance_results,
            "risk_analysis": risk_results,
            "portfolio_weights": weights_results,
            "run_timestamp": datetime.now().isoformat(),
        }
        
        # Step 7: Export results
        if output_dir is not None:
            logger.info(f"Step 7: Exporting results to {output_dir}")
            self._export_results(final_results, output_dir)
        
        logger.info("Backtesting pipeline completed successfully!")
        return final_results

    def run_backtest_with_weights(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run backtesting with pre-generated weights.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights over time
        returns : pd.DataFrame
            Asset returns
        prices : pd.DataFrame, optional
            Asset prices
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        start_date : str, optional
            Start date
        end_date : str, optional
            End date
        output_dir : str, optional
            Output directory

        Returns
        -------
        dict
            Backtesting results
        """
        logger.info("Running backtesting with provided weights...")
        
        # Convert date strings to timestamps if provided
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None
        
        # Update backtest engine with benchmark if provided
        if benchmark_returns is not None:
            self.backtest_engine.benchmark_returns = benchmark_returns
        
        # Run backtest
        backtest_results = self.backtest_engine.run_backtest(
            weights=weights,
            returns=returns,
            prices=prices,
            start_date=start_ts,
            end_date=end_ts,
        )
        
        # Calculate metrics
        portfolio_returns = backtest_results["portfolio_history"]["daily_return"]
        
        performance_metrics = self.performance_metrics.calculate_all_metrics(
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            portfolio_values=backtest_results["portfolio_history"]["total_value"],
        )
        
        risk_analysis = self._perform_risk_analysis(backtest_results)
        
        results = {
            "backtest_results": backtest_results,
            "performance_metrics": performance_metrics,
            "risk_analysis": risk_analysis,
            "run_timestamp": datetime.now().isoformat(),
        }
        
        # Export if requested
        if output_dir is not None:
            self._export_results(results, output_dir)
        
        return results

    def _run_etl(
        self, 
        data_root: str, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> dict[str, Any]:
        """Run ETL pipeline to get processed data."""
        etl_config = self.config.get("etl", {})
        
        # Run ETL pipeline
        etl_results = run_etl_pipeline(
            data_root=data_root,
            start_date=start_date,
            end_date=end_date,
            cache_dir=etl_config.get("cache_dir", "data/cache"),
            force_reload=etl_config.get("force_reload", False),
        )
        
        return etl_results

    def _generate_weights(
        self,
        etl_results: dict[str, Any],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> dict[str, Any]:
        """Generate portfolio weights using optimization pipeline."""
        returns = etl_results["returns"]
        
        # Filter date range if specified
        if start_date or end_date:
            start_ts = pd.Timestamp(start_date) if start_date else returns.index[0]
            end_ts = pd.Timestamp(end_date) if end_date else returns.index[-1]
            date_mask = (returns.index >= start_ts) & (returns.index <= end_ts)
            returns = returns[date_mask]
        
        # Generate expected returns (simple historical mean for now)
        lookback_window = self.config.get("portfolio", {}).get("lookback_window", 252)
        expected_returns = []
        
        rebalance_config = self.config.get("portfolio", {}).get("rebalance", {})
        rebalancer_freq = rebalance_config.get("schedule", "month_end")
        
        # Generate rebalance dates
        if rebalancer_freq == "month_end":
            rebalance_dates = pd.date_range(
                start=returns.index[lookback_window],
                end=returns.index[-1],
                freq='M'
            )
        elif rebalancer_freq == "quarter_end":
            rebalance_dates = pd.date_range(
                start=returns.index[lookback_window],
                end=returns.index[-1],
                freq='Q'
            )
        else:
            # Monthly default
            rebalance_dates = pd.date_range(
                start=returns.index[lookback_window],
                end=returns.index[-1],
                freq='M'
            )
        
        # Generate expected returns for each rebalance date
        for date in rebalance_dates:
            if date in returns.index:
                # Get historical data up to this date
                hist_end_idx = returns.index.get_loc(date)
                hist_start_idx = max(0, hist_end_idx - lookback_window)
                hist_returns = returns.iloc[hist_start_idx:hist_end_idx+1]
                
                if len(hist_returns) >= 30:  # Minimum data requirement
                    exp_ret = hist_returns.mean() * 252  # Annualized
                    expected_returns.append(pd.Series(exp_ret.values, index=exp_ret.index, name=date))
        
        expected_returns_df = pd.DataFrame(expected_returns)
        
        # Generate weights using portfolio pipeline
        weights = self.portfolio_pipeline.generate_weight_series(
            returns=returns,
            expected_returns=expected_returns_df,
            rebalance_dates=list(rebalance_dates),
            prices=etl_results.get("prices"),
        )
        
        return {
            "weights": weights,
            "expected_returns": expected_returns_df,
            "rebalance_dates": rebalance_dates,
        }

    def _run_backtest(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> dict[str, Any]:
        """Run backtest simulation."""
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None
        
        return self.backtest_engine.run_backtest(
            weights=weights,
            returns=returns,
            prices=prices,
            start_date=start_ts,
            end_date=end_ts,
        )

    def _calculate_performance_metrics(
        self, backtest_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        portfolio_returns = backtest_results["portfolio_history"]["daily_return"]
        portfolio_values = backtest_results["portfolio_history"]["total_value"]
        
        # Get benchmark returns if available
        benchmark_returns = None
        if hasattr(self.backtest_engine, 'benchmark_returns') and self.backtest_engine.benchmark_returns is not None:
            benchmark_returns = self.backtest_engine.benchmark_returns
        
        # Calculate all metrics
        metrics = self.performance_metrics.calculate_all_metrics(
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
        )
        
        # Add rolling metrics
        rolling_window = self.config.get("analysis", {}).get("rolling_window", 252)
        rolling_metrics = self.performance_metrics.rolling_metrics(
            returns=portfolio_returns,
            window=rolling_window,
            metrics=["sharpe_ratio", "volatility", "max_drawdown"],
        )
        
        return {
            "summary_metrics": metrics,
            "rolling_metrics": rolling_metrics,
        }

    def _perform_risk_analysis(self, backtest_results: dict[str, Any]) -> dict[str, Any]:
        """Perform comprehensive risk analysis."""
        portfolio_returns = backtest_results["portfolio_history"]["daily_return"]
        
        risk_results = {}
        
        # VaR analysis
        for confidence_level in self.risk_analytics.confidence_levels:
            var_results = self.risk_analytics.calculate_var(
                returns=portfolio_returns,
                confidence_level=confidence_level,
                method='all',
            )
            
            cvar_result = self.risk_analytics.calculate_cvar(
                returns=portfolio_returns,
                confidence_level=confidence_level,
                method='historical',
            )
            
            risk_results[f"var_{int(confidence_level*100)}"] = var_results
            risk_results[f"cvar_{int(confidence_level*100)}"] = cvar_result
        
        # Extreme value analysis
        try:
            eva_results = self.risk_analytics.extreme_value_analysis(
                returns=portfolio_returns,
                method='peaks_over_threshold',
            )
            risk_results["extreme_value_analysis"] = eva_results
        except Exception as e:
            logger.warning(f"Extreme value analysis failed: {e}")
            risk_results["extreme_value_analysis"] = {"error": str(e)}
        
        # Correlation analysis if we have asset-level data
        if "weights" in backtest_results or hasattr(self, '_asset_returns'):
            try:
                # This would require asset-level returns - placeholder for now
                risk_results["correlation_analysis"] = {"note": "Requires asset-level returns"}
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
        
        return risk_results

    def _export_results(self, results: dict[str, Any], output_dir: str) -> None:
        """Export all results to specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export backtest engine results
        if "backtest_results" in results:
            self.backtest_engine.export_results(output_path / "backtest")
        
        # Export performance metrics
        if "performance_metrics" in results:
            perf_metrics = results["performance_metrics"]
            
            # Summary metrics
            with open(output_path / "performance_summary.yaml", "w") as f:
                summary_for_yaml = {}
                for key, value in perf_metrics["summary_metrics"].items():
                    if hasattr(value, 'item'):
                        summary_for_yaml[key] = value.item()
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        summary_for_yaml[key] = str(value)
                    else:
                        summary_for_yaml[key] = value
                yaml.dump(summary_for_yaml, f, default_flow_style=False)
            
            # Rolling metrics
            if "rolling_metrics" in perf_metrics:
                perf_metrics["rolling_metrics"].to_parquet(
                    output_path / "rolling_metrics.parquet"
                )
        
        # Export risk analysis
        if "risk_analysis" in results:
            with open(output_path / "risk_analysis.yaml", "w") as f:
                risk_for_yaml = {}
                for key, value in results["risk_analysis"].items():
                    if isinstance(value, dict):
                        risk_for_yaml[key] = {
                            k: v.item() if hasattr(v, 'item') else v
                            for k, v in value.items()
                        }
                    else:
                        risk_for_yaml[key] = value.item() if hasattr(value, 'item') else value
                yaml.dump(risk_for_yaml, f, default_flow_style=False)
        
        # Export portfolio weights if available
        if "portfolio_weights" in results and "weights" in results["portfolio_weights"]:
            results["portfolio_weights"]["weights"].to_parquet(
                output_path / "portfolio_weights.parquet"
            )
        
        # Export summary report
        self._generate_summary_report(results, output_path)
        
        logger.info(f"Results exported to {output_path}")

    def _generate_summary_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate a summary report in markdown format."""
        report_lines = []
        
        report_lines.append("# Backtesting Results Summary")
        report_lines.append("")
        report_lines.append(f"**Generated:** {results.get('run_timestamp', 'Unknown')}")
        report_lines.append("")
        
        # ETL Summary
        if "etl_summary" in results:
            etl = results["etl_summary"]
            report_lines.append("## Data Summary")
            report_lines.append(f"- **Period:** {etl.get('data_period', 'Unknown')}")
            report_lines.append(f"- **Assets:** {etl.get('n_assets', 'Unknown')}")
            report_lines.append(f"- **Observations:** {etl.get('n_observations', 'Unknown')}")
            report_lines.append("")
        
        # Performance Summary
        if "performance_metrics" in results and "summary_metrics" in results["performance_metrics"]:
            perf = results["performance_metrics"]["summary_metrics"]
            report_lines.append("## Performance Metrics")
            report_lines.append(f"- **Total Return:** {perf.get('total_return', 0):.2%}")
            report_lines.append(f"- **Annualized Return:** {perf.get('annualized_return', 0):.2%}")
            report_lines.append(f"- **Volatility:** {perf.get('volatility', 0):.2%}")
            report_lines.append(f"- **Sharpe Ratio:** {perf.get('sharpe_ratio', 0):.3f}")
            report_lines.append(f"- **Max Drawdown:** {perf.get('max_drawdown', 0):.2%}")
            report_lines.append(f"- **Calmar Ratio:** {perf.get('calmar_ratio', 0):.3f}")
            report_lines.append("")
        
        # Risk Metrics
        if "risk_analysis" in results:
            risk = results["risk_analysis"]
            report_lines.append("## Risk Analysis")
            
            if "var_5" in risk and "historical" in risk["var_5"]:
                report_lines.append(f"- **VaR (95%):** {risk['var_5']['historical']:.2%}")
            
            if "cvar_5" in risk:
                report_lines.append(f"- **CVaR (95%):** {risk['cvar_5']:.2%}")
            
            report_lines.append("")
        
        # Backtest Configuration
        if "backtest_config" in results:
            config = results["backtest_config"]
            report_lines.append("## Configuration")
            if "backtest" in config:
                bt_config = config["backtest"]
                report_lines.append(f"- **Initial Capital:** ${bt_config.get('initial_capital', 1000000):,.0f}")
                report_lines.append(f"- **Transaction Costs:** {bt_config.get('transaction_cost_bps', 25)} bps")
            report_lines.append("")
        
        # Write report
        with open(output_path / "summary_report.md", "w") as f:
            f.write("\n".join(report_lines))

    def _get_default_config(self) -> dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "etl": {
                "cache_dir": "data/cache",
                "force_reload": False,
            },
            "portfolio": {
                "objective": "max_sharpe",
                "risk_free_rate": 0.02,
                "lookback_window": 252,
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
                "var_confidence_levels": [0.01, 0.05, 0.10],
                "rolling_window": 252,
            },
        }


def main():
    """CLI entry point for backtesting pipeline."""
    parser = argparse.ArgumentParser(description="Run portfolio backtesting pipeline")
    parser.add_argument("--data-root", required=True, help="Root directory containing raw data")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize pipeline
    pipeline = BacktestPipeline(config_path=args.config)
    
    # Run backtesting
    results = pipeline.run_full_backtest(
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
    
    # Print summary
    if "performance_metrics" in results and "summary_metrics" in results["performance_metrics"]:
        perf = results["performance_metrics"]["summary_metrics"]
        print(f"\n=== Backtesting Results Summary ===")
        print(f"Total Return: {perf.get('total_return', 0):.2%}")
        print(f"Annualized Return: {perf.get('annualized_return', 0):.2%}")
        print(f"Volatility: {perf.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
    
    print(f"\nResults saved to: {args.output_dir or 'console only'}")


if __name__ == "__main__":
    main()