# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies
```bash
make dev          # Install development dependencies and setup pre-commit hooks
make install      # Install production dependencies only
```

### Code Quality
```bash
make lint         # Run all linters (ruff, black, mypy)
make format       # Auto-format code with black and ruff
make ci-test      # Run the same checks as CI (lint + test)
make pre-commit   # Manually run pre-commit hooks
```

### Testing
```bash
make test         # Run pytest with coverage
pytest tests/test_etl.py -v    # Run specific test module
pytest -v --cov=krx_portfolio --cov-report=html  # Detailed coverage report
```

### Data Pipeline
```bash
make etl          # Run ETL pipeline: python -m krx_portfolio.etl.main
make train        # Train portfolio optimization models: python -m krx_portfolio.models.train
make backtest     # Run backtesting: python -m krx_portfolio.backtesting.main
make app          # Launch Streamlit dashboard: streamlit run krx_portfolio/app/dashboard.py
```

## Project Architecture

### Core Structure
The project implements a complete financial data pipeline with four main components:

1. **ETL Pipeline** (`krx_portfolio/etl/`):
   - `data_loader.py`: KRX JSON data loading from external sources
   - `preprocessor.py`: Data cleaning, filtering, and feature engineering
   - `main.py`: CLI entry point with caching and date range handling

2. **Portfolio Models** (`krx_portfolio/models/`):
   - Portfolio optimization using Modern Portfolio Theory
   - Risk modeling and dynamic rebalancing algorithms
   - Model training and persistence

3. **Backtesting Engine** (`krx_portfolio/backtesting/`):
   - Historical strategy performance evaluation
   - Performance metrics calculation
   - Risk-adjusted return analysis

4. **Dashboard App** (`krx_portfolio/app/`):
   - Streamlit-based interactive visualization
   - Real-time portfolio monitoring
   - Strategy comparison and analysis

### Data Flow
1. Raw KRX data → ETL processing → Cached features
2. Features → Model training → Optimized portfolios
3. Portfolios → Backtesting → Performance metrics
4. All outputs → Dashboard visualization

### Key Configuration Files
- `pyproject.toml`: All project configuration (dependencies, tools, build settings)
- `Makefile`: Development workflow automation
- Data caching in `data/cache/` with structured subdirectories (`raw/`, `processed/`, `features/`)

### ETL Pipeline Details
The ETL pipeline requires a `--data-root` parameter pointing to KRX JSON data location. It supports:
- Date range filtering (`--start-date`, `--end-date`)
- Intelligent caching system with force reload option
- Investment universe generation (KOSPI top 100 + KOSDAQ top 50)
- Data quality reporting and validation

### Code Quality Standards
- Black formatting (88 character line limit)
- Ruff linting with import sorting
- MyPy type checking (strict mode enabled)
- Pre-commit hooks for automated quality checks
- Pytest with coverage reporting (target: krx_portfolio module)