# Repository Guidelines

This repository contains a Python package for KRX dynamic portfolio ETL, optimization, backtesting, and a Streamlit dashboard.

## Project Structure & Modules
- `krx_portfolio/`: Main package
  - `etl/`, `models/`, `backtesting/`, `app/`, `utils.py`
- `tests/`: Pytest suite (`test_*.py`, classes `Test*`, functions `test_*`)
- `configs/portfolio.yaml`: Portfolio/config parameters
- `docs/`: Architecture, API, troubleshooting, performance guides
- `Makefile`: Common developer commands

## Build, Test, and Development Commands
- `make install`: Editable install of the package
- `make dev`: Install dev deps and set up pre-commit
- `make lint`: Run Ruff, Black check, and MyPy
- `make format`: Apply Black and Ruff fixes
- `make test`: Run pytest with coverage
- `make etl` | `make train` | `make backtest`: Run pipelines
- `make app`: Launch Streamlit dashboard (`krx_portfolio/app/dashboard.py`)
- `make ci-test`: Lint + tests locally (CI parity)
- `make clean`: Remove build, caches, and coverage artifacts

## Coding Style & Naming Conventions
- Formatting: Black (88 cols); Ruff rules `E,F,I,N,W,UP`; imports sorted (first-party: `krx_portfolio`)
- Types: MyPy strict settings; add annotations for new/changed code
- Python style: 4-space indent; functions/modules `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`
- Keep modules focused; prefer small, testable functions; document public APIs briefly

## Testing Guidelines
- Framework: Pytest with coverage on `krx_portfolio`
- Layout: put tests in `tests/` following `test_*.py`
- Aim to cover new logic; write unit tests near affected modules
- Mock external calls (e.g., `yfinance`, I/O); keep tests deterministic (seed RNG)
- Run `make test` and ensure no warnings/errors

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
  - Example: `feat: backtesting engine adds rolling Sharpe`
- PRs: concise description, scope, linked issues; screenshots/GIFs for dashboard changes
- Before opening: `make ci-test`, update `docs/` if behavior/CLI changes, note config impacts

## Security & Configuration
- Do not commit credentials or private data; configure access via environment variables
- Update `configs/portfolio.yaml` for tunables; avoid hardcoding paths/keys
- When adding dependencies, update `pyproject.toml` and run `make dev`
