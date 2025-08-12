.PHONY: help install dev lint format test clean etl train backtest app

help:  ## 사용 가능한 명령어들을 보여줍니다
	@echo "KRX Dynamic Portfolio 개발 명령어들:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## 의존성을 설치합니다
	pip install -e .

dev:  ## 개발 의존성을 설치합니다
	pip install -e ".[dev]"
	pre-commit install

lint:  ## 코드 스타일을 검사합니다
	ruff check .
	black --check .
	mypy .

format:  ## 코드를 포맷팅합니다
	black .
	ruff check --fix .

test:  ## 테스트를 실행합니다
	pytest -v --cov=krx_portfolio --cov-report=term-missing

clean:  ## 빌드 파일들을 정리합니다
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# =============================================================================
# 데이터 파이프라인 명령어들
# =============================================================================

etl:  ## KRX 데이터를 수집하고 전처리합니다
	python -m krx_portfolio.etl.main

train:  ## 포트폴리오 최적화 모델을 학습합니다
	python -m krx_portfolio.models.train

backtest:  ## 백테스트를 실행합니다
	python -m krx_portfolio.backtesting.main

app:  ## Streamlit 대시보드를 실행합니다
	streamlit run krx_portfolio/app/dashboard.py

# =============================================================================
# CI/CD 관련 명령어들
# =============================================================================

ci-test: lint test  ## CI에서 실행되는 모든 검사를 로컬에서 실행합니다

pre-commit:  ## pre-commit 훅을 수동 실행합니다
	pre-commit run --all-files

# =============================================================================
# 개발 편의 명령어들
# =============================================================================

jupyter:  ## Jupyter Lab을 실행합니다
	jupyter lab

requirements:  ## requirements.txt를 생성합니다 (배포용)
	pip freeze > requirements.txt

docker-build:  ## Docker 이미지를 빌드합니다
	docker build -t krx-portfolio .

docker-run:  ## Docker 컨테이너를 실행합니다
	docker run -p 8501:8501 krx-portfolio