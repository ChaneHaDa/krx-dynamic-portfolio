# KRX Dynamic Portfolio - 문서 디렉토리

이 디렉토리는 KRX Dynamic Portfolio 시스템의 완전한 문서화를 제공합니다.

## 📖 문서 구조

### API 및 사용법 가이드

#### 1. **ETL 모듈 문서**
- [`etl_api.md`](./etl_api.md) - ETL 파이프라인 API 레퍼런스
- [`etl_usage_guide.md`](./etl_usage_guide.md) - ETL 사용법 가이드

#### 2. **포트폴리오 최적화 문서**
- [`portfolio_api.md`](./portfolio_api.md) - 포트폴리오 최적화 API 레퍼런스  
- [`portfolio_optimization_guide.md`](./portfolio_optimization_guide.md) - 포트폴리오 최적화 완전 가이드 (165줄)

#### 3. **백테스팅 모듈 문서**
- [`backtesting_guide.md`](./backtesting_guide.md) - 백테스팅 완전 가이드 (400+ 줄)

## 🚀 빠른 시작 가이드

### 데이터 처리부터 백테스팅까지

1. **데이터 수집 및 전처리**
   ```bash
   # ETL 파이프라인 실행
   python -m krx_portfolio.etl.main --data-root /path/to/krx/data
   ```
   📖 상세 가이드: [`etl_usage_guide.md`](./etl_usage_guide.md)

2. **포트폴리오 최적화**
   ```python
   from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline
   
   pipeline = PortfolioOptimizationPipeline()
   results = pipeline.build_weights(mu, returns)
   ```
   📖 상세 가이드: [`portfolio_optimization_guide.md`](./portfolio_optimization_guide.md)

3. **백테스팅 실행**
   ```bash
   # 완전한 백테스팅 파이프라인
   python -m krx_portfolio.backtesting.main --data-root /path/to/data --output-dir ./results
   ```
   📖 상세 가이드: [`backtesting_guide.md`](./backtesting_guide.md)

## 📊 구현 완료 현황

### ✅ Phase 1: ETL 파이프라인 
- **구현 완료**: 100% (39개 테스트, 95% 커버리지)
- **문서화**: API 레퍼런스 + 사용법 가이드

### ✅ Phase 2: 포트폴리오 최적화
- **구현 완료**: 100% (83개 테스트)
- **핵심 모듈**: MPT, Risk Model, Rebalancer, Pipeline
- **문서화**: 165줄 완전 구현 가이드

### ✅ Phase 3: 백테스팅 엔진
- **구현 완료**: 100% (110+ 테스트, 4,700+ 줄)
- **핵심 모듈**: BacktestEngine, PerformanceMetrics, RiskAnalytics, Pipeline
- **문서화**: 400+ 줄 실무급 백테스팅 가이드

### 📋 Phase 4: 사용자 인터페이스 (예정)
- Streamlit 대시보드
- 인터랙티브 시각화
- 실시간 모니터링

## 🔧 개발자 가이드

### 코드 품질 및 테스팅
- **린팅**: `make lint` (ruff + black + mypy)
- **테스트**: `make test` (230+ 테스트)
- **CI/CD**: GitHub Actions 자동화

### 프로젝트 설정
- **환경 설정**: `make dev`
- **의존성 관리**: `pyproject.toml`
- **설정 파일**: `configs/portfolio.yaml`

## 📈 문서 활용 가이드

### 초보자를 위한 학습 경로

1. **기초 이해**
   - [`etl_usage_guide.md`](./etl_usage_guide.md)에서 데이터 처리 과정 학습
   - 샘플 데이터로 ETL 파이프라인 실행해보기

2. **중급 활용**
   - [`portfolio_optimization_guide.md`](./portfolio_optimization_guide.md)에서 최적화 알고리즘 학습
   - 다양한 최적화 전략 실험해보기

3. **고급 분석**
   - [`backtesting_guide.md`](./backtesting_guide.md)에서 백테스팅 마스터하기
   - VaR, CVaR, 극값분석 등 고급 리스크 분석 활용

### 실무자를 위한 활용 가이드

1. **전략 개발**
   - 커스텀 최적화 목표 함수 구현
   - 섹터 제약, 회전율 제한 등 실무 제약조건 적용

2. **리스크 관리**
   - 스트레스 테스팅 및 시나리오 분석
   - 다중 VaR 모델을 통한 리스크 측정

3. **성과 평가**
   - 40+ 성과지표를 통한 종합 분석
   - 벤치마크 대비 성과 평가 및 기여도 분석

## 🎯 고급 활용 사례

### 1. 다중 전략 비교
```python
# 여러 최적화 전략 동시 백테스팅
strategies = ["max_sharpe", "min_variance", "risk_parity"]
for strategy in strategies:
    results = run_backtest_with_strategy(strategy)
    compare_performance(results)
```

### 2. 실시간 포트폴리오 모니터링
```python
# 롤링 백테스팅으로 실시간 성과 추적
rolling_backtest = setup_rolling_monitor()
rolling_backtest.start_monitoring()
```

### 3. 커스텀 리스크 지표 개발
```python
# 사용자 정의 리스크 지표 구현
class CustomRiskMetrics(RiskAnalytics):
    def calculate_custom_var(self, returns):
        # 커스텀 VaR 구현
        pass
```

## 📞 지원 및 기여

### 문서 개선
- 문서에 오류나 개선사항이 있으면 이슈를 생성해주세요
- 새로운 사용 사례나 예제 추가를 환영합니다

### 코드 기여
- 새로운 기능 개발 전 이슈로 논의해주세요
- 모든 기여는 테스트 코드와 문서화를 포함해야 합니다

---

**⚠️ 투자 주의사항**: 이 시스템은 교육 및 연구 목적으로 개발되었습니다. 실제 투자 결정에 사용하기 전 충분한 검증과 전문가 자문이 필요합니다.