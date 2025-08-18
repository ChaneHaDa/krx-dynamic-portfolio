# Portfolio Optimization API Documentation

> 포트폴리오 최적화 모듈의 상세 API 문서

## 📋 목차

1. [개요](#개요)
2. [MPT 최적화 모듈](#mpt-최적화-모듈)
3. [리스크 모델링 모듈](#리스크-모델링-모듈)
4. [동적 리밸런싱 모듈](#동적-리밸런싱-모듈)
5. [통합 파이프라인](#통합-파이프라인)
6. [사용 예제](#사용-예제)
7. [설정 가이드](#설정-가이드)

## 개요

포트폴리오 최적화 모듈은 Modern Portfolio Theory를 기반으로 한 3개의 핵심 컴포넌트와 통합 파이프라인으로 구성됩니다:

- **MPT 최적화**: Max Sharpe, Min Variance, Mean-Variance 목적함수
- **리스크 모델링**: 공분산 추정, PSD 보정, 리스크 버짓팅
- **동적 리밸런싱**: 스케줄/임계치 기반 거래비용 최적화
- **통합 파이프라인**: End-to-end 최적화 워크플로우

## MPT 최적화 모듈

### `krx_portfolio.models.mpt.MPTOptimizer`

Modern Portfolio Theory 기반 포트폴리오 최적화 클래스입니다.

#### 초기화

```python
from krx_portfolio.models.mpt import MPTOptimizer

optimizer = MPTOptimizer(
    bounds=(0.0, 0.1),          # 개별 자산 비중 범위 [하한, 상한]
    rf=0.0,                     # 무위험 수익률
    sector_caps=None,           # 섹터별 비중 상한 {'IT': 0.4, 'Finance': 0.3}
    turnover_budget=None,       # 턴오버 예산 (포트폴리오 대비 비율)
    penalty=None                # 턴오버 패널티 ('l1', 'l2')
)
```

#### 주요 메서드

##### `max_sharpe(mu, Sigma, w_prev=None)`

샤프 비율을 최대화하는 포트폴리오를 구합니다.

**Parameters:**
- `mu` (np.ndarray): 기대수익률 벡터 (N,)
- `Sigma` (np.ndarray): 공분산 행렬 (N, N)
- `w_prev` (np.ndarray, optional): 이전 포트폴리오 비중 (턴오버 패널티용)

**Returns:**
- `np.ndarray`: 최적 포트폴리오 비중 (N,)

**Example:**
```python
import numpy as np

# 샘플 데이터
mu = np.array([0.08, 0.10, 0.12, 0.09])  # 기대수익률
Sigma = np.array([  # 공분산 행렬
    [0.04, 0.01, 0.02, 0.01],
    [0.01, 0.06, 0.01, 0.02],
    [0.02, 0.01, 0.08, 0.02],
    [0.01, 0.02, 0.02, 0.05]
])

# 최적화
weights = optimizer.max_sharpe(mu, Sigma)
print(f"Optimal weights: {weights}")
```

##### `min_variance(mu, Sigma, w_prev=None)`

포트폴리오 변동성을 최소화합니다.

##### `mean_variance(mu, Sigma, risk_aversion=None, target_return=None, w_prev=None)`

평균-분산 최적화를 수행합니다.

**Parameters:**
- `risk_aversion` (float): 위험 회피 계수 λ ∈ [0, 1]
- `target_return` (float): 목표 수익률 (risk_aversion 대신 사용 가능)

##### `_calculate_portfolio_stats(w, mu, Sigma)`

포트폴리오 통계를 계산합니다.

**Returns:**
```python
{
    'return': float,      # 포트폴리오 기대수익률
    'volatility': float,  # 포트폴리오 변동성
    'sharpe': float       # 샤프 비율
}
```

## 리스크 모델링 모듈

### `krx_portfolio.models.risk.RiskModel`

다양한 공분산 추정 방법을 제공하는 리스크 모델링 클래스입니다.

#### 초기화

```python
from krx_portfolio.models.risk import RiskModel

risk_model = RiskModel(
    method="ledoit_wolf",       # 공분산 추정 방법
    ewma_lambda=0.94,          # EWMA 감쇠 계수
    ridge=1e-6,                # Ridge 정규화 계수
    factor_model=None,         # 팩터 모델 타입
    min_periods=252            # 최소 필요 기간
)
```

#### 지원하는 추정 방법

- `"sample"`: 표본 공분산
- `"ledoit_wolf"`: Ledoit-Wolf 수축 추정
- `"oas"`: Oracle Approximating Shrinkage
- `"ewma"`: 지수가중이동평균

#### 주요 메서드

##### `fit(returns)`

수익률 데이터에 리스크 모델을 적합시킵니다.

**Parameters:**
- `returns` (pd.DataFrame): 수익률 데이터 (T×N)

**Returns:**
- `RiskModel`: 메서드 체이닝을 위한 self

##### `cov()`

공분산 행렬을 반환합니다.

**Returns:**
- `np.ndarray`: 공분산 행렬 (N×N)

##### `corr()`

상관계수 행렬을 반환합니다.

##### `vol()`

자산별 변동성을 반환합니다.

**Returns:**
- `pd.Series`: 자산별 변동성

##### `nearest_psd(matrix, eps=1e-8)`

최근접 양정부호 행렬을 구합니다.

##### `risk_budget(weights, risk_budgets=None)`

리스크 버짓팅 분석을 수행합니다.

**Example:**
```python
import pandas as pd
import numpy as np

# 샘플 수익률 데이터 생성
dates = pd.date_range('2020-01-01', periods=252, freq='D')
returns = pd.DataFrame(
    np.random.randn(252, 4) * 0.02,
    index=dates,
    columns=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
)

# 리스크 모델 적합
risk_model = RiskModel(method='ledoit_wolf')
risk_model.fit(returns)

# 공분산 행렬 및 변동성 확인
cov_matrix = risk_model.cov()
volatilities = risk_model.vol()
print(f"Portfolio volatilities: {volatilities}")
```

## 동적 리밸런싱 모듈

### `krx_portfolio.models.rebalance.Rebalancer`

비용을 고려한 동적 리밸런싱 클래스입니다.

#### 초기화

```python
from krx_portfolio.models.rebalance import Rebalancer

rebalancer = Rebalancer(
    schedule="month_end",       # 리밸런싱 스케줄
    turnover_budget=0.2,       # 턴오버 예산
    rebalance_threshold=0.05,  # 리밸런싱 임계치
    tc_bps=25.0,              # 거래비용 (bp)
    min_trade_size=0.001,     # 최소 거래 크기
    max_positions=None        # 최대 포지션 수
)
```

#### 리밸런싱 스케줄

- `"month_end"`: 월말
- `"quarter_end"`: 분기말  
- `"weekly"`: 주간 (금요일)
- `"daily"`: 매일

#### 주요 메서드

##### `next_rebalance_dates(dates, start_date=None, end_date=None)`

리밸런싱 일정을 생성합니다.

**Parameters:**
- `dates` (pd.DatetimeIndex): 영업일 캘린더
- `start_date`, `end_date` (pd.Timestamp): 시작/종료 일자

**Returns:**
- `List[pd.Timestamp]`: 리밸런싱 일자 목록

##### `should_rebalance(w_target, w_current, date, rebalance_dates)`

리밸런싱 실행 여부를 판단합니다.

##### `apply(w_target, w_current, prices, portfolio_value=1.0, asset_names=None)`

리밸런싱을 실행합니다.

**Returns:**
```python
{
    'w_executed': np.ndarray,    # 최종 실행 비중
    'w_diff': np.ndarray,        # 비중 변화량
    'orders': pd.DataFrame,      # 주문 내역
    'turnover': float,           # 총 턴오버
    'tc_cost': float,           # 거래비용
    'portfolio_value': float    # 포트폴리오 가치
}
```

**Example:**
```python
import pandas as pd

# 영업일 캘린더 생성
business_dates = pd.bdate_range('2023-01-01', '2023-12-31')

# 리밸런싱 일정 생성
rebalance_dates = rebalancer.next_rebalance_dates(business_dates)
print(f"Rebalance dates: {rebalance_dates[:5]}")  # 처음 5개 날짜

# 리밸런싱 실행
w_target = np.array([0.3, 0.2, 0.3, 0.2])
w_current = np.array([0.25, 0.25, 0.25, 0.25])
prices = pd.Series([100, 200, 150, 300], index=['A', 'B', 'C', 'D'])

result = rebalancer.apply(w_target, w_current, prices, portfolio_value=1000000)
print(f"Turnover: {result['turnover']:.3f}")
print(f"Transaction cost: {result['tc_cost']:.3f}")
```

## 통합 파이프라인

### `krx_portfolio.models.pipeline.PortfolioOptimizationPipeline`

End-to-end 포트폴리오 최적화 파이프라인입니다.

#### 초기화

```python
from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline

# 설정 파일로 초기화
pipeline = PortfolioOptimizationPipeline(config_path='configs/portfolio.yaml')

# 또는 설정 딕셔너리로 초기화
config = {
    'objective': 'max_sharpe',
    'risk_model': {'method': 'ledoit_wolf'},
    'constraints': {'w_bounds': [0.0, 0.1]}
}
pipeline = PortfolioOptimizationPipeline(config=config)
```

#### 주요 메서드

##### `build_weights(mu, returns, sector_map=None, prices=None, current_weights=None, date=None)`

포트폴리오 가중치를 생성합니다.

**Parameters:**
- `mu` (np.ndarray): 기대수익률 벡터
- `returns` (pd.DataFrame): 과거 수익률 데이터
- `sector_map` (dict): 섹터 매핑 {'asset': 'sector'}
- `prices` (pd.Series): 현재 가격 (리밸런싱용)
- `current_weights` (np.ndarray): 현재 포트폴리오 비중
- `date` (pd.Timestamp): 현재 날짜

**Returns:**
```python
{
    'target_weights': np.ndarray,     # 목표 비중
    'w_executed': np.ndarray,         # 실행 비중
    'risk_metrics': dict,             # 리스크 지표
    'rebalanced': bool,               # 리밸런싱 여부
    'turnover': float,                # 턴오버 (리밸런싱 시)
    'tc_cost': float,                # 거래비용 (리밸런싱 시)
    'sector_analysis': dict           # 섹터 분석 (제공시)
}
```

##### `generate_weight_series(returns, expected_returns, rebalance_dates=None, initial_weights=None, prices=None)`

가중치 시계열을 생성합니다.

##### `save_results(results, output_path)`

최적화 결과를 저장합니다.

### 편의 함수

#### `build_weights(mu, returns, sector_map=None, config=None)`

단발성 포트폴리오 최적화 편의 함수입니다.

#### `create_monthly_weights(returns_data, expected_returns, config=None, output_dir=None)`

월별 포트폴리오 가중치 생성 편의 함수입니다.

## 사용 예제

### 기본 사용법

```python
import numpy as np
import pandas as pd
from krx_portfolio.models.pipeline import build_weights

# 샘플 데이터 준비
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='B')
assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

# 수익률 데이터 생성
returns = pd.DataFrame(
    np.random.randn(252, 4) * 0.015,
    index=dates, 
    columns=assets
)

# 기대수익률
mu = np.array([0.08, 0.10, 0.09, 0.12]) / 252  # 일간 수익률로 변환

# 포트폴리오 최적화
results = build_weights(mu=mu, returns=returns)

print(f"Optimal weights: {results['target_weights']}")
print(f"Expected return: {results['risk_metrics']['expected_return']:.4f}")
print(f"Volatility: {results['risk_metrics']['volatility']:.4f}")
print(f"Sharpe ratio: {results['risk_metrics']['sharpe_ratio']:.4f}")
```

### 고급 사용법 (리밸런싱 포함)

```python
from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline

# 고급 설정
config = {
    'objective': 'max_sharpe',
    'risk_model': {
        'method': 'ledoit_wolf',
        'ridge': 1e-6
    },
    'constraints': {
        'w_bounds': [0.0, 0.25],
        'sector_caps': {
            'Technology': 0.6,
            'Healthcare': 0.4
        }
    },
    'rebalance': {
        'schedule': 'month_end',
        'threshold': 0.05,
        'tc_bps': 25
    }
}

# 파이프라인 생성
pipeline = PortfolioOptimizationPipeline(config=config)

# 현재 포트폴리오와 가격 정보
current_weights = np.array([0.25, 0.25, 0.25, 0.25])
prices = pd.Series([150, 2800, 350, 200], index=assets)
sector_map = {
    'AAPL': 'Technology', 'GOOGL': 'Technology',
    'MSFT': 'Technology', 'TSLA': 'Healthcare'
}

# 리밸런싱 포함 최적화
results = pipeline.build_weights(
    mu=mu,
    returns=returns,
    sector_map=sector_map,
    prices=prices,
    current_weights=current_weights,
    date=pd.Timestamp('2023-06-30')  # 월말
)

if results['rebalanced']:
    print(f"Rebalancing executed!")
    print(f"Turnover: {results['turnover']:.3f}")
    print(f"Transaction cost: {results['tc_cost']:.4f}")
    print(f"Orders:\n{results['orders']}")
else:
    print("No rebalancing needed")

print(f"Sector allocation: {results['sector_analysis']}")
```

### 월별 포트폴리오 생성

```python
from krx_portfolio.models.pipeline import create_monthly_weights

# 월별 기대수익률 데이터 (예시)
monthly_dates = pd.date_range('2023-01-31', '2023-12-31', freq='M')
expected_returns = pd.DataFrame(
    np.tile(mu, (len(monthly_dates), 1)),
    index=monthly_dates,
    columns=assets
)

# 월별 포트폴리오 가중치 생성
weight_series = create_monthly_weights(
    returns_data=returns,
    expected_returns=expected_returns,
    config=config,
    output_dir='data/portfolios'
)

print(f"Generated {len(weight_series)} monthly portfolios")
print(weight_series.head())
```

## 설정 가이드

### 설정 파일 구조 (`configs/portfolio.yaml`)

```yaml
# 최적화 목적함수
objective: "max_sharpe"  # "min_variance", "mean_variance"

# 무위험 수익률
risk_free_rate: 0.0

# 리스크 모델 설정
risk_model:
  method: "ledoit_wolf"  # "sample", "oas", "ewma"
  ewma_lambda: 0.94     # EWMA 전용
  ridge: 1.0e-6         # 정규화 계수

# 포트폴리오 제약조건
constraints:
  long_only: true
  w_bounds: [0.0, 0.1]  # 개별 자산 범위
  sector_caps:          # 섹터별 상한
    "Technology": 0.6
    "Healthcare": 0.4
  min_positions: 10

# 리밸런싱 설정
rebalance:
  schedule: "month_end"    # "quarter_end", "weekly", "daily"
  threshold: 0.05          # L1 거리 임계치
  turnover_budget: 0.25    # 월간 최대 턴오버
  tc_bps: 25              # 거래비용 (bp)

# 변동성 타겟팅 (선택사항)
vol_target:
  enable: false
  annual_vol: 0.10
```

### 주요 매개변수 가이드

#### 최적화 목적함수
- `max_sharpe`: 샤프 비율 최대화 (일반적 선택)
- `min_variance`: 변동성 최소화 (보수적 전략)
- `mean_variance`: 수익-위험 균형 (risk_aversion 설정 필요)

#### 리스크 모델
- `ledoit_wolf`: 수축 추정, 안정적 (권장)
- `sample`: 표본 공분산, 단순
- `ewma`: 최근 데이터 가중, lambda=0.94 권장

#### 제약조건
- `w_bounds`: [0, 0.1] = 롱온리, 최대 10%
- `sector_caps`: 섹터 집중 방지
- `min_positions`: 최소 분산 요구사항

#### 리밸런싱
- `threshold`: 0.05 = 5% L1 거리시 리밸런싱
- `turnover_budget`: 0.25 = 월 25% 이내 거래
- `tc_bps`: 25 = 왕복 0.25% 거래비용

## 에러 처리 및 디버깅

### 일반적인 에러

1. **수렴 실패**: ridge 값 증가 (1e-6 → 1e-4)
2. **PSD 위반**: `nearest_psd()` 자동 보정
3. **제약조건 충돌**: bounds와 sector_caps 확인
4. **데이터 부족**: min_periods 조정

### 로깅 활성화

```python
import logging
logging.basicConfig(level=logging.INFO)

# 파이프라인 실행시 상세 로그 확인
results = pipeline.build_weights(mu, returns)
```

### 성능 최적화

```python
# 대용량 데이터 처리시
config['advanced'] = {
    'performance': {
        'parallel': True,
        'n_jobs': -1
    }
}
```

## 참고자료

- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Ledoit-Wolf Shrinkage](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance)
- [Transaction Cost Analysis](https://www.investopedia.com/terms/t/transactioncosts.asp)
- [Risk Budgeting](https://www.investopedia.com/terms/r/risk-budget.asp)