# Portfolio Optimization Module Guide

## Overview

KRX Dynamic Portfolio의 포트폴리오 최적화 모듈은 Modern Portfolio Theory(MPT)를 기반으로 한 완전한 최적화 파이프라인을 제공합니다. 이 모듈은 리스크 모델링, 포트폴리오 최적화, 그리고 동적 리밸런싱의 통합된 워크플로우를 구현합니다.

## 주요 구성 요소

### 1. MPT Optimizer (`krx_portfolio.models.mpt.MPTOptimizer`)

Modern Portfolio Theory 기반의 포트폴리오 최적화를 수행합니다.

#### 지원하는 최적화 목표

- **Max Sharpe Ratio**: 샤프 비율 최대화
- **Min Variance**: 포트폴리오 분산 최소화  
- **Mean-Variance**: 위험회피 파라미터 또는 목표 수익률 기반 최적화

#### 기본 사용법

```python
from krx_portfolio.models.mpt import MPTOptimizer
import numpy as np

# 최적화기 초기화
optimizer = MPTOptimizer(
    bounds=(0.0, 0.1),  # 개별 자산 가중치 제한
    rf=0.02,            # 무위험 수익률
    turnover_budget=0.2, # 회전율 예산
    penalty='l1'        # 회전율 페널티 유형
)

# 예상 수익률과 공분산 행렬
mu = np.array([0.08, 0.10, 0.12, 0.09, 0.11])  # 5개 자산
sigma = np.eye(5) * 0.04  # 공분산 행렬

# 1. 샤프 비율 최대화
weights_sharpe = optimizer.max_sharpe(mu, sigma)

# 2. 분산 최소화
weights_minvar = optimizer.min_variance(mu, sigma)

# 3. 평균-분산 최적화
weights_meanvar = optimizer.mean_variance(
    mu, sigma, 
    risk_aversion=0.5  # 위험회피 계수
)
```

#### 제약 조건

- **Position Bounds**: 개별 자산의 최소/최대 가중치 제한
- **Sector Caps**: 섹터별 가중치 상한선
- **Turnover Budget**: 회전율 예산 제약
- **Long-Only**: 매수만 가능 (공매도 금지)

### 2. Risk Model (`krx_portfolio.models.risk.RiskModel`)

다양한 공분산 추정 방법을 지원하는 리스크 모델링 모듈입니다.

#### 지원하는 추정 방법

- **Sample Covariance**: 표본 공분산
- **Ledoit-Wolf Shrinkage**: 수축 추정
- **Oracle Approximating Shrinkage (OAS)**: 오라클 근사 수축
- **EWMA**: 지수가중이동평균

#### 기본 사용법

```python
from krx_portfolio.models.risk import RiskModel
import pandas as pd
import numpy as np

# 수익률 데이터 준비
np.random.seed(42)
returns = pd.DataFrame(
    np.random.randn(252, 5) * 0.02,
    columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D', 'Asset_E'],
    index=pd.date_range('2023-01-01', periods=252, freq='D')
)

# 리스크 모델 생성 및 적합
risk_model = RiskModel(
    method='ledoit_wolf',  # 추정 방법
    ridge=1e-6,           # 릿지 정규화
    min_periods=30        # 최소 관측치
)

risk_model.fit(returns)

# 공분산 행렬 추출
cov_matrix = risk_model.cov()
corr_matrix = risk_model.corr()
volatilities = risk_model.vol()

print(f"공분산 행렬 크기: {cov_matrix.shape}")
print(f"변동성:\n{volatilities}")
```

#### 고급 기능

##### 1. 리스크 버짓팅

```python
# 포트폴리오 가중치
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# 리스크 기여도 분석
risk_budget = risk_model.risk_budget(weights)

print("리스크 기여도:")
for key, value in risk_budget.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value}")
```

##### 2. 팩터 노출도 분석

```python
# 팩터 데이터 (예: 시장 팩터, 산업 팩터)
factors = pd.DataFrame(
    np.random.randn(252, 3) * 0.01,
    columns=['Market', 'Value', 'Growth'],
    index=returns.index
)

# 팩터 노출도 분석
factor_results = risk_model.factor_exposure(factors, method='ols')

print("팩터 로딩:")
print(factor_results['factor_loadings'])
print("\n특이 리스크:")
print(factor_results['specific_risk'])
```

### 3. Dynamic Rebalancer (`krx_portfolio.models.rebalance.Rebalancer`)

거래비용을 고려한 동적 리밸런싱 모듈입니다.

#### 기본 사용법

```python
from krx_portfolio.models.rebalance import Rebalancer
import pandas as pd
import numpy as np

# 리밸런서 초기화
rebalancer = Rebalancer(
    schedule='month_end',      # 리밸런싱 일정
    turnover_budget=0.2,       # 회전율 예산
    rebalance_threshold=0.05,  # 리밸런싱 임계치
    tc_bps=25.0,              # 거래비용 (bp)
    min_trade_size=0.001       # 최소 거래 크기
)

# 거래일 생성
dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')

# 리밸런싱 날짜 생성
rebalance_dates = rebalancer.next_rebalance_dates(dates)
print(f"리밸런싱 날짜 수: {len(rebalance_dates)}")

# 리밸런싱 실행
target_weights = np.array([0.25, 0.25, 0.25, 0.25])
current_weights = np.array([0.3, 0.2, 0.3, 0.2])
prices = pd.Series([100, 50, 75, 120], name='2023-01-31')

rebalance_result = rebalancer.apply(
    target_weights, current_weights, prices
)

print("리밸런싱 결과:")
print(f"회전율: {rebalance_result['turnover']:.4f}")
print(f"거래비용: {rebalance_result['tc_cost']:.4f}")
print("주문 내역:")
print(rebalance_result['orders'])
```

### 4. Portfolio Pipeline (`krx_portfolio.models.pipeline.PortfolioOptimizationPipeline`)

통합 포트폴리오 최적화 파이프라인입니다.

#### 기본 사용법

```python
from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline
import numpy as np
import pandas as np

# 설정
config = {
    'objective': 'max_sharpe',
    'risk_free_rate': 0.02,
    'risk_model': {
        'method': 'ledoit_wolf',
        'ewma_lambda': 0.94
    },
    'constraints': {
        'w_bounds': [0.0, 0.15],
        'turnover_budget': 0.25
    },
    'rebalance': {
        'schedule': 'month_end',
        'tc_bps': 30.0
    }
}

# 파이프라인 초기화
pipeline = PortfolioOptimizationPipeline(config=config)

# 데이터 준비
returns = pd.DataFrame(np.random.randn(252, 10) * 0.02)
mu = np.random.randn(10) * 0.08
prices = pd.Series(np.random.uniform(50, 150, 10))

# 포트폴리오 최적화 실행
results = pipeline.build_weights(
    mu=mu,
    returns=returns,
    prices=prices,
    date=pd.Timestamp('2023-12-31')
)

print("최적화 결과:")
print(f"목표 가중치: {results['target_weights']}")
print(f"실행 가중치: {results['w_executed']}")
print(f"리스크 지표: {results['risk_metrics']}")
```

#### 시계열 가중치 생성

```python
# 월말 리밸런싱 가중치 시계열 생성
expected_returns = pd.DataFrame(
    np.random.randn(12, 10) * 0.01,
    index=pd.date_range('2023-01-31', '2023-12-31', freq='M')
)

weight_series = pipeline.generate_weight_series(
    returns=returns,
    expected_returns=expected_returns
)

print("월별 포트폴리오 가중치:")
print(weight_series.head())
```

## 설정 파일 (portfolio.yaml)

포트폴리오 최적화 설정은 YAML 파일로 관리할 수 있습니다.

```yaml
# 최적화 목표
objective: "max_sharpe"  # max_sharpe, min_variance, mean_variance

# 무위험 수익률
risk_free_rate: 0.02

# 리스크 모델 설정
risk_model:
  method: "ledoit_wolf"  # sample, ledoit_wolf, oas, ewma
  ewma_lambda: 0.94
  ridge: 1e-6
  min_periods: 252

# 제약 조건
constraints:
  long_only: true
  w_bounds: [0.0, 0.15]  # 개별 자산 가중치 제한
  turnover_budget: 0.25   # 회전율 예산
  turnover_penalty: "l1"  # 회전율 페널티
  sector_caps:            # 섹터별 상한
    "Technology": 0.30
    "Healthcare": 0.25
    "Finance": 0.20

# 리밸런싱 설정
rebalance:
  schedule: "month_end"     # month_end, quarter_end, weekly
  threshold: 0.05          # 리밸런싱 임계치
  turnover_budget: 0.25    # 회전율 예산
  tc_bps: 25.0            # 거래비용 (basis points)
  min_trade_size: 0.001   # 최소 거래 크기

# 기타 설정
vol_target:
  enable: false
  annual_vol: 0.12

lookback_window: 252  # 리스크 모델 룩백 윈도우
```

## 사용 예제

### 1. 기본 최적화

```python
from krx_portfolio.models.pipeline import build_weights
import numpy as np
import pandas as pd

# 데이터 준비
returns = pd.DataFrame(np.random.randn(252, 5) * 0.02)
mu = np.array([0.08, 0.10, 0.12, 0.09, 0.11])

# 간단한 최적화
results = build_weights(mu, returns)
print(f"최적 가중치: {results['target_weights']}")
```

### 2. 커스텀 설정 최적화

```python
config = {
    'objective': 'mean_variance',
    'risk_aversion': 0.7,
    'constraints': {
        'w_bounds': [0.05, 0.25],
        'turnover_budget': 0.15
    }
}

results = build_weights(mu, returns, config=config)
```

### 3. 섹터 제약 최적화

```python
sector_map = {
    'Asset_A': 'Technology',
    'Asset_B': 'Healthcare', 
    'Asset_C': 'Finance',
    'Asset_D': 'Technology',
    'Asset_E': 'Healthcare'
}

config = {
    'constraints': {
        'sector_caps': {
            'Technology': 0.35,
            'Healthcare': 0.30,
            'Finance': 0.25
        }
    }
}

results = build_weights(mu, returns, sector_map, config)
print("섹터 분석:")
print(results['sector_analysis'])
```

## 성능 지표

최적화 결과는 다음과 같은 리스크 지표를 포함합니다:

- **Expected Return**: 기대 수익률
- **Volatility**: 포트폴리오 변동성  
- **Sharpe Ratio**: 샤프 비율
- **Max Weight**: 최대 개별 가중치
- **Min Weight**: 최소 개별 가중치
- **Effective Positions**: 유효 포지션 수

## 고급 기능

### 1. 거래비용 최적화

리밸런싱 시 거래비용을 고려한 최적화가 자동으로 수행됩니다:

- 최소 거래 크기 필터링
- 회전율 예산 제약
- 포지션 수 제한
- 비례 거래비용 계산

### 2. 리스크 관리

- PSD(Positive Semi-Definite) 행렬 보정
- 릿지 정규화를 통한 수치적 안정성
- 아웃라이어 처리
- 최소 관측치 요구사항

### 3. 백테스팅 준비

생성된 가중치 시계열은 백테스팅 모듈과 직접 연동됩니다:

```python
# 월별 가중치 저장
from krx_portfolio.models.pipeline import create_monthly_weights

monthly_weights = create_monthly_weights(
    returns_data=returns,
    expected_returns=expected_returns,
    config=config,
    output_dir='./portfolio_weights'
)
```

이 가중치들은 백테스팅 엔진에서 성과 평가에 사용됩니다.