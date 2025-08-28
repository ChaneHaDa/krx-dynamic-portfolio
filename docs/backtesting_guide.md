# Backtesting Module Guide

## Overview

KRX Dynamic Portfolio의 백테스팅 모듈은 포트폴리오 전략의 과거 성과를 실제적으로 시뮬레이션하는 완전한 백테스팅 프레임워크를 제공합니다. 이 모듈은 거래비용, 마켓 임팩트, 리밸런싱 최적화를 모두 고려한 현실적인 백테스팅 환경을 구현합니다.

## 주요 구성 요소

### 1. BacktestEngine (`krx_portfolio.backtesting.engine.BacktestEngine`)

포트폴리오 성과 시뮬레이션의 핵심 엔진입니다.

#### 주요 기능

- **포트폴리오 가치 추적**: 일별 포트폴리오 가치 및 수익률 계산
- **리밸런싱 실행**: 목표 가중치에 따른 최적 리밸런싱
- **거래비용 모델링**: 거래비용 및 마켓 임팩트 반영
- **벤치마크 비교**: 벤치마크 대비 성과 분석

#### 기본 사용법

```python
from krx_portfolio.backtesting.engine import BacktestEngine
import pandas as pd
import numpy as np

# 백테스트 엔진 초기화
engine = BacktestEngine(
    initial_capital=1_000_000,     # 초기 자본
    transaction_cost_bps=25.0,     # 거래비용 (basis points)
    market_impact_model="linear",   # 마켓 임팩트 모델
    cash_rate=0.02                 # 현금 이자율
)

# 샘플 데이터 생성
dates = pd.date_range('2023-01-01', periods=252, freq='D')
assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

# 수익률 데이터
returns = pd.DataFrame(
    np.random.normal(0.001, 0.02, (252, 4)),
    index=dates, columns=assets
)

# 포트폴리오 가중치 (월말 리밸런싱)
weight_dates = pd.date_range('2023-01-31', '2023-12-31', freq='M')
weights = pd.DataFrame(
    np.random.dirichlet([1, 1, 1, 1], len(weight_dates)),
    index=weight_dates, columns=assets
)

# 백테스팅 실행
results = engine.run_backtest(
    weights=weights,
    returns=returns,
    start_date=pd.Timestamp('2023-01-01'),
    end_date=pd.Timestamp('2023-12-31')
)

# 결과 확인
print(f"총 수익률: {results['total_return']:.2%}")
print(f"연간 수익률: {results['annualized_return']:.2%}")
print(f"변동성: {results['volatility']:.2%}")
print(f"샤프 비율: {results['sharpe_ratio']:.3f}")
print(f"최대 낙폭: {results['max_drawdown']:.2%}")
```

#### 고급 설정

```python
from krx_portfolio.models.rebalance import Rebalancer

# 커스텀 리밸런서
custom_rebalancer = Rebalancer(
    schedule="quarter_end",        # 분기말 리밸런싱
    turnover_budget=0.3,          # 회전율 예산
    rebalance_threshold=0.05,     # 리밸런싱 임계치
    tc_bps=30.0                   # 거래비용
)

# 벤치마크 수익률
benchmark_returns = returns.mean(axis=1)  # 동일가중 벤치마크

# 고급 백테스트 엔진
engine = BacktestEngine(
    initial_capital=5_000_000,
    rebalancer=custom_rebalancer,
    market_impact_model="sqrt",    # Square-root 임팩트
    benchmark_returns=benchmark_returns
)

results = engine.run_backtest(weights=weights, returns=returns)

# 벤치마크 비교 지표
print(f"초과 수익률: {results['excess_return']:.2%}")
print(f"추적 오차: {results['tracking_error']:.2%}")
print(f"정보 비율: {results['information_ratio']:.3f}")
```

### 2. PerformanceMetrics (`krx_portfolio.backtesting.metrics.PerformanceMetrics`)

포괄적인 성과 지표 계산 모듈입니다.

#### 지원하는 성과 지표

##### 기본 수익률 지표
- **총 수익률**: 전체 기간 수익률
- **연간 수익률**: 연간화된 수익률
- **산술/기하 평균**: 평균 수익률
- **최고/최악의 날**: 일별 최대/최소 수익률

##### 위험 지표
- **변동성**: 연간화된 표준편차
- **하방 변동성**: 음수 수익률만 고려한 변동성
- **준변동성**: 평균 이하 수익률의 변동성

##### 위험조정 지표
- **샤프 비율**: 위험 대비 초과 수익률
- **소르티노 비율**: 하방 위험 대비 수익률
- **칼마 비율**: 최대 낙폭 대비 수익률

#### 기본 사용법

```python
from krx_portfolio.backtesting.metrics import PerformanceMetrics
import numpy as np
import pandas as pd

# 성과 지표 계산기 초기화
metrics = PerformanceMetrics(
    risk_free_rate=0.02,      # 무위험 수익률
    confidence_level=0.05     # VaR 신뢰수준
)

# 포트폴리오 수익률 데이터
portfolio_returns = pd.Series(
    np.random.normal(0.001, 0.015, 252),
    index=pd.date_range('2023-01-01', periods=252, freq='D')
)

# 모든 지표 계산
all_metrics = metrics.calculate_all_metrics(portfolio_returns)

# 주요 지표 출력
print("=== 성과 지표 ===")
for key, value in all_metrics.items():
    if isinstance(value, (int, float)):
        if 'ratio' in key.lower() or 'alpha' in key.lower():
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.2%}")
```

#### 벤치마크 비교 분석

```python
# 벤치마크 수익률
benchmark_returns = pd.Series(
    np.random.normal(0.0008, 0.012, 252),
    index=portfolio_returns.index
)

# 포트폴리오 가치 (드로다운 분석용)
portfolio_values = (1 + portfolio_returns).cumprod() * 1_000_000

# 벤치마크 포함 종합 분석
comprehensive_metrics = metrics.calculate_all_metrics(
    returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    portfolio_values=portfolio_values
)

print("=== 벤치마크 비교 ===")
print(f"정보 비율: {comprehensive_metrics['information_ratio']:.3f}")
print(f"추적 오차: {comprehensive_metrics['tracking_error']:.2%}")
print(f"베타: {comprehensive_metrics['beta']:.3f}")
print(f"알파: {comprehensive_metrics['alpha']:.2%}")
```

#### 롤링 지표 분석

```python
# 1년 롤링 지표 계산
rolling_metrics = metrics.rolling_metrics(
    returns=portfolio_returns,
    window=252,  # 1년 윈도우
    metrics=['sharpe_ratio', 'volatility', 'max_drawdown']
)

print("=== 롤링 지표 (마지막 5일) ===")
print(rolling_metrics.tail())

# 시각화 (선택사항)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

rolling_metrics['sharpe_ratio'].plot(ax=axes[0], title='Rolling Sharpe Ratio')
rolling_metrics['volatility'].plot(ax=axes[1], title='Rolling Volatility')
rolling_metrics['max_drawdown'].plot(ax=axes[2], title='Rolling Max Drawdown')

plt.tight_layout()
plt.show()
```

### 3. RiskAnalytics (`krx_portfolio.backtesting.risk_analytics.RiskAnalytics`)

고급 리스크 분석 및 측정 모듈입니다.

#### 주요 리스크 지표

##### Value at Risk (VaR)
- **Historical VaR**: 과거 데이터 기반
- **Parametric VaR**: 정규분포 가정
- **Cornish-Fisher VaR**: 비대칭성 고려

##### 극값 분석
- **Peaks over Threshold**: 임계치 초과 분석
- **Block Maxima**: 블록 최대값 분석

#### 기본 사용법

```python
from krx_portfolio.backtesting.risk_analytics import RiskAnalytics
import numpy as np
import pandas as pd

# 리스크 분석 도구 초기화
risk_analytics = RiskAnalytics(
    confidence_levels=[0.01, 0.05, 0.10],  # VaR 신뢰수준들
    risk_free_rate=0.02
)

# 포트폴리오 수익률 (음수 편향으로 손실 시나리오 포함)
np.random.seed(42)
returns = pd.Series(
    np.random.normal(-0.0002, 0.025, 1000),  # 약간 음수 편향
    index=pd.date_range('2021-01-01', periods=1000, freq='D')
)

# VaR 계산 (모든 방법)
var_results = risk_analytics.calculate_var(
    returns, confidence_level=0.05, method='all'
)

print("=== VaR 분석 (95% 신뢰수준) ===")
for method, var_value in var_results.items():
    print(f"{method.title()} VaR: {var_value:.2%}")

# CVaR 계산
cvar_5 = risk_analytics.calculate_cvar(
    returns, confidence_level=0.05, method='historical'
)
print(f"CVaR (95%): {cvar_5:.2%}")
```

#### 극값 분석

```python
# Peaks over Threshold 분석
eva_results = risk_analytics.extreme_value_analysis(
    returns, 
    method='peaks_over_threshold',
    threshold_percentile=0.95
)

print("=== 극값 분석 ===")
print(f"임계치: {eva_results['threshold']:.2%}")
print(f"초과 관측치 수: {eva_results['n_exceedances']}")
if not np.isnan(eva_results['scale']):
    print(f"Scale 파라미터: {eva_results['scale']:.4f}")
    print(f"Shape 파라미터: {eva_results['shape']:.4f}")
```

#### 스트레스 테스팅

```python
# 포트폴리오 가중치
portfolio_weights = pd.Series([0.3, 0.25, 0.25, 0.2], 
                             index=['Tech', 'Finance', 'Healthcare', 'Energy'])

# 자산별 수익률 데이터
asset_returns = pd.DataFrame({
    'Tech': np.random.normal(0.0012, 0.03, 252),
    'Finance': np.random.normal(0.0008, 0.025, 252), 
    'Healthcare': np.random.normal(0.0006, 0.018, 252),
    'Energy': np.random.normal(0.001, 0.035, 252),
})

# 스트레스 시나리오 정의
stress_scenarios = {
    'market_crash': {
        'Tech': -0.30, 'Finance': -0.25, 
        'Healthcare': -0.15, 'Energy': -0.20
    },
    'tech_selloff': {
        'Tech': -0.40, 'Finance': -0.05, 
        'Healthcare': 0.02, 'Energy': -0.02
    },
    'oil_crisis': {
        'Tech': -0.05, 'Finance': -0.10, 
        'Healthcare': -0.02, 'Energy': -0.35
    }
}

# 스트레스 테스트 실행
stress_results = risk_analytics.stress_testing(
    portfolio_weights, asset_returns, stress_scenarios
)

print("=== 스트레스 테스트 결과 ===")
for scenario, impact in stress_results.items():
    print(f"{scenario}: {impact:.2%}")
```

#### 상관관계 분석

```python
# 자산간 상관관계 분석
correlation_analysis = risk_analytics.correlation_analysis(
    asset_returns, 
    method='pearson',
    rolling_window=60  # 2개월 롤링 윈도우
)

print("=== 상관관계 분석 ===")
print("상관관계 행렬:")
print(correlation_analysis['correlation_matrix'].round(3))
print(f"\n평균 상관관계: {correlation_analysis['average_correlation']:.3f}")

# PCA 분산 설명력
print("\nPCA 분산 설명력:")
for i, var_explained in enumerate(correlation_analysis['pca_variance_explained'][:3]):
    print(f"PC{i+1}: {var_explained:.1%}")
```

### 4. 통합 백테스팅 파이프라인 (`krx_portfolio.backtesting.main.BacktestPipeline`)

ETL → 최적화 → 백테스팅의 완전한 End-to-End 파이프라인입니다.

#### 기본 사용법

```python
from krx_portfolio.backtesting.main import BacktestPipeline

# 설정 파일 (YAML)
config = {
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
            "w_bounds": [0.0, 0.15],  # 개별 자산 15% 상한
        },
    },
    "backtest": {
        "initial_capital": 10_000_000,
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

# 파이프라인 초기화
pipeline = BacktestPipeline(config=config)

# 완전한 백테스팅 실행
results = pipeline.run_full_backtest(
    data_root="/path/to/krx/data",
    start_date="2022-01-01",
    end_date="2023-12-31",
    output_dir="./backtest_results"
)

# 결과 요약
print("=== 백테스팅 결과 요약 ===")
perf_metrics = results["performance_metrics"]["summary_metrics"]
print(f"총 수익률: {perf_metrics['total_return']:.2%}")
print(f"연간 수익률: {perf_metrics['annualized_return']:.2%}")
print(f"샤프 비율: {perf_metrics['sharpe_ratio']:.3f}")
print(f"최대 낙폭: {perf_metrics['max_drawdown']:.2%}")

# 거래비용 분석
bt_results = results["backtest_results"]
print(f"\n거래 비용: ${bt_results['total_transaction_costs']:,.0f}")
print(f"비용 비율: {bt_results['cost_ratio']:.2%}")
print(f"리밸런싱 횟수: {bt_results['num_rebalances']}")
```

#### 사전 생성된 가중치로 백테스팅

```python
# 이미 계산된 포트폴리오 가중치가 있는 경우
weights = pd.read_parquet("portfolio_weights.parquet")
returns = pd.read_parquet("asset_returns.parquet")
benchmark = pd.read_parquet("benchmark_returns.parquet")['benchmark']

# 가중치 기반 백테스팅
results = pipeline.run_backtest_with_weights(
    weights=weights,
    returns=returns,
    benchmark_returns=benchmark,
    start_date="2023-01-01",
    end_date="2023-12-31",
    output_dir="./custom_backtest"
)
```

#### 명령줄 인터페이스 (CLI)

```bash
# 명령줄에서 백테스팅 실행
python -m krx_portfolio.backtesting.main \
    --data-root /path/to/krx/data \
    --config config/backtest_config.yaml \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --output-dir ./results \
    --verbose

# 결과 요약이 자동으로 출력됩니다
```

## 설정 파일 구조 (YAML)

```yaml
# backtest_config.yaml

etl:
  cache_dir: "data/cache"
  force_reload: false

portfolio:
  objective: "max_sharpe"  # max_sharpe, min_variance, mean_variance
  risk_free_rate: 0.02
  lookback_window: 252
  
  rebalance:
    schedule: "month_end"    # month_end, quarter_end, weekly
    turnover_budget: 0.25
    tc_bps: 25.0
  
  constraints:
    w_bounds: [0.0, 0.15]    # 개별 자산 제한
    sector_caps:             # 섹터별 제한
      Technology: 0.40
      Finance: 0.30
      Healthcare: 0.25

backtest:
  initial_capital: 10_000_000
  transaction_cost_bps: 25.0
  market_impact_model: "linear"  # linear, sqrt, none
  cash_rate: 0.02

analysis:
  risk_free_rate: 0.02
  confidence_level: 0.05
  var_confidence_levels: [0.01, 0.05, 0.10]
  rolling_window: 252
```

## 결과 파일 구조

백테스팅 실행 후 다음과 같은 구조로 결과가 저장됩니다:

```
results/
├── backtest/
│   ├── portfolio_history.parquet     # 일별 포트폴리오 기록
│   ├── rebalance_history.parquet     # 리밸런싱 기록
│   └── summary_metrics.yaml          # 요약 지표
├── performance_summary.yaml          # 성과 지표 요약
├── rolling_metrics.parquet           # 롤링 지표
├── risk_analysis.yaml               # 리스크 분석 결과
├── portfolio_weights.parquet         # 포트폴리오 가중치
└── summary_report.md                 # 마크다운 요약 보고서
```

## 고급 사용 사례

### 1. 다중 전략 비교

```python
# 여러 최적화 전략 비교
strategies = {
    "Max Sharpe": {"objective": "max_sharpe"},
    "Min Variance": {"objective": "min_variance"}, 
    "Risk Parity": {"objective": "mean_variance", "risk_aversion": 0.5}
}

strategy_results = {}

for name, strategy_config in strategies.items():
    config_copy = config.copy()
    config_copy["portfolio"].update(strategy_config)
    
    pipeline = BacktestPipeline(config=config_copy)
    results = pipeline.run_full_backtest(
        data_root="/path/to/data",
        start_date="2022-01-01", 
        end_date="2023-12-31",
        output_dir=f"./results/{name.lower().replace(' ', '_')}"
    )
    
    strategy_results[name] = results["performance_metrics"]["summary_metrics"]

# 전략별 성과 비교
comparison_df = pd.DataFrame(strategy_results).T
print("=== 전략 비교 ===")
print(comparison_df[['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown']].round(4))
```

### 2. 커스텀 성과 지표 추가

```python
from krx_portfolio.backtesting.metrics import PerformanceMetrics

class CustomMetrics(PerformanceMetrics):
    def calculate_custom_metrics(self, returns):
        """사용자 정의 지표 계산."""
        # 예: 상승장 대 하락장 성과
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        return {
            'upside_capture': positive_returns.mean() / abs(negative_returns.mean()) 
                             if len(negative_returns) > 0 else float('inf'),
            'win_rate': len(positive_returns) / len(returns),
            'profit_factor': positive_returns.sum() / abs(negative_returns.sum())
                            if negative_returns.sum() != 0 else float('inf')
        }

# 사용
custom_metrics = CustomMetrics()
portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
custom_results = custom_metrics.calculate_custom_metrics(portfolio_returns)
print("커스텀 지표:", custom_results)
```

### 3. 실시간 백테스팅 모니터링

```python
import time
from datetime import datetime, timedelta

def rolling_backtest_update(pipeline, data_source, window_days=252):
    """롤링 백테스팅 업데이트."""
    
    while True:
        try:
            # 최신 데이터 가져오기
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=window_days)
            
            # 백테스팅 실행
            results = pipeline.run_full_backtest(
                data_root=data_source,
                start_date=str(start_date),
                end_date=str(end_date),
                output_dir=f"./live_results/{end_date}"
            )
            
            # 주요 지표 출력
            metrics = results["performance_metrics"]["summary_metrics"]
            print(f"\n=== {end_date} 업데이트 ===")
            print(f"연간 수익률: {metrics['annualized_return']:.2%}")
            print(f"샤프 비율: {metrics['sharpe_ratio']:.3f}")
            print(f"최대 낙폭: {metrics['max_drawdown']:.2%}")
            
            # 1시간 대기
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("백테스팅 모니터링 종료")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(60)  # 1분 후 재시도

# 실행 (예시)
# rolling_backtest_update(pipeline, "/path/to/live/data")
```

## 성능 최적화 팁

### 1. 데이터 전처리 최적화

```python
# 대용량 데이터 처리시
config["etl"]["cache_dir"] = "/fast/ssd/cache"  # SSD 캐시 사용
config["backtest"]["initial_capital"] = 1_000_000  # 적절한 자본 규모
```

### 2. 메모리 효율적 처리

```python
# 큰 데이터셋 처리시 청크 단위 처리
def chunked_backtest(pipeline, data_chunks, chunk_size=1000):
    """청크 단위 백테스팅."""
    results = []
    
    for i in range(0, len(data_chunks), chunk_size):
        chunk = data_chunks[i:i+chunk_size]
        chunk_result = pipeline.run_backtest_with_weights(
            weights=chunk["weights"],
            returns=chunk["returns"],
            prices=chunk.get("prices")
        )
        results.append(chunk_result)
    
    return results
```

### 3. 병렬 처리

```python
from multiprocessing import Pool
import functools

def parallel_strategy_backtest(strategies, data_path):
    """병렬 전략 백테스팅."""
    
    def run_strategy(strategy_config):
        pipeline = BacktestPipeline(config=strategy_config)
        return pipeline.run_full_backtest(
            data_root=data_path,
            start_date="2022-01-01",
            end_date="2023-12-31"
        )
    
    # 병렬 실행
    with Pool(processes=4) as pool:
        results = pool.map(run_strategy, strategies.values())
    
    return dict(zip(strategies.keys(), results))
```

## 트러블슈팅

### 일반적인 문제들

1. **메모리 부족**
   - `lookback_window` 줄이기
   - `rolling_window` 크기 조정
   - 청크 단위 처리 사용

2. **느린 성능**
   - SSD 캐시 디렉토리 사용
   - `force_reload=False` 설정
   - 불필요한 지표 계산 제외

3. **수치적 불안정성**
   - `ridge` 정규화 증가
   - 극값 데이터 전처리
   - `market_impact_model="none"` 사용

4. **데이터 정렬 오류**
   - 날짜 인덱스 확인
   - 자산 이름 일치 확인
   - 결측치 처리

### 디버깅 도구

```python
import logging

# 상세한 로깅 활성화
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('krx_portfolio.backtesting')

# 디버그 모드 백테스팅
pipeline = BacktestPipeline(config=config)
pipeline.backtest_engine.logger.setLevel(logging.DEBUG)

results = pipeline.run_full_backtest(
    data_root="/path/to/data",
    start_date="2023-01-01",
    end_date="2023-03-31"
)
```

이 가이드는 KRX Dynamic Portfolio 백테스팅 모듈의 완전한 사용법을 다룹니다. 실무에서 바로 활용할 수 있는 현실적인 백테스팅 환경을 제공하여, 포트폴리오 전략의 과거 성과를 정확하게 평가할 수 있습니다.