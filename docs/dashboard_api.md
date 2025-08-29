# Dashboard API 문서

> **KRX Dynamic Portfolio** Phase 4 - Streamlit 대시보드 모듈 API 레퍼런스

## 📋 목차

1. [모듈 개요](#모듈-개요)
2. [메인 대시보드](#메인-대시보드)
3. [데이터 통합](#데이터-통합)
4. [UI 컴포넌트](#ui-컴포넌트)
5. [차트 컴포넌트](#차트-컴포넌트)
6. [사용 예시](#사용-예시)

---

## 🎯 모듈 개요

대시보드는 다음 4개 주요 모듈로 구성됩니다:

```
krx_portfolio/app/
├── dashboard.py              # 메인 대시보드 애플리케이션
├── data_integration.py       # 실시간 데이터 연동
└── components/
    ├── __init__.py          # 컴포넌트 패키지
    ├── portfolio_components.py  # 포트폴리오 UI 컴포넌트
    ├── charts.py            # 차트 생성 함수
    └── data_components.py   # 데이터 관리 컴포넌트
```

---

## 📊 메인 대시보드 (`dashboard.py`)

### `main()`

**메인 대시보드 함수**

Streamlit 애플리케이션의 진입점으로, 페이지 설정과 네비게이션을 담당합니다.

```python
def main() -> None:
    """
    메인 대시보드 함수
    
    Features:
    - 페이지 설정 (타이틀, 아이콘, 레이아웃)
    - 사이드바 네비게이션
    - 5개 페이지 라우팅
    """
```

### 페이지 함수들

#### `show_home_page()`

홈 페이지 렌더링 - 실시간 시장 현황 및 시스템 개요

```python
def show_home_page() -> None:
    """
    홈 페이지 표시
    
    Components:
    - 실시간 KOSPI 지수 및 변동률
    - 시장 거래 상태 (장중/장마감)
    - 시스템 상태 (테스트, 커버리지)
    - 빠른 시작 가이드
    """
```

#### `show_optimization_page()`

포트폴리오 최적화 인터페이스

```python
def show_optimization_page() -> None:
    """
    포트폴리오 최적화 페이지
    
    Controls:
    - 최적화 방법 선택 (max_sharpe, min_variance, mean_variance)
    - 리스크 모델 선택 (sample, ledoit_wolf, oas, ewma)
    - 제약 조건 설정 (max_weight, lookback_days)
    - 최적화 실행 버튼
    
    Outputs:
    - 포트폴리오 구성 테이블
    - 자산 배분 파이 차트
    """
```

#### `show_backtesting_page()`

백테스팅 설정 및 결과 분석

```python
def show_backtesting_page() -> None:
    """
    백테스팅 페이지
    
    Settings:
    - 기간 설정 (start_date, end_date)
    - 자본 설정 (initial_capital, transaction_cost)
    - 리밸런싱 설정 (frequency, benchmark)
    
    Results:
    - 누적 수익률 차트
    - 성과 지표 메트릭
    """
```

#### `show_risk_analytics_page()`

리스크 분석 및 시각화

```python
def show_risk_analytics_page() -> None:
    """
    리스크 분석 페이지
    
    Metrics:
    - VaR/CVaR (95% 신뢰구간)
    - 추적 오차, 하방 편차
    - 최대 낙폭 지속 기간
    
    Charts:
    - 수익률 분포 히스토그램
    - VaR 경계선 표시
    """
```

#### `show_data_management_page()`

데이터 관리 및 ETL 제어

```python
def show_data_management_page() -> None:
    """
    데이터 관리 페이지
    
    Controls:
    - ETL 파이프라인 실행
    - 캐시 관리 (삭제, 통계)
    - 데이터 품질 체크
    
    Monitoring:
    - 처리 상태, 업데이트 시점
    - 저장 공간 사용량
    """
```

---

## 🔄 데이터 통합 (`data_integration.py`)

### 실시간 데이터 수집

#### `fetch_real_time_data()`

yfinance를 사용한 주식 데이터 수집

```python
@st.cache_data(ttl=3600)
def fetch_real_time_data(
    symbols: List[str], 
    period: str = "1y"
) -> pd.DataFrame:
    """
    실시간 주식 데이터 수집
    
    Parameters:
    - symbols: 주식 심볼 리스트 ['005930.KS', '035420.KS']
    - period: 데이터 기간 ('1d', '1mo', '1y', 'max')
    
    Returns:
    - pd.DataFrame: OHLCV 데이터 (MultiIndex)
    
    Cache: 1시간 TTL
    """
```

#### `get_current_prices()`

현재 주식 가격 조회

```python
@st.cache_data(ttl=900)  
def get_current_prices(symbols: List[str]) -> pd.Series:
    """
    현재 주식 가격 조회
    
    Parameters:
    - symbols: 주식 심볼 리스트
    
    Returns:
    - pd.Series: 현재 가격 시리즈
    
    Cache: 15분 TTL
    """
```

### 유틸리티 함수

#### `krx_symbol_to_yfinance()`

종목코드 변환

```python
def krx_symbol_to_yfinance(krx_code: str) -> str:
    """
    KRX 종목코드를 yfinance 심볼로 변환
    
    Parameters:
    - krx_code: KRX 종목코드 ('005930')
    
    Returns:
    - str: yfinance 심볼 ('005930.KS')
    """
```

#### `get_real_time_market_status()`

시장 현황 조회

```python
def get_real_time_market_status() -> Dict[str, Any]:
    """
    실시간 시장 현황 조회
    
    Returns:
    - dict: {
        'kospi_current': float,      # 현재 지수
        'kospi_change': float,       # 전일 대비 변화
        'kospi_change_pct': float,   # 변동률 (%)
        'is_trading_hours': bool,    # 거래 시간 여부
        'last_update': str          # 마지막 업데이트
      }
    """
```

#### `create_sample_portfolio_data()`

샘플 포트폴리오 데이터 생성

```python
def create_sample_portfolio_data(n_assets: int = 20) -> Dict[str, Any]:
    """
    샘플 포트폴리오 데이터 생성
    
    Parameters:
    - n_assets: 자산 개수
    
    Returns:
    - dict: {
        'prices': pd.DataFrame,           # 가격 데이터
        'returns': pd.DataFrame,          # 수익률 데이터
        'weights': pd.Series,             # 포트폴리오 가중치
        'sector_map': dict,               # 섹터 정보
        'portfolio_returns': pd.Series,   # 포트폴리오 수익률
        'cumulative_returns': pd.Series,  # 누적 수익률
        'total_value': float,             # 총 자산가치
        'daily_change_pct': float,        # 일간 변동률
        'num_holdings': int,              # 보유 종목 수
        'expected_return': float,         # 예상 수익률
        'volatility': float               # 변동성
      }
    """
```

---

## 🎨 UI 컴포넌트 (`components/`)

### 포트폴리오 컴포넌트 (`portfolio_components.py`)

#### `render_portfolio_summary()`

포트폴리오 요약 정보 렌더링

```python
def render_portfolio_summary(portfolio_data: Dict[str, Any]) -> None:
    """
    포트폴리오 요약 정보 렌더링
    
    Parameters:
    - portfolio_data: {
        'total_value': float,        # 총 자산가치
        'daily_change_pct': float,   # 일간 변동률
        'num_holdings': int,         # 보유 종목 수  
        'expected_return': float,    # 예상 수익률
        'volatility': float          # 변동성
      }
    
    UI Elements:
    - 4개 메트릭 카드 (총자산, 보유종목수, 예상수익률, 변동성)
    - Delta 표시로 전일 대비 변화 표시
    """
```

#### `render_allocation_chart()`

자산 배분 차트 렌더링

```python
def render_allocation_chart(
    weights: pd.Series, 
    names: Optional[List[str]] = None
) -> None:
    """
    포트폴리오 자산배분 차트 렌더링
    
    Parameters:
    - weights: 자산별 비중 시리즈
    - names: 자산 이름 리스트 (선택사항)
    
    Features:
    - 상위 10개 자산 표시, 나머지는 '기타'로 합계
    - 도넛 차트 (hole=0.3)
    - 상세 보유 현황 테이블 (확장 가능)
    """
```

#### `render_performance_metrics()`

성과 지표 메트릭 렌더링

```python
def render_performance_metrics(metrics: Dict[str, float]) -> None:
    """
    성과 지표 메트릭 렌더링
    
    Parameters:
    - metrics: {
        'total_return': float,       # 총 수익률
        'annualized_return': float,  # 연평균 수익률
        'sharpe_ratio': float,       # 샤프 비율
        'max_drawdown': float,       # 최대 낙폭
        'volatility': float,         # 변동성
        'var_95': float             # VaR 95%
      }
    
    Layout:
    - 3개 컬럼으로 구분 (수익률/위험조정수익률/위험지표)
    - 각 컬럼당 3개 메트릭 표시
    """
```

### 데이터 관리 컴포넌트 (`data_components.py`)

#### `render_data_status()`

데이터 현황 정보 렌더링

```python
def render_data_status(data_info: Dict[str, Any]) -> None:
    """
    데이터 상태 정보 렌더링
    
    Parameters:
    - data_info: {
        'total_stocks': int,        # 총 종목 수
        'last_update': str,         # 마지막 업데이트
        'data_period': dict,        # 데이터 기간
        'quality_score': float      # 데이터 품질 점수
      }
    
    UI Elements:
    - 4개 메트릭 카드 (종목수, 업데이트, 기간, 품질)
    """
```

#### `render_etl_controls()`

ETL 파이프라인 제어 인터페이스

```python
def render_etl_controls() -> Dict[str, Any]:
    """
    ETL 파이프라인 제어 인터페이스 렌더링
    
    Returns:
    - dict: ETL 설정 딕셔너리 (실행 버튼 클릭 시)
        {
            'data_root': str,      # 데이터 루트 경로
            'start_date': str,     # 시작일
            'end_date': str,       # 종료일
            'force_reload': bool   # 강제 재로드
        }
    
    Controls:
    - 데이터 경로 입력
    - 날짜 범위 선택
    - 강제 재로드 체크박스
    - ETL 실행 버튼
    """
```

#### `render_cache_management()`

캐시 관리 인터페이스

```python
def render_cache_management(cache_info: Dict[str, Any]) -> None:
    """
    캐시 관리 인터페이스 렌더링
    
    Parameters:
    - cache_info: {
        'total_size': int,         # 총 캐시 크기 (bytes)
        'file_count': int,         # 캐시 파일 수
        'last_cleared': str,       # 마지막 정리 시점
        'directories': dict        # 디렉토리별 상세 정보
      }
    
    Features:
    - 캐시 현황 메트릭 (크기, 파일수, 마지막 정리)
    - 디렉토리별 상세 테이블
    - 관리 버튼 (전체삭제, 오래된파일만, 통계, 검증)
    """
```

---

## 📈 차트 컴포넌트 (`charts.py`)

### `create_cumulative_return_chart()`

누적 수익률 차트 생성

```python
def create_cumulative_return_chart(
    returns_data: Dict[str, pd.Series], 
    title: str = "누적 수익률",
    height: int = 500
) -> go.Figure:
    """
    누적 수익률 차트 생성
    
    Parameters:
    - returns_data: 포트폴리오별 수익률 시리즈 딕셔너리
    - title: 차트 제목
    - height: 차트 높이
    
    Returns:
    - go.Figure: Plotly 차트 객체
    
    Features:
    - 다중 시리즈 지원
    - 호버 템플릿 최적화
    - 범례 위치 조정
    """
```

### `create_drawdown_chart()`

드로다운 차트 생성

```python
def create_drawdown_chart(
    returns: pd.Series,
    title: str = "최대 낙폭 (Drawdown)",
    height: int = 400
) -> go.Figure:
    """
    드로다운 차트 생성
    
    Parameters:
    - returns: 수익률 시리즈
    - title: 차트 제목
    - height: 차트 높이
    
    Returns:
    - go.Figure: 영역 차트 (빨간색 음영)
    
    Features:
    - Running maximum 대비 낙폭 계산
    - 0% 기준선 표시
    - 음영 영역으로 손실 구간 강조
    """
```

### `create_risk_return_scatter()`

위험-수익률 산점도 생성

```python
def create_risk_return_scatter(
    portfolios_data: Dict[str, Dict[str, float]],
    title: str = "위험-수익률 분포",
    height: int = 500
) -> go.Figure:
    """
    위험-수익률 산점도 생성
    
    Parameters:
    - portfolios_data: 포트폴리오별 위험-수익률 데이터
        {
            'portfolio_name': {
                'return': float,    # 수익률
                'risk': float,      # 위험 (변동성)
                'sharpe': float     # 샤프 비율
            }
        }
    
    Features:
    - 샤프 비율에 따른 색상 매핑
    - 컬러바 표시
    - 텍스트 라벨 포함
    """
```

### `create_correlation_heatmap()`

상관관계 히트맵 생성

```python
def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "상관관계 히트맵",
    height: int = 600
) -> go.Figure:
    """
    상관관계 히트맵 생성
    
    Parameters:
    - correlation_matrix: 상관관계 매트릭스
    - title: 차트 제목
    - height: 차트 높이
    
    Returns:
    - go.Figure: 히트맵 차트
    
    Features:
    - RdBu 컬러스케일 (빨강-파랑)
    - 중심값 0으로 설정
    - 셀 안에 상관계수 값 표시
    """
```

### `create_returns_distribution()`

수익률 분포 히스토그램 생성

```python
def create_returns_distribution(
    returns: pd.Series,
    title: str = "수익률 분포",
    height: int = 400
) -> go.Figure:
    """
    수익률 분포 히스토그램 생성
    
    Parameters:
    - returns: 수익률 시리즈
    - title: 차트 제목
    - height: 차트 높이
    
    Returns:
    - go.Figure: 히스토그램 + 통계선
    
    Features:
    - 50개 구간 히스토그램
    - 평균 수익률 선 (녹색 점선)
    - VaR 95%/99% 선 (빨간색 점선)
    - 통계 주석 표시
    """
```

---

## 💡 사용 예시

### 기본 대시보드 실행

```python
# main.py
import streamlit as st
from krx_portfolio.app.dashboard import main

if __name__ == "__main__":
    main()
```

### 커스텀 포트폴리오 컴포넌트

```python
import streamlit as st
import pandas as pd
from krx_portfolio.app.components.portfolio_components import (
    render_portfolio_summary,
    render_allocation_chart
)

# 포트폴리오 데이터 준비
portfolio_data = {
    'total_value': 100_000_000,    # 1억원
    'daily_change_pct': 1.25,      # +1.25%
    'num_holdings': 15,            # 15개 종목
    'expected_return': 8.5,        # 8.5% 연간 수익률
    'volatility': 12.3             # 12.3% 변동성
}

weights = pd.Series([0.15, 0.12, 0.10, 0.08, 0.07], 
                   index=['005930', '000660', '035420', '051910', '005380'])

# 컴포넌트 렌더링
render_portfolio_summary(portfolio_data)
render_allocation_chart(weights)
```

### 실시간 데이터 연동

```python
from krx_portfolio.app.data_integration import (
    get_real_time_market_status,
    fetch_real_time_data,
    krx_symbol_to_yfinance
)

# 실시간 시장 현황
market_status = get_real_time_market_status()
st.metric("KOSPI", f"{market_status['kospi_current']:.2f}", 
          f"{market_status['kospi_change_pct']:+.2f}%")

# 개별 종목 데이터
symbols = ['005930', '035420', '000660']  # 삼성전자, NAVER, SK하이닉스
yf_symbols = [krx_symbol_to_yfinance(code) for code in symbols]
data = fetch_real_time_data(yf_symbols, period='1mo')
```

### 커스텀 차트 생성

```python
from krx_portfolio.app.components.charts import (
    create_cumulative_return_chart,
    create_risk_return_scatter
)

# 누적 수익률 차트
returns_data = {
    '포트폴리오 A': portfolio_a_returns,
    '포트폴리오 B': portfolio_b_returns,
    'KOSPI': kospi_returns
}

chart = create_cumulative_return_chart(returns_data, "포트폴리오 성과 비교")
st.plotly_chart(chart, use_container_width=True)

# 위험-수익률 산점도
portfolios_data = {
    'Conservative': {'return': 5.2, 'risk': 8.1, 'sharpe': 0.64},
    'Aggressive': {'return': 12.8, 'risk': 18.5, 'sharpe': 0.69},
    'Balanced': {'return': 8.9, 'risk': 12.3, 'sharpe': 0.72}
}

scatter = create_risk_return_scatter(portfolios_data)
st.plotly_chart(scatter, use_container_width=True)
```

---

## 🔧 확장 가이드

### 새로운 페이지 추가

1. **dashboard.py에 페이지 함수 추가**:
```python
def show_new_page():
    st.title("새로운 페이지")
    # 페이지 내용 구현
```

2. **네비게이션에 등록**:
```python
pages = {
    # ... 기존 페이지들
    "🆕 새 기능": show_new_page
}
```

### 새로운 차트 타입 추가

```python
# charts.py에 추가
def create_custom_chart(data, **kwargs) -> go.Figure:
    """커스텀 차트 생성"""
    fig = go.Figure()
    # 차트 구현
    return fig
```

### 새로운 데이터 소스 연동

```python
# data_integration.py에 추가
@st.cache_data(ttl=1800)  # 30분 캐시
def fetch_external_data(source: str) -> pd.DataFrame:
    """외부 데이터 소스 연동"""
    # API 호출 및 데이터 처리
    return processed_data
```

---

**📝 문서 버전**: v1.0.0  
**📅 마지막 업데이트**: 2025-08-29  
**✍️ 작성자**: KRX Dynamic Portfolio Team