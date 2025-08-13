# ETL 모듈 API 문서

KRX Dynamic Portfolio 프로젝트의 ETL (Extract, Transform, Load) 파이프라인 API 문서입니다.

## 개요

ETL 파이프라인은 KRX JSON 데이터를 로드하고 전처리하여 포트폴리오 분석에 적합한 형태로 변환합니다.

### 주요 구성 요소

- **KRXDataLoader**: KRX JSON 파일 로딩
- **KRXPreprocessor**: 데이터 정제 및 전처리  
- **ETL Pipeline**: 전체 파이프라인 실행 및 캐싱

## KRXDataLoader

KRX JSON 데이터를 pandas DataFrame으로 로드하는 클래스입니다.

### 클래스 초기화

```python
from krx_portfolio.etl.data_loader import KRXDataLoader

loader = KRXDataLoader(data_root="/path/to/krx-json-data")
```

**Parameters:**
- `data_root` (str | Path): KRX JSON 데이터 루트 경로

### 메서드

#### `load_single_date(date)`

단일 날짜의 주식 데이터를 로드합니다.

```python
df = loader.load_single_date("20231215")
# 또는
df = loader.load_single_date(datetime(2023, 12, 15))
```

**Parameters:**
- `date` (str | datetime): 날짜 (YYYYMMDD 형태 또는 datetime 객체)

**Returns:**
- `pd.DataFrame`: 해당 날짜의 주식 데이터

**Raises:**
- `FileNotFoundError`: 데이터 파일이 존재하지 않는 경우

#### `load_date_range(start_date, end_date)`

날짜 범위의 주식 데이터를 로드합니다.

```python
df = loader.load_date_range("20231201", "20231215")
```

**Parameters:**
- `start_date` (str | datetime): 시작 날짜
- `end_date` (str | datetime): 종료 날짜

**Returns:**
- `pd.DataFrame`: 기간 내 모든 주식 데이터를 합친 DataFrame

**Raises:**
- `ValueError`: 지정된 기간에 데이터가 없는 경우

#### `load_latest_available(days_back=30)`

최근 사용 가능한 데이터를 로드합니다.

```python
df = loader.load_latest_available(days_back=7)
```

**Parameters:**
- `days_back` (int): 과거 며칠까지 검색할지 (기본값: 30)

**Returns:**
- `pd.DataFrame`: 가장 최근 사용 가능한 날짜의 주식 데이터

#### `get_available_dates(year=None)`

사용 가능한 날짜 목록을 반환합니다.

```python
dates = loader.get_available_dates(year=2023)
```

**Parameters:**
- `year` (int, optional): 특정 연도만 조회 (None이면 모든 연도)

**Returns:**
- `List[str]`: 사용 가능한 날짜 목록 (YYYYMMDD 형태)

#### `get_market_summary(date)`

시장 전체 요약 통계를 반환합니다.

```python
summary = loader.get_market_summary("20231215")
```

**Parameters:**
- `date` (str | datetime): 조회 날짜

**Returns:**
- `Dict`: 시장 요약 정보
  - `date`: 날짜
  - `total_stocks`: 전체 종목 수
  - `kospi`: KOSPI 시장 정보 (count, total_market_cap, total_volume)
  - `kosdaq`: KOSDAQ 시장 정보 (count, total_market_cap, total_volume)

### 데이터 스키마

로드된 DataFrame의 주요 컬럼:

| 컬럼명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| `basDt` | datetime64 | 기준일 |
| `srtnCd` | object | 단축코드 (종목코드) |
| `itmsNm` | object | 종목명 |
| `mrktCtg` | category | 시장구분 (KOSPI/KOSDAQ) |
| `clpr` | int64 | 종가 |
| `vs` | int64 | 전일대비 |
| `fltRt` | float64 | 등락률 (소수점 형태) |
| `mkp` | int64 | 시가 |
| `hipr` | int64 | 고가 |
| `lopr` | int64 | 저가 |
| `trqu` | int64 | 거래량 |
| `trPrc` | int64 | 거래대금 |
| `lstgStCnt` | int64 | 상장주식수 |
| `mrktTotAmt` | int64 | 시가총액 |

## KRXPreprocessor

원시 KRX 데이터를 포트폴리오 분석에 적합한 형태로 전처리하는 클래스입니다.

### 클래스 초기화

```python
from krx_portfolio.etl.preprocessor import KRXPreprocessor

preprocessor = KRXPreprocessor(
    min_market_cap=100e8,  # 최소 시가총액 (원)
    min_volume=1000        # 최소 거래량 (주)
)
```

**Parameters:**
- `min_market_cap` (float): 최소 시가총액 필터 (기본값: 100억원)
- `min_volume` (int): 최소 거래량 필터 (기본값: 1000주)

### 메서드

#### `clean_data(df)`

기본 데이터 정제를 수행합니다.

```python
cleaned_df = preprocessor.clean_data(raw_df)
```

**Parameters:**
- `df` (pd.DataFrame): 원시 KRX 데이터프레임

**Returns:**
- `pd.DataFrame`: 정제된 데이터프레임

**처리 과정:**
1. 결측치 처리
2. 이상치 필터링
3. 기본 필터 적용 (시가총액, 거래량)
4. 파생 변수 생성

#### `create_ohlcv_panel(df)`

OHLCV 패널 데이터를 생성합니다.

```python
panel_df = preprocessor.create_ohlcv_panel(cleaned_df)
```

**Parameters:**
- `df` (pd.DataFrame): 정제된 KRX 데이터프레임

**Returns:**
- `pd.DataFrame`: 멀티인덱스 (date, stock_code) OHLCV 데이터

**컬럼:**
- `stock_name`: 종목명
- `open`: 시가
- `high`: 고가  
- `low`: 저가
- `close`: 종가
- `volume`: 거래량
- `amount`: 거래대금

#### `create_returns_matrix(df, period='1D')`

수익률 매트릭스를 생성합니다.

```python
returns_df = preprocessor.create_returns_matrix(ohlcv_df, period='1D')
```

**Parameters:**
- `df` (pd.DataFrame): OHLCV 패널 데이터
- `period` (str): 수익률 계산 주기 ('1D', '5D', '1M')

**Returns:**
- `pd.DataFrame`: 인덱스(날짜) × 컬럼(종목코드) 수익률 매트릭스

#### `create_market_cap_weights(df)`

시가총액 기준 가중치를 생성합니다.

```python
weights_df = preprocessor.create_market_cap_weights(raw_df)
```

**Parameters:**
- `df` (pd.DataFrame): 원시 KRX 데이터프레임

**Returns:**
- `pd.DataFrame`: 인덱스(날짜) × 컬럼(종목코드) 시가총액 비중 매트릭스

#### `filter_investable_universe(df, top_n=200, market=None)`

투자 유니버스를 필터링합니다.

```python
# KOSPI 상위 100개
kospi_stocks = preprocessor.filter_investable_universe(
    df, top_n=100, market='KOSPI'
)

# 전체 시장 상위 200개
all_stocks = preprocessor.filter_investable_universe(
    df, top_n=200, market=None
)
```

**Parameters:**
- `df` (pd.DataFrame): 원시 KRX 데이터프레임
- `top_n` (int): 상위 N개 종목 (기본값: 200)
- `market` (str, optional): 시장 필터 ('KOSPI', 'KOSDAQ', None)

**Returns:**
- `List[str]`: 선별된 종목코드 리스트

#### `get_data_quality_report(df)`

데이터 품질 보고서를 생성합니다.

```python
report = preprocessor.get_data_quality_report(df)
```

**Parameters:**
- `df` (pd.DataFrame): 분석할 데이터프레임

**Returns:**
- `Dict`: 데이터 품질 보고서
  - `total_records`: 전체 레코드 수
  - `date_range`: 날짜 범위 정보
  - `universe_size`: 종목 수 정보
  - `missing_data`: 결측치 정보
  - `market_coverage`: 시장 커버리지 정보

### 파생 변수

전처리 과정에서 생성되는 파생 변수들:

| 변수명 | 설명 | 계산식 |
|--------|------|--------|
| `intraday_volatility` | 일중 변동성 | (고가 - 저가) / 종가 |
| `open_close_spread` | 시가-종가 스프레드 | (종가 - 시가) / 시가 |
| `turnover_ratio` | 거래회전율 | 거래량 / 상장주식수 |
| `trading_value_ratio` | 거래대금 비율 | 거래대금 / 시가총액 |
| `log_market_cap` | 로그 시가총액 | log(시가총액) |
| `market_cap_rank` | 시가총액 분위수 | 시가총액 백분위수 순위 |

## ETL Pipeline

전체 ETL 파이프라인을 실행하는 메인 함수입니다.

### 함수

#### `run_etl_pipeline(data_root, start_date=None, end_date=None, cache_path="./data/cache", force_reload=False)`

ETL 파이프라인을 실행합니다.

```python
from krx_portfolio.etl.main import run_etl_pipeline

run_etl_pipeline(
    data_root="/path/to/krx-json-data",
    start_date="20231201",
    end_date="20231215",
    cache_path="./data/cache",
    force_reload=False
)
```

**Parameters:**
- `data_root` (str): KRX JSON 데이터 루트 경로
- `start_date` (str, optional): 시작 날짜 (YYYYMMDD, None이면 최근 30일)
- `end_date` (str, optional): 종료 날짜 (YYYYMMDD, None이면 오늘)
- `cache_path` (str): 캐시 저장 경로 (기본값: "./data/cache")
- `force_reload` (bool): 강제 리로드 여부 (기본값: False)

**산출물:**

캐시 디렉토리에 다음 파일들이 생성됩니다:

```
cache/
├── raw/
│   └── krx_data_{start_date}_{end_date}.parquet
├── processed/
│   └── krx_processed_{start_date}_{end_date}.parquet
└── features/
    ├── ohlcv_panel_{start_date}_{end_date}.parquet
    ├── daily_returns_{start_date}_{end_date}.parquet
    ├── market_cap_weights_{start_date}_{end_date}.parquet
    └── investment_universe_{end_date}.json
```

### CLI 사용법

```bash
python -m krx_portfolio.etl.main \
    --data-root /path/to/krx-json-data \
    --start-date 20231201 \
    --end-date 20231215 \
    --cache-path ./data/cache \
    --force-reload
```

**Arguments:**
- `--data-root`: KRX JSON 데이터 루트 경로 (필수)
- `--start-date`: 시작 날짜 (YYYYMMDD)
- `--end-date`: 종료 날짜 (YYYYMMDD)
- `--cache-path`: 캐시 저장 경로
- `--force-reload`: 강제 리로드 플래그

## 예외 처리

### 일반적인 예외들

- `FileNotFoundError`: 데이터 파일이 존재하지 않는 경우
- `ValueError`: 잘못된 날짜 형식이나 범위
- `KeyError`: JSON 구조가 예상과 다른 경우
- `TypeError`: 잘못된 데이터 타입 전달

### 에러 처리 예시

```python
try:
    df = loader.load_single_date("20231215")
except FileNotFoundError as e:
    print(f"데이터 파일 없음: {e}")
except Exception as e:
    print(f"로딩 실패: {e}")
```

## 성능 최적화

### 캐싱 시스템

- 원시 데이터, 전처리 데이터, 특성 데이터 단계별 캐싱
- Parquet 형식으로 빠른 I/O 성능
- 캐시 무효화는 `force_reload` 옵션으로 제어

### 메모리 관리

- 대용량 데이터 처리 시 청크 단위 처리 권장
- 불필요한 컬럼 제거로 메모리 사용량 최적화
- 카테고리형 데이터 활용으로 메모리 절약

### 병렬 처리

현재 버전에서는 단일 스레드로 동작하지만, 향후 다음과 같은 병렬 처리 개선 예정:

- 날짜별 병렬 로딩
- 종목별 병렬 전처리
- 특성 생성 병렬화

## 버전 호환성

- Python 3.9+
- pandas 2.0+
- numpy 1.24+

데이터 스키마 변경 시 하위 호환성을 위해 버전별 변환 로직을 제공합니다.