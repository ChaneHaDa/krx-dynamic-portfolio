# ETL 사용 가이드

KRX Dynamic Portfolio ETL 파이프라인 사용법을 단계별로 설명합니다.

## 시작하기

### 1. 설치 및 설정

```bash
# 프로젝트 의존성 설치
make install

# 개발 환경 설정 (옵션)
make dev
```

### 2. KRX 데이터 준비

KRX JSON 데이터가 다음 구조로 준비되어 있어야 합니다:

```
/path/to/krx-json-data/
└── Price/
    └── STOCK/
        ├── 2023/
        │   ├── 20231201.json
        │   ├── 20231202.json
        │   └── ...
        └── 2024/
            ├── 20240101.json
            └── ...
```

## 기본 사용법

### 1. CLI로 ETL 실행

가장 간단한 방법으로 전체 파이프라인을 실행합니다:

```bash
# 기본 실행 (최근 30일 데이터)
python -m krx_portfolio.etl.main --data-root /path/to/krx-json-data

# 특정 기간 실행
python -m krx_portfolio.etl.main \
    --data-root /path/to/krx-json-data \
    --start-date 20231201 \
    --end-date 20231215

# 캐시 경로 지정
python -m krx_portfolio.etl.main \
    --data-root /path/to/krx-json-data \
    --cache-path ./my_cache \
    --force-reload
```

### 2. Python 코드로 ETL 실행

```python
from krx_portfolio.etl.main import run_etl_pipeline

# 기본 실행
run_etl_pipeline(
    data_root="/path/to/krx-json-data",
    start_date="20231201",
    end_date="20231215"
)
```

## 단계별 사용법

### 1. 데이터 로딩

```python
from krx_portfolio.etl.data_loader import KRXDataLoader

# 로더 초기화
loader = KRXDataLoader("/path/to/krx-json-data")

# 단일 날짜 로드
df = loader.load_single_date("20231215")
print(f"로드된 종목 수: {len(df)}")

# 날짜 범위 로드
df = loader.load_date_range("20231201", "20231215")
print(f"전체 레코드: {len(df)}")

# 최근 데이터 로드
df = loader.load_latest_available(days_back=7)

# 사용 가능한 날짜 확인
dates = loader.get_available_dates(year=2023)
print(f"2023년 사용 가능한 날짜: {len(dates)}일")
```

### 2. 데이터 전처리

```python
from krx_portfolio.etl.preprocessor import KRXPreprocessor

# 전처리기 초기화 (필터 조건 설정)
preprocessor = KRXPreprocessor(
    min_market_cap=100e8,  # 100억원 이상
    min_volume=1000        # 1000주 이상
)

# 데이터 정제
cleaned_df = preprocessor.clean_data(raw_df)
print(f"정제 후 레코드: {len(cleaned_df)}")

# 데이터 품질 보고서
report = preprocessor.get_data_quality_report(cleaned_df)
print(f"총 {report['universe_size']['total_stocks']}개 종목")
print(f"KOSPI: {report['universe_size']['kospi_stocks']}개")
print(f"KOSDAQ: {report['universe_size']['kosdaq_stocks']}개")
```

### 3. 분석용 데이터 생성

```python
# OHLCV 패널 데이터 생성
ohlcv_df = preprocessor.create_ohlcv_panel(cleaned_df)
print(f"OHLCV 패널 크기: {ohlcv_df.shape}")

# 일일 수익률 매트릭스
returns_df = preprocessor.create_returns_matrix(ohlcv_df, period='1D')
print(f"수익률 매트릭스: {returns_df.shape}")

# 시가총액 가중치
weights_df = preprocessor.create_market_cap_weights(cleaned_df)
print(f"가중치 매트릭스: {weights_df.shape}")

# 투자 유니버스 생성
kospi_top100 = preprocessor.filter_investable_universe(
    cleaned_df, top_n=100, market='KOSPI'
)
kosdaq_top50 = preprocessor.filter_investable_universe(
    cleaned_df, top_n=50, market='KOSDAQ'
)
print(f"KOSPI Top 100: {len(kospi_top100)}개")
print(f"KOSDAQ Top 50: {len(kosdaq_top50)}개")
```

## 고급 사용법

### 1. 데이터 품질 분석

```python
# 상세 데이터 품질 분석
def analyze_data_quality(df):
    report = preprocessor.get_data_quality_report(df)
    
    print("=== 데이터 품질 보고서 ===")
    print(f"전체 레코드: {report['total_records']:,}건")
    print(f"데이터 기간: {report['date_range']['start']} ~ {report['date_range']['end']}")
    print(f"거래일수: {report['date_range']['trading_days']}일")
    
    print(f"\n=== 종목 정보 ===")
    print(f"전체 종목: {report['universe_size']['total_stocks']:,}개")
    print(f"KOSPI: {report['universe_size']['kospi_stocks']:,}개")
    print(f"KOSDAQ: {report['universe_size']['kosdaq_stocks']:,}개")
    
    print(f"\n=== 시장 커버리지 ===")
    print(f"일평균 종목수: {report['market_coverage']['avg_daily_stocks']:.1f}개")
    print(f"최소 종목수: {report['market_coverage']['min_daily_stocks']}개")
    print(f"최대 종목수: {report['market_coverage']['max_daily_stocks']}개")
    
    if report['missing_data']:
        print(f"\n=== 결측치 정보 ===")
        for col, count in report['missing_data'].items():
            print(f"{col}: {count}건")
    else:
        print(f"\n=== 결측치: 없음 ✅ ===")

analyze_data_quality(cleaned_df)
```

### 2. 시장 분석

```python
# 시장별 통계 분석
def market_analysis(df, date):
    summary = loader.get_market_summary(date)
    
    print(f"=== {date} 시장 현황 ===")
    print(f"전체 종목: {summary['total_stocks']}개")
    
    print(f"\n=== KOSPI ===")
    print(f"종목수: {summary['kospi']['count']:,}개")
    print(f"시가총액: {summary['kospi']['total_market_cap']/1e12:.1f}조원")
    print(f"거래량: {summary['kospi']['total_volume']/1e8:.1f}억주")
    
    print(f"\n=== KOSDAQ ===")
    print(f"종목수: {summary['kosdaq']['count']:,}개")
    print(f"시가총액: {summary['kosdaq']['total_market_cap']/1e12:.1f}조원")
    print(f"거래량: {summary['kosdaq']['total_volume']/1e8:.1f}억주")

market_analysis(cleaned_df, "20231215")
```

### 3. 수익률 분석

```python
import numpy as np

# 수익률 통계 분석
def returns_analysis(returns_df):
    print("=== 수익률 분석 ===")
    
    # 기본 통계
    mean_returns = returns_df.mean()
    std_returns = returns_df.std()
    
    print(f"평균 일일수익률: {mean_returns.mean()*100:.3f}%")
    print(f"평균 변동성: {std_returns.mean()*100:.3f}%")
    
    # 상위/하위 종목
    top_performers = mean_returns.nlargest(5)
    worst_performers = mean_returns.nsmallest(5)
    
    print(f"\n=== 상위 5개 종목 (평균 일일수익률) ===")
    for stock, ret in top_performers.items():
        print(f"{stock}: {ret*100:.3f}%")
    
    print(f"\n=== 하위 5개 종목 (평균 일일수익률) ===")
    for stock, ret in worst_performers.items():
        print(f"{stock}: {ret*100:.3f}%")
    
    # 리스크 분석
    high_vol_stocks = std_returns.nlargest(5)
    low_vol_stocks = std_returns.nsmallest(5)
    
    print(f"\n=== 고변동성 5개 종목 ===")
    for stock, vol in high_vol_stocks.items():
        print(f"{stock}: {vol*100:.3f}%")

returns_analysis(returns_df)
```

## 캐싱 활용

### 1. 캐시 구조 이해

```python
import os
from pathlib import Path

def explore_cache(cache_path="./data/cache"):
    cache_dir = Path(cache_path)
    
    print("=== 캐시 구조 ===")
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(str(cache_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.1f}MB)")

explore_cache()
```

### 2. 캐시 관리

```python
def clear_cache(cache_path="./data/cache", keep_raw=True):
    """캐시 정리 (원시 데이터는 보존 옵션)"""
    cache_dir = Path(cache_path)
    
    dirs_to_clear = ["processed", "features"]
    if not keep_raw:
        dirs_to_clear.append("raw")
    
    for dir_name in dirs_to_clear:
        dir_path = cache_dir / dir_name
        if dir_path.exists():
            for file in dir_path.iterdir():
                file.unlink()
                print(f"삭제됨: {file}")

def cache_info(cache_path="./data/cache"):
    """캐시 정보 확인"""
    cache_dir = Path(cache_path)
    
    for subdir in ["raw", "processed", "features"]:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"{subdir}: {len(files)}개 파일, {total_size/1024/1024:.1f}MB")

cache_info()
```

## 성능 최적화

### 1. 메모리 효율적인 처리

```python
# 대용량 데이터 처리 시 청크 단위로 처리
def process_large_dataset(data_root, start_date, end_date, chunk_days=7):
    """청크 단위로 대용량 데이터 처리"""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    current_dt = start_dt
    all_results = []
    
    while current_dt <= end_dt:
        chunk_end = min(current_dt + timedelta(days=chunk_days), end_dt)
        
        print(f"처리 중: {current_dt.strftime('%Y%m%d')} ~ {chunk_end.strftime('%Y%m%d')}")
        
        # 청크 단위 ETL 실행
        run_etl_pipeline(
            data_root=data_root,
            start_date=current_dt.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
            cache_path=f"./cache_chunk_{current_dt.strftime('%Y%m%d')}"
        )
        
        current_dt = chunk_end + timedelta(days=1)

# 메모리 사용량 모니터링
def monitor_memory():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"현재 메모리 사용량: {memory_mb:.1f}MB")

monitor_memory()
```

### 2. 병렬 처리 (향후 구현 예정)

```python
# 향후 버전에서 지원될 병렬 처리 예시
def parallel_etl_pipeline(data_root, date_list, n_workers=4):
    """병렬 ETL 처리 (향후 구현 예정)"""
    from concurrent.futures import ProcessPoolExecutor
    
    def process_single_date(date):
        return run_etl_pipeline(
            data_root=data_root,
            start_date=date,
            end_date=date,
            cache_path=f"./cache_{date}"
        )
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_single_date, date_list))
    
    return results
```

## 문제 해결

### 1. 일반적인 오류

**파일을 찾을 수 없음**
```python
try:
    df = loader.load_single_date("20231215")
except FileNotFoundError:
    # 사용 가능한 날짜 확인
    available_dates = loader.get_available_dates(year=2023)
    print(f"사용 가능한 날짜: {available_dates[-5:]}")  # 최근 5일
```

**메모리 부족**
```python
# 더 작은 기간으로 나누어 처리
try:
    df = loader.load_date_range("20230101", "20231231")
except MemoryError:
    print("메모리 부족 - 더 작은 기간으로 나누어 처리하세요")
    # 월별로 나누어 처리
    for month in range(1, 13):
        start_date = f"2023{month:02d}01"
        end_date = f"2023{month:02d}31"
        monthly_df = loader.load_date_range(start_date, end_date)
```

**데이터 품질 문제**
```python
# 데이터 품질 검사 후 처리
report = preprocessor.get_data_quality_report(raw_df)

if report['missing_data']:
    print("⚠️ 결측치 발견:")
    for col, count in report['missing_data'].items():
        print(f"  {col}: {count}건")

# 최소 기준 미달 시 경고
if report['market_coverage']['avg_daily_stocks'] < 2000:
    print("⚠️ 일평균 종목수가 예상보다 적습니다")
```

### 2. 성능 모니터링

```python
import time

def timed_etl_pipeline(*args, **kwargs):
    """실행 시간 측정이 포함된 ETL 파이프라인"""
    start_time = time.time()
    
    try:
        result = run_etl_pipeline(*args, **kwargs)
        success = True
    except Exception as e:
        print(f"ETL 실행 실패: {e}")
        success = False
        result = None
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n⏱️ 실행 시간: {elapsed:.2f}초")
    print(f"🎯 실행 결과: {'성공' if success else '실패'}")
    
    return result

# 사용 예시
timed_etl_pipeline(
    data_root="/path/to/data",
    start_date="20231201",
    end_date="20231215"
)
```

## Makefile 명령어

편의를 위해 제공되는 Makefile 명령어들:

```bash
# ETL 파이프라인 실행
make etl

# 특정 날짜 범위로 ETL 실행 (환경변수 설정 필요)
DATA_ROOT=/path/to/data START_DATE=20231201 END_DATE=20231215 make etl

# 캐시 정리 후 ETL 실행
make clean-cache etl

# 테스트 실행
make test

# 코드 품질 검사
make lint

# 전체 CI 파이프라인 (린트 + 테스트)
make ci-test
```

이 가이드를 통해 KRX Dynamic Portfolio ETL 파이프라인을 효과적으로 활용할 수 있습니다.