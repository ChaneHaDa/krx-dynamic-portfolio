# 문제 해결 가이드

> **KRX Dynamic Portfolio** - 일반적인 문제와 해결 방법

## 📋 목차

1. [설치 및 환경 문제](#설치-및-환경-문제)
2. [대시보드 실행 문제](#대시보드-실행-문제)
3. [데이터 관련 문제](#데이터-관련-문제)
4. [성능 문제](#성능-문제)
5. [테스트 관련 문제](#테스트-관련-문제)
6. [고급 문제 해결](#고급-문제-해결)

---

## 🛠️ 설치 및 환경 문제

### Python 버전 호환성 문제

**문제**: `Python 3.8 이하에서 실행 오류`
```bash
SyntaxError: positional argument follows keyword argument
```

**해결 방법**:
```bash
# Python 3.9 이상 사용 확인
python --version  # Python 3.9+ 필요

# pyenv로 버전 관리 (권장)
pyenv install 3.12.3
pyenv local 3.12.3
```

### 가상환경 설정 문제

**문제**: `externally-managed-environment` 오류
```bash
error: externally-managed-environment
× This environment is externally managed
```

**해결 방법**:
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 의존성 충돌 문제

**문제**: `pandas 버전 충돌`
```bash
ERROR: pip's dependency resolver does not currently consider all the packages
```

**해결 방법**:
```bash
# 의존성 업그레이드
pip install --upgrade pip
pip install --upgrade pandas numpy

# 충돌 시 강제 재설치
pip install --force-reinstall pandas>=2.0.0
```

---

## 📊 대시보드 실행 문제

### Streamlit 실행 실패

**문제**: `streamlit: command not found`
```bash
/bin/bash: line 1: streamlit: command not found
```

**해결 방법**:
```bash
# 가상환경 활성화 확인
source venv/bin/activate

# Streamlit 설치 확인
pip install streamlit

# 실행
streamlit run krx_portfolio/app/dashboard.py
```

### Import 오류

**문제**: `ImportError: cannot import name 'run_portfolio_optimization'`
```python
ImportError: cannot import name 'run_portfolio_optimization' 
from 'krx_portfolio.models.pipeline'
```

**해결 방법**:
```python
# 올바른 import 확인
from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline

# 사용법
pipeline = PortfolioOptimizationPipeline()
results = pipeline.build_weights(mu, returns)
```

### 브라우저 연결 실패

**문제**: `This site can't be reached`
```
localhost refused to connect
```

**해결 방법**:
```bash
# 포트 확인 및 변경
streamlit run krx_portfolio/app/dashboard.py --server.port 8502

# 방화벽 설정 확인 (Windows)
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501

# 네트워크 인터페이스 바인딩
streamlit run app.py --server.address 0.0.0.0
```

### 페이지 로딩 오류

**문제**: `Streamlit 페이지가 무한 로딩`

**해결 방법**:
```python
# 캐시 초기화
import streamlit as st
st.cache_data.clear()

# 또는 터미널에서
rm -rf ~/.streamlit  # Linux/macOS
```

---

## 📁 데이터 관련 문제

### yfinance API 연결 실패

**문제**: `HTTPError: 429 Too Many Requests`
```python
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**해결 방법**:
```python
import time
import yfinance as yf

# 요청 간격 조정
def fetch_with_retry(symbol, retries=3, delay=1):
    for i in range(retries):
        try:
            data = yf.download(symbol, progress=False)
            return data
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay * (2 ** i))  # 지수 백오프
            else:
                raise e

# 사용법
data = fetch_with_retry("005930.KS")
```

### 종목코드 변환 오류

**문제**: `No data found for symbol 005930`
```python
ValueError: No data found, symbol may be delisted
```

**해결 방법**:
```python
def krx_symbol_to_yfinance(krx_code: str) -> str:
    """KRX 종목코드를 yfinance 심볼로 변환"""
    # KOSPI/KOSDAQ 구분 (실제 구현 필요)
    if krx_code in KOSDAQ_CODES:  # 미리 정의된 KOSDAQ 종목 리스트
        return f"{krx_code}.KQ"
    else:
        return f"{krx_code}.KS"

# 검증 함수
def validate_symbol(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        return not hist.empty
    except:
        return False

# 사용법
symbol = krx_symbol_to_yfinance("035420")  # NAVER
if validate_symbol(symbol):
    data = yf.download(symbol)
```

### 캐시 관련 문제

**문제**: `Streamlit 캐시 메모리 부족`
```python
MemoryError: Unable to allocate array
```

**해결 방법**:
```python
# TTL 기반 캐시 사용
@st.cache_data(ttl=1800, max_entries=10)  # 30분, 최대 10개 항목
def fetch_data(symbol):
    return expensive_data_operation(symbol)

# 수동 캐시 정리
def clear_cache_periodically():
    if st.button("🧹 캐시 정리"):
        st.cache_data.clear()
        st.success("캐시가 정리되었습니다.")
        st.experimental_rerun()
```

### 데이터 형식 오류

**문제**: `AttributeError: 'NoneType' object has no attribute 'iloc'`
```python
AttributeError: 'NoneType' object has no attribute 'iloc'
```

**해결 방법**:
```python
def safe_data_processing(data):
    """안전한 데이터 처리 함수"""
    if data is None:
        st.warning("⚠️ 데이터를 불러올 수 없습니다.")
        return pd.DataFrame()
    
    if data.empty:
        st.warning("⚠️ 빈 데이터입니다.")
        return pd.DataFrame()
    
    # 데이터 정제
    data = data.dropna()
    
    if len(data) < 10:
        st.warning("⚠️ 데이터가 부족합니다 (최소 10개 필요).")
        return pd.DataFrame()
    
    return data

# 사용법
processed_data = safe_data_processing(raw_data)
if not processed_data.empty:
    # 정상 처리
    result = calculate_metrics(processed_data)
```

---

## ⚡ 성능 문제

### 느린 차트 렌더링

**문제**: `10초 이상 차트 로딩`

**해결 방법**:
```python
import plotly.graph_objects as go

# 데이터 샘플링
def optimize_chart_data(data, max_points=1000):
    """차트 최적화를 위한 데이터 샘플링"""
    if len(data) <= max_points:
        return data
    
    # 균등 샘플링
    step = len(data) // max_points
    return data.iloc[::step]

# 차트 최적화 설정
fig = go.Figure()
fig.update_layout(
    # 성능 최적화 옵션
    hovermode='closest',  # 호버 최적화
    showlegend=False,     # 범례 비활성화 (필요시)
    # 애니메이션 비활성화
    transition={'duration': 0},
    template="simple_white"  # 단순한 테마
)

# 차트 표시 최적화
st.plotly_chart(fig, 
                use_container_width=True,
                config={'displayModeBar': False})  # 툴바 숨김
```

### 메모리 사용량 증가

**문제**: `브라우저 탭 크래시`

**해결 방법**:
```python
import psutil
import gc

def monitor_memory():
    """메모리 사용량 모니터링"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        st.warning(f"⚠️ 메모리 사용량 높음: {memory_percent:.1f}%")
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # 캐시 일부 정리
        if memory_percent > 90:
            st.cache_data.clear()

# 주기적 모니터링
def main():
    monitor_memory()
    
    # 대용량 데이터 처리 시 청크 단위 처리
    def process_large_dataset(data, chunk_size=1000):
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            result = process_chunk(chunk)
            results.append(result)
            
            # 중간 메모리 정리
            if i % (chunk_size * 10) == 0:
                gc.collect()
        
        return pd.concat(results, ignore_index=True)
```

### 대용량 파일 처리

**문제**: `pandas.errors.OutOfMemoryError`

**해결 방법**:
```python
def load_large_csv(file_path, chunksize=10000):
    """대용량 CSV 파일을 청크 단위로 로드"""
    chunks = []
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # 필요한 컬럼만 선택
            chunk = chunk[['date', 'symbol', 'close', 'volume']]
            
            # 데이터 타입 최적화
            chunk['close'] = pd.to_numeric(chunk['close'], downcast='float')
            chunk['volume'] = pd.to_numeric(chunk['volume'], downcast='integer')
            
            chunks.append(chunk)
            
            # 진행 상황 표시
            if len(chunks) % 10 == 0:
                st.write(f"Loading... {len(chunks) * chunksize} rows")
        
        return pd.concat(chunks, ignore_index=True)
    
    except MemoryError:
        st.error("❌ 메모리 부족으로 파일을 로드할 수 없습니다.")
        st.info("💡 더 작은 청크 크기를 사용하거나 데이터를 분할하세요.")
        return pd.DataFrame()
```

---

## 🧪 테스트 관련 문제

### 테스트 실행 실패

**문제**: `pytest: command not found`

**해결 방법**:
```bash
# 가상환경에서 pytest 설치
pip install pytest pytest-cov

# 테스트 실행
pytest -v
pytest tests/test_data_loader.py -v  # 특정 모듈만
```

### 테스트 데이터 부족

**문제**: `FileNotFoundError: test_data.csv not found`

**해결 방법**:
```python
# tests/conftest.py에 픽스처 추가
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_returns_data():
    """샘플 수익률 데이터 생성"""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # 모의 수익률 생성
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(symbols))),
        index=dates,
        columns=symbols
    )
    return returns

# 테스트에서 사용
def test_portfolio_optimization(sample_returns_data):
    optimizer = MPTOptimizer()
    weights = optimizer.max_sharpe(mu, cov_matrix)
    assert len(weights) == len(sample_returns_data.columns)
```

### Coverage 리포트 오류

**문제**: `coverage: command not found`

**해결 방법**:
```bash
# Coverage 설치
pip install coverage

# 테스트 실행 및 리포트 생성
pytest --cov=krx_portfolio --cov-report=html
pytest --cov=krx_portfolio --cov-report=term-missing

# HTML 리포트 확인
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 🔧 고급 문제 해결

### 로깅 설정

```python
import logging

# 로깅 설정
def setup_logging(level=logging.INFO):
    """로깅 설정 함수"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('krx_portfolio.log'),
            logging.StreamHandler()
        ]
    )
    
    # 특정 모듈 로그 레벨 조정
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# 사용법
setup_logging(logging.DEBUG)  # 디버그 모드
logger = logging.getLogger(__name__)
logger.info("대시보드 시작")
```

### 디버깅 도구

```python
import streamlit as st
import traceback

# 전역 오류 핸들러
def handle_error(func):
    """데코레이터: 오류 처리 및 로깅"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            
            # 상세 오류 정보 (디버그 모드에서만)
            if st.session_state.get('debug_mode', False):
                st.code(traceback.format_exc())
            
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            return None
    return wrapper

# 사용법
@handle_error
def risky_function():
    # 위험한 연산
    result = 1 / 0  # ZeroDivisionError
    return result

# 디버그 모드 토글
if st.sidebar.checkbox("🐛 디버그 모드"):
    st.session_state.debug_mode = True
```

### 성능 프로파일링

```python
import cProfile
import pstats
import io
from contextlib import contextmanager

@contextmanager
def profile_code():
    """코드 프로파일링 컨텍스트 매니저"""
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        yield pr
    finally:
        pr.disable()
        
        # 결과 분석
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        # Streamlit에 표시
        st.text("🔍 성능 프로파일링 결과:")
        st.code(s.getvalue())

# 사용법
with profile_code():
    expensive_operation()
```

### 환경별 설정 관리

```python
import os
from typing import Dict, Any

class Config:
    """환경별 설정 관리"""
    
    def __init__(self):
        self.env = os.getenv('ENVIRONMENT', 'development')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """환경별 설정 로드"""
        base_config = {
            'debug': False,
            'cache_ttl': 3600,
            'max_data_points': 10000,
        }
        
        if self.env == 'development':
            base_config.update({
                'debug': True,
                'cache_ttl': 60,  # 개발 시 짧은 캐시
                'log_level': 'DEBUG'
            })
        elif self.env == 'production':
            base_config.update({
                'debug': False,
                'cache_ttl': 3600,
                'log_level': 'WARNING'
            })
        
        return base_config
    
    def get(self, key: str, default=None):
        """설정값 조회"""
        return self.config.get(key, default)

# 전역 설정 객체
config = Config()

# 사용법
if config.get('debug'):
    st.write("🐛 디버그 모드 활성화")
```

---

## 📞 추가 도움이 필요한 경우

### 1. GitHub Issues
- 새로운 버그 발견: [이슈 등록](https://github.com/your-repo/issues)
- 기능 요청: Feature Request 템플릿 사용

### 2. 로그 수집
문제 해결을 위해 다음 정보를 포함하여 이슈 등록:

```bash
# 시스템 정보
python --version
pip list | grep -E "(streamlit|pandas|numpy|plotly)"

# 오류 로그
tail -50 krx_portfolio.log

# 브라우저 개발자 도구 콘솔 오류
# F12 → Console 탭 → 오류 메시지 복사
```

### 3. 최소 재현 코드
```python
# 문제를 재현할 수 있는 최소한의 코드 작성
import streamlit as st
import pandas as pd

# 문제 상황 재현
data = pd.DataFrame({'A': [1, 2, None]})
st.write(data.mean())  # 오류 발생 지점
```

### 4. 임시 해결책
급한 경우 사용할 수 있는 임시 해결책:

```python
# 1. 샘플 데이터 사용
if data.empty:
    data = create_sample_portfolio_data()

# 2. 기본값 사용
metric_value = portfolio_data.get('total_return', 0.0)

# 3. try-except로 우회
try:
    result = risky_calculation()
except Exception:
    result = fallback_value
    st.warning("일부 계산에서 기본값을 사용했습니다.")
```

---

**📝 문서 버전**: v1.0.0  
**📅 마지막 업데이트**: 2025-08-29  
**✍️ 작성자**: KRX Dynamic Portfolio Team

**💡 팁**: 문제가 해결되지 않으면 `known_issues.md` 파일도 확인해보세요!