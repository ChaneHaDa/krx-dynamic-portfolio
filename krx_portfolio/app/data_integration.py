"""
실시간 데이터 연동 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@st.cache_data(ttl=1800, max_entries=50, show_spinner=False)  # 30분 캐시, 최대 50개 엔트리
def fetch_real_time_data(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    """
    yfinance를 사용하여 실시간 주식 데이터 수집
    
    Parameters
    ----------
    symbols : List[str]
        주식 심볼 리스트 (예: ['005930.KS', '035420.KS'])
    period : str
        데이터 기간 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns
    -------
    pd.DataFrame
        OHLCV 데이터 (MultiIndex: Date x Symbol x OHLCV)
    """
    try:
        # yfinance로 데이터 다운로드
        data = yf.download(symbols, period=period, group_by='ticker', progress=False)
        
        if len(symbols) == 1:
            # 단일 종목인 경우 MultiIndex 생성
            symbol = symbols[0]
            data.columns = pd.MultiIndex.from_product([[symbol], data.columns])
        
        return data
        
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=20, show_spinner=False)  # 5분 캐시, 실시간 가격용
def get_current_prices(symbols: List[str]) -> pd.Series:
    """
    현재 주식 가격 조회
    
    Parameters
    ----------
    symbols : List[str]
        주식 심볼 리스트
    
    Returns
    -------
    pd.Series
        현재 가격 시리즈
    """
    try:
        tickers = yf.Tickers(' '.join(symbols))
        current_prices = {}
        
        for symbol in symbols:
            ticker = tickers.tickers[symbol]
            info = ticker.info
            current_prices[symbol] = info.get('regularMarketPrice', info.get('currentPrice', np.nan))
        
        return pd.Series(current_prices)
        
    except Exception as e:
        logger.error(f"현재 가격 조회 실패: {e}")
        return pd.Series(dtype=float)


def krx_symbol_to_yfinance(krx_code: str) -> str:
    """
    KRX 종목코드를 yfinance 심볼로 변환
    
    Parameters
    ----------
    krx_code : str
        KRX 종목코드 (예: '005930')
    
    Returns
    -------
    str
        yfinance 심볼 (예: '005930.KS')
    """
    # KOSPI 종목은 .KS, KOSDAQ 종목은 .KQ 접미사 추가
    # 간단한 구현으로 모든 종목을 .KS로 처리 (실제로는 KOSPI/KOSDAQ 구분 필요)
    return f"{krx_code}.KS"


def get_korean_stock_info(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    한국 주식 정보 조회
    
    Parameters
    ----------
    symbols : List[str]
        yfinance 심볼 리스트
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        주식 정보 딕셔너리
    """
    stock_info = {}
    
    try:
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info[symbol] = {
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', info.get('trailingPE', np.nan)),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'currency': info.get('currency', 'KRW'),
                'exchange': info.get('exchange', 'KRX')
            }
            
    except Exception as e:
        logger.error(f"주식 정보 조회 실패: {e}")
    
    return stock_info


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    가격 데이터로부터 수익률 계산
    
    Parameters
    ----------
    prices : pd.DataFrame
        가격 데이터
    method : str
        수익률 계산 방법 ('simple', 'log')
    
    Returns
    -------
    pd.DataFrame
        수익률 데이터
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:  # simple returns
        return (prices / prices.shift(1) - 1).dropna()


def get_market_indices(period: str = "1y") -> pd.DataFrame:
    """
    주요 시장 지수 데이터 조회
    
    Parameters
    ----------
    period : str
        데이터 기간
    
    Returns
    -------
    pd.DataFrame
        시장 지수 데이터
    """
    indices = {
        'KOSPI': '^KS11',
        'KOSDAQ': '^KQ11', 
        'S&P500': '^GSPC',
        'NASDAQ': '^IXIC'
    }
    
    try:
        index_data = {}
        for name, symbol in indices.items():
            data = yf.download(symbol, period=period, progress=False)['Adj Close']
            index_data[name] = data
        
        return pd.DataFrame(index_data).dropna()
        
    except Exception as e:
        logger.error(f"지수 데이터 조회 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200, max_entries=10)  # 2시간 캐시, 샘플 데이터용
def create_sample_portfolio_data(n_assets: int = 20) -> Dict[str, Any]:
    """
    샘플 포트폴리오 데이터 생성 (실제 데이터가 없을 때 사용)
    
    Parameters
    ----------
    n_assets : int
        자산 개수
    
    Returns
    -------
    Dict[str, Any]
        샘플 포트폴리오 데이터
    """
    np.random.seed(42)
    
    # 가상의 KRX 종목 코드들
    sample_codes = [f"{i:06d}" for i in range(5000, 5000 + n_assets)]
    
    # 가격 데이터 생성 (1년간 일간 데이터)
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), 
                         end=datetime.now(), 
                         freq='D')
    
    # 각 종목별 가격 시뮬레이션 (기하 브라운 운동)
    price_data = {}
    for code in sample_codes:
        initial_price = np.random.uniform(10000, 100000)  # 초기 가격
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 일간 수익률
        prices = initial_price * np.exp(np.cumsum(returns))
        price_data[code] = prices
    
    prices_df = pd.DataFrame(price_data, index=dates)
    returns_df = calculate_returns(prices_df)
    
    # 포트폴리오 가중치 (랜덤)
    weights = np.random.dirichlet(np.ones(n_assets))
    
    # 섹터 정보
    sectors = ['기술', '금융', '소비재', '헬스케어', '산업재', '에너지', '유틸리티']
    sector_map = {code: np.random.choice(sectors) for code in sample_codes}
    
    # 포트폴리오 성과 계산
    portfolio_returns = (returns_df * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return {
        'prices': prices_df,
        'returns': returns_df,
        'weights': pd.Series(weights, index=sample_codes),
        'sector_map': sector_map,
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'total_value': 100000000,  # 1억원
        'daily_change_pct': portfolio_returns.iloc[-1] * 100,
        'num_holdings': n_assets,
        'expected_return': portfolio_returns.mean() * 252 * 100,
        'volatility': portfolio_returns.std() * np.sqrt(252) * 100
    }


def validate_symbols(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    yfinance 심볼 유효성 검증
    
    Parameters
    ----------
    symbols : List[str]
        검증할 심볼 리스트
    
    Returns
    -------
    Tuple[List[str], List[str]]
        (유효한 심볼 리스트, 무효한 심볼 리스트)
    """
    valid_symbols = []
    invalid_symbols = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # 간단한 데이터 요청으로 유효성 확인
            hist = ticker.history(period="5d")
            if not hist.empty:
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
        except:
            invalid_symbols.append(symbol)
    
    return valid_symbols, invalid_symbols


@st.cache_data(ttl=60, max_entries=5, show_spinner=False)  # 1분 캐시, 시장 상태용
def get_real_time_market_status() -> Dict[str, Any]:
    """
    실시간 시장 현황 조회
    
    Returns
    -------
    Dict[str, Any]
        시장 현황 정보
    """
    try:
        # KOSPI 지수 조회
        kospi = yf.Ticker("^KS11")
        kospi_data = kospi.history(period="2d")
        
        if len(kospi_data) >= 2:
            current_price = kospi_data['Close'].iloc[-1]
            prev_price = kospi_data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
        else:
            current_price = prev_price = change = change_pct = 0
        
        # 시장 시간 확인 (한국 시간 기준)
        now = datetime.now()
        is_trading_hours = (9 <= now.hour < 15) and (now.weekday() < 5)
        
        return {
            'kospi_current': current_price,
            'kospi_change': change,
            'kospi_change_pct': change_pct,
            'is_trading_hours': is_trading_hours,
            'last_update': now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"시장 현황 조회 실패: {e}")
        return {
            'kospi_current': 0,
            'kospi_change': 0, 
            'kospi_change_pct': 0,
            'is_trading_hours': False,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }