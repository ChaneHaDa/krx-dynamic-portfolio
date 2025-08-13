"""Pytest 설정 및 공통 픽스처"""

import json
import pytest
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from krx_portfolio.etl.data_loader import KRXDataLoader
from krx_portfolio.etl.preprocessor import KRXPreprocessor


@pytest.fixture
def sample_krx_data() -> Dict:
    """샘플 KRX JSON 데이터"""
    return {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "basDt": "20231215",
                            "srtnCd": "005930",
                            "itmsNm": "삼성전자",
                            "mrktCtg": "KOSPI",
                            "clpr": "70000",
                            "vs": "1000",
                            "fltRt": "1.45",
                            "mkp": "69500",
                            "hipr": "70500",
                            "lopr": "69000",
                            "trqu": "10000000",
                            "trPrc": "700000000000",
                            "lstgStCnt": "5969782550",
                            "mrktTotAmt": "417834978500000"
                        },
                        {
                            "basDt": "20231215",
                            "srtnCd": "000660",
                            "itmsNm": "SK하이닉스",
                            "mrktCtg": "KOSPI",
                            "clpr": "125000",
                            "vs": "-2000",
                            "fltRt": "-1.57",
                            "mkp": "127000",
                            "hipr": "128000",
                            "lopr": "124500",
                            "trqu": "5000000",
                            "trPrc": "625000000000",
                            "lstgStCnt": "728002365",
                            "mrktTotAmt": "91000295625000"
                        },
                        {
                            "basDt": "20231215",
                            "srtnCd": "035420",
                            "itmsNm": "NAVER",
                            "mrktCtg": "KOSDAQ",
                            "clpr": "200000",
                            "vs": "3000",
                            "fltRt": "1.52",
                            "mkp": "198000",
                            "hipr": "201000",
                            "lopr": "197500",
                            "trqu": "1000000",
                            "trPrc": "200000000000",
                            "lstgStCnt": "164831339",
                            "mrktTotAmt": "32966267800000"
                        }
                    ]
                }
            }
        }
    }


@pytest.fixture
def temp_data_dir(sample_krx_data):
    """임시 데이터 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # KRX 데이터 구조 생성
        price_dir = temp_path / "Price" / "STOCK" / "2023"
        price_dir.mkdir(parents=True)
        
        # 샘플 JSON 파일 생성
        json_file = price_dir / "20231215.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_krx_data, f, ensure_ascii=False)
        
        # 최근 날짜에 동일한 데이터 생성 (load_latest_available 테스트용)
        from datetime import datetime
        recent_date = datetime.now().strftime("%Y%m%d")
        recent_year_dir = temp_path / "Price" / "STOCK" / recent_date[:4]
        recent_year_dir.mkdir(parents=True, exist_ok=True)
        recent_file = recent_year_dir / f"{recent_date}.json"
        with open(recent_file, 'w', encoding='utf-8') as f:
            json.dump(sample_krx_data, f, ensure_ascii=False)
        
        # 추가 날짜 데이터 생성 (빈 거래일)
        empty_data = {
            "response": {
                "body": {
                    "items": {
                        "item": []
                    }
                }
            }
        }
        
        empty_file = price_dir / "20231216.json"  # 토요일 - 거래 없음
        with open(empty_file, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f, ensure_ascii=False)
        
        yield temp_path


@pytest.fixture
def krx_loader(temp_data_dir):
    """KRXDataLoader 인스턴스"""
    return KRXDataLoader(temp_data_dir)


@pytest.fixture
def krx_preprocessor():
    """KRXPreprocessor 인스턴스"""
    return KRXPreprocessor(min_market_cap=100e8, min_volume=1000)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """전처리 전 원시 데이터프레임"""
    data = {
        'basDt': pd.to_datetime(['2023-12-15', '2023-12-15', '2023-12-15']),
        'srtnCd': ['005930', '000660', '035420'],
        'itmsNm': ['삼성전자', 'SK하이닉스', 'NAVER'],
        'mrktCtg': ['KOSPI', 'KOSPI', 'KOSDAQ'],
        'clpr': [70000, 125000, 200000],
        'vs': [1000, -2000, 3000],
        'fltRt': [0.0145, -0.0157, 0.0152],
        'mkp': [69500, 127000, 198000],
        'hipr': [70500, 128000, 201000],
        'lopr': [69000, 124500, 197500],
        'trqu': [10000000, 5000000, 1000000],
        'trPrc': [7.0e11, 6.25e11, 2.0e11],
        'lstgStCnt': [5969782550, 728002365, 164831339],
        'mrktTotAmt': [4.178e14, 9.1e13, 3.297e13]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_df(sample_raw_df) -> pd.DataFrame:
    """전처리 후 데이터프레임"""
    df = sample_raw_df.copy()
    
    # 파생 변수 추가
    df['intraday_volatility'] = (df['hipr'] - df['lopr']) / df['clpr']
    df['open_close_spread'] = (df['clpr'] - df['mkp']) / df['mkp']
    df['turnover_ratio'] = df['trqu'] / df['lstgStCnt']
    df['trading_value_ratio'] = df['trPrc'] / df['mrktTotAmt']
    df['log_market_cap'] = df['mrktTotAmt'].apply(lambda x: pd.Series([x]).apply('log').iloc[0])
    df['market_cap_rank'] = df['mrktTotAmt'].rank(pct=True)
    
    return df


@pytest.fixture
def temp_cache_dir():
    """임시 캐시 디렉토리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir)
        
        # 캐시 구조 생성
        (cache_path / "raw").mkdir()
        (cache_path / "processed").mkdir()
        (cache_path / "features").mkdir()
        
        yield cache_path


@pytest.fixture
def sample_ohlcv_panel() -> pd.DataFrame:
    """OHLCV 패널 데이터"""
    dates = pd.date_range('2023-12-14', '2023-12-15', freq='D')
    stocks = ['005930', '000660']
    
    data = []
    for date in dates:
        for stock in stocks:
            base_price = 70000 if stock == '005930' else 125000
            data.append({
                'date': date,
                'stock_code': stock,
                'stock_name': '삼성전자' if stock == '005930' else 'SK하이닉스',
                'open': base_price * 0.99,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': 10000000 if stock == '005930' else 5000000,
                'amount': base_price * (10000000 if stock == '005930' else 5000000)
            })
    
    df = pd.DataFrame(data)
    return df.set_index(['date', 'stock_code'])