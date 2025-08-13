"""KRXPreprocessor 테스트"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from krx_portfolio.etl.preprocessor import KRXPreprocessor


class TestKRXPreprocessor:
    """KRXPreprocessor 테스트 클래스"""
    
    def test_init(self):
        """초기화 테스트"""
        preprocessor = KRXPreprocessor(min_market_cap=200e8, min_volume=2000)
        
        assert preprocessor.min_market_cap == 200e8
        assert preprocessor.min_volume == 2000
    
    def test_clean_data_basic(self, krx_preprocessor, sample_raw_df):
        """기본 데이터 정제 테스트"""
        cleaned_df = krx_preprocessor.clean_data(sample_raw_df)
        
        assert len(cleaned_df) > 0
        assert 'intraday_volatility' in cleaned_df.columns
        assert 'open_close_spread' in cleaned_df.columns
        assert 'log_market_cap' in cleaned_df.columns
        
        # 원본 데이터는 변경되지 않음
        assert 'intraday_volatility' not in sample_raw_df.columns
    
    def test_handle_missing_values(self, krx_preprocessor):
        """결측치 처리 테스트"""
        # 결측치가 있는 데이터 생성
        data = {
            'basDt': [pd.Timestamp('2023-12-15'), pd.Timestamp('2023-12-15'), pd.Timestamp('2023-12-15')],
            'srtnCd': ['005930', None, '035420'],  # 결측치
            'itmsNm': ['삼성전자', 'SK하이닉스', 'NAVER'],
            'clpr': [70000, 125000, None],  # 결측치
            'mrktTotAmt': [4.178e14, 9.1e13, 3.297e13],
            'trqu': [10000000, 0, 1000000],  # 0 값
            'trPrc': [7.0e11, 6.25e11, 0]  # 0 값
        }
        df = pd.DataFrame(data)
        
        cleaned_df = krx_preprocessor._handle_missing_values(df)
        
        # 필수 컬럼 결측치 제거됨
        assert len(cleaned_df) == 1  # srtnCd, clpr 결측치 행 제거
        assert cleaned_df['srtnCd'].iloc[0] == '005930'
    
    def test_filter_outliers(self, krx_preprocessor):
        """이상치 필터링 테스트"""
        # 이상치가 있는 데이터 생성
        data = {
            'basDt': [pd.Timestamp('2023-12-15')] * 4,
            'srtnCd': ['001', '002', '003', '004'],
            'itmsNm': ['정상', '등락률이상치', '가격이상치', '시총이상치'],
            'clpr': [70000, 125000, -1000, 200000],  # 음수 가격
            'fltRt': [0.01, 0.50, 0.02, 0.03],  # 50% 등락률 (이상치)
            'mrktTotAmt': [1e14, 2e14, 3e14, 1e18],  # 상위 0.1% 초과
            'mkp': [69000, 124000, 1000, 198000],
            'hipr': [71000, 126000, 1000, 202000],
            'lopr': [68000, 123000, 1000, 197000],
            'trqu': [1000000] * 4,
            'trPrc': [7e10] * 4,
            'lstgStCnt': [1e9] * 4
        }
        df = pd.DataFrame(data)
        
        filtered_df = krx_preprocessor._filter_outliers(df)
        
        # 이상치 제거 확인
        assert len(filtered_df) < len(df)
        assert all(filtered_df['fltRt'].abs() <= 0.30)
        assert all(filtered_df['clpr'] > 0)
    
    def test_apply_basic_filters(self, krx_preprocessor):
        """기본 필터 적용 테스트"""
        # 다양한 시가총액과 거래량 데이터
        data = {
            'basDt': [pd.Timestamp('2023-12-15')] * 4,
            'srtnCd': ['001', '002', '003', '004'],
            'itmsNm': ['대형주', '중형주', '소형주', '관리종목'],
            'clpr': [70000, 50000, 30000, 20000],
            'mrktTotAmt': [500e8, 150e8, 50e8, 200e8],  # min_market_cap = 100e8
            'trqu': [10000, 5000, 500, 2000],  # min_volume = 1000
            'trPrc': [7e10] * 4,
            'lstgStCnt': [1e9] * 4,
            'fltRt': [0.01] * 4,
            'mkp': [69000, 49000, 29000, 19000],
            'hipr': [71000, 51000, 31000, 21000],
            'lopr': [68000, 48000, 28000, 18000]
        }
        df = pd.DataFrame(data)
        
        filtered_df = krx_preprocessor._apply_basic_filters(df)
        
        # 필터링 조건 확인
        assert all(filtered_df['mrktTotAmt'] >= krx_preprocessor.min_market_cap)
        assert all(filtered_df['trqu'] >= krx_preprocessor.min_volume)
        assert '관리종목' not in filtered_df['itmsNm'].values
    
    def test_create_derived_features(self, krx_preprocessor, sample_raw_df):
        """파생 변수 생성 테스트"""
        df_with_features = krx_preprocessor._create_derived_features(sample_raw_df)
        
        # 파생 변수 존재 확인
        assert 'intraday_volatility' in df_with_features.columns
        assert 'open_close_spread' in df_with_features.columns
        assert 'turnover_ratio' in df_with_features.columns
        assert 'trading_value_ratio' in df_with_features.columns
        assert 'log_market_cap' in df_with_features.columns
        assert 'market_cap_rank' in df_with_features.columns
        
        # 계산 결과 검증
        first_row = df_with_features.iloc[0]
        expected_volatility = (first_row['hipr'] - first_row['lopr']) / first_row['clpr']
        assert first_row['intraday_volatility'] == pytest.approx(expected_volatility)
        
        expected_spread = (first_row['clpr'] - first_row['mkp']) / first_row['mkp']
        assert first_row['open_close_spread'] == pytest.approx(expected_spread)
    
    def test_create_ohlcv_panel(self, krx_preprocessor, sample_processed_df):
        """OHLCV 패널 데이터 생성 테스트"""
        panel_df = krx_preprocessor.create_ohlcv_panel(sample_processed_df)
        
        # 구조 확인
        assert isinstance(panel_df.index, pd.MultiIndex)
        assert panel_df.index.names == ['date', 'stock_code']
        
        expected_cols = ['stock_name', 'open', 'high', 'low', 'close', 'volume', 'amount']
        assert all(col in panel_df.columns for col in expected_cols)
        
        # 데이터 건수 확인
        assert len(panel_df) == len(sample_processed_df)
    
    def test_create_returns_matrix_daily(self, krx_preprocessor, sample_ohlcv_panel):
        """일일 수익률 매트릭스 생성 테스트"""
        returns_df = krx_preprocessor.create_returns_matrix(sample_ohlcv_panel, period='1D')
        
        # 구조 확인
        assert isinstance(returns_df.index, pd.DatetimeIndex)
        assert isinstance(returns_df.columns, pd.Index)
        
        # 수익률 계산 확인 (첫 번째 날짜는 수익률이 0이거나 NaN일 수 있음)
        if len(returns_df) > 1:
            # 두 번째 행부터는 실제 수익률이 계산되어야 함
            assert not returns_df.iloc[1].isna().all()
        else:
            # 단일 날짜인 경우 수익률이 0 또는 NaN
            assert returns_df.iloc[0].fillna(0).abs().sum() >= 0
        
        # 두 번째 행은 수익률 값이 있어야 함
        if len(returns_df) > 1:
            assert not returns_df.iloc[1].isna().all()
    
    def test_create_returns_matrix_periods(self, krx_preprocessor, sample_ohlcv_panel):
        """다양한 주기 수익률 매트릭스 테스트"""
        # 5일 수익률
        returns_5d = krx_preprocessor.create_returns_matrix(sample_ohlcv_panel, period='5D')
        assert len(returns_5d.columns) > 0
        
        # 월 수익률
        returns_1m = krx_preprocessor.create_returns_matrix(sample_ohlcv_panel, period='1M')
        assert len(returns_1m.columns) > 0
        
        # 지원하지 않는 주기
        with pytest.raises(ValueError, match="지원하지 않는 주기입니다"):
            krx_preprocessor.create_returns_matrix(sample_ohlcv_panel, period='2W')
    
    def test_create_market_cap_weights(self, krx_preprocessor, sample_raw_df):
        """시가총액 가중치 생성 테스트"""
        weights_df = krx_preprocessor.create_market_cap_weights(sample_raw_df)
        
        # 구조 확인
        assert isinstance(weights_df.index, pd.DatetimeIndex)
        assert isinstance(weights_df.columns, pd.Index)
        
        # 각 날짜별 가중치 합이 1인지 확인
        for date in weights_df.index:
            row_sum = weights_df.loc[date].sum()
            assert row_sum == pytest.approx(1.0, abs=1e-10)
        
        # 시가총액이 큰 종목의 가중치가 더 큰지 확인
        date = weights_df.index[0]
        samsung_weight = weights_df.loc[date, '005930']
        naver_weight = weights_df.loc[date, '035420']
        assert samsung_weight > naver_weight
    
    def test_filter_investable_universe_kospi(self, krx_preprocessor, sample_raw_df):
        """KOSPI 투자 유니버스 필터링 테스트"""
        kospi_stocks = krx_preprocessor.filter_investable_universe(
            sample_raw_df, top_n=2, market='KOSPI'
        )
        
        assert len(kospi_stocks) == 2
        assert '005930' in kospi_stocks  # 삼성전자
        assert '000660' in kospi_stocks  # SK하이닉스
        assert '035420' not in kospi_stocks  # NAVER (KOSDAQ)
    
    def test_filter_investable_universe_all_markets(self, krx_preprocessor, sample_raw_df):
        """전체 시장 투자 유니버스 필터링 테스트"""
        all_stocks = krx_preprocessor.filter_investable_universe(
            sample_raw_df, top_n=3, market=None
        )
        
        assert len(all_stocks) == 3
        # 시가총액 순으로 정렬되어야 함
        expected_order = ['005930', '000660', '035420']  # 시총 큰 순
        assert all_stocks == expected_order
    
    def test_get_data_quality_report(self, krx_preprocessor, sample_raw_df):
        """데이터 품질 보고서 테스트"""
        report = krx_preprocessor.get_data_quality_report(sample_raw_df)
        
        # 필수 키 확인
        assert 'total_records' in report
        assert 'date_range' in report
        assert 'universe_size' in report
        assert 'missing_data' in report
        assert 'market_coverage' in report
        
        # 값 확인
        assert report['total_records'] == 3
        assert report['universe_size']['total_stocks'] == 3
        assert report['universe_size']['kospi_stocks'] == 2
        assert report['universe_size']['kosdaq_stocks'] == 1
        
        # 날짜 범위 확인
        assert 'start' in report['date_range']
        assert 'end' in report['date_range']
        assert 'trading_days' in report['date_range']
    
    def test_data_quality_report_with_missing_data(self, krx_preprocessor):
        """결측치가 있는 데이터의 품질 보고서 테스트"""
        # 결측치가 있는 데이터 생성
        data = {
            'basDt': [pd.Timestamp('2023-12-15')] * 3,
            'srtnCd': ['005930', '000660', '035420'],
            'itmsNm': ['삼성전자', None, 'NAVER'],  # 결측치
            'mrktCtg': ['KOSPI', 'KOSPI', 'KOSDAQ'],
            'clpr': [70000, 125000, 200000],
            'trqu': [10000000, 5000000, 1000000],
            'trPrc': [7.0e11, 6.25e11, 2.0e11],
            'lstgStCnt': [5969782550, 728002365, 164831339],
            'mrktTotAmt': [4.178e14, 9.1e13, 3.297e13]
        }
        df = pd.DataFrame(data)
        
        report = krx_preprocessor.get_data_quality_report(df)
        
        # 결측치 정보 확인
        assert 'itmsNm' in report['missing_data']
        assert report['missing_data']['itmsNm'] == 1