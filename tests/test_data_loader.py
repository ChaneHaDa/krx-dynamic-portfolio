"""KRXDataLoader 테스트"""

import pytest
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

from krx_portfolio.etl.data_loader import KRXDataLoader


class TestKRXDataLoader:
    """KRXDataLoader 테스트 클래스"""
    
    def test_init(self, temp_data_dir):
        """초기화 테스트"""
        loader = KRXDataLoader(temp_data_dir)
        
        assert loader.data_root == temp_data_dir
        assert loader.price_path == temp_data_dir / "Price" / "STOCK"
    
    def test_load_single_date_success(self, krx_loader):
        """단일 날짜 로드 성공 테스트"""
        df = krx_loader.load_single_date("20231215")
        
        assert len(df) == 3
        assert 'basDt' in df.columns
        assert 'srtnCd' in df.columns
        assert 'clpr' in df.columns
        
        # 데이터 타입 확인
        assert pd.api.types.is_datetime64_any_dtype(df['basDt'])
        assert pd.api.types.is_numeric_dtype(df['clpr'])
        assert isinstance(df['mrktCtg'].dtype, pd.CategoricalDtype)
        
        # 등락률 변환 확인 (퍼센트 → 소수)
        assert df['fltRt'].iloc[0] == pytest.approx(0.0145, abs=1e-4)
    
    def test_load_single_date_datetime_input(self, krx_loader):
        """datetime 입력으로 단일 날짜 로드 테스트"""
        date = datetime(2023, 12, 15)
        df = krx_loader.load_single_date(date)
        
        assert len(df) == 3
        assert df['basDt'].iloc[0] == pd.Timestamp('2023-12-15')
    
    def test_load_single_date_file_not_found(self, krx_loader):
        """존재하지 않는 날짜 로드 실패 테스트"""
        with pytest.raises(FileNotFoundError, match="데이터 파일을 찾을 수 없습니다"):
            krx_loader.load_single_date("20231201")
    
    def test_load_date_range_success(self, krx_loader):
        """날짜 범위 로드 성공 테스트"""
        df = krx_loader.load_date_range("20231215", "20231215")
        
        assert len(df) == 3
        assert df['basDt'].iloc[0] == pd.Timestamp('2023-12-15')
    
    def test_load_date_range_with_missing_files(self, krx_loader, capsys):
        """누락된 파일이 있는 날짜 범위 로드 테스트"""
        # 20231214는 존재하지 않는 파일
        df = krx_loader.load_date_range("20231214", "20231215")
        
        # 출력 메시지 확인
        captured = capsys.readouterr()
        assert "파일 없음: 20231214" in captured.out
        assert "로드 완료: 20231215" in captured.out
        
        # 존재하는 날짜의 데이터만 로드됨
        assert len(df) == 3
    
    def test_load_date_range_no_data(self, krx_loader):
        """데이터가 없는 날짜 범위 로드 실패 테스트"""
        with pytest.raises(ValueError, match="지정된 기간에 데이터가 없습니다"):
            krx_loader.load_date_range("20231201", "20231205")
    
    def test_load_latest_available_success(self, krx_loader):
        """최근 사용 가능한 데이터 로드 성공 테스트"""
        df = krx_loader.load_latest_available(days_back=30)
        
        assert len(df) == 3
        assert 'basDt' in df.columns
    
    def test_load_latest_available_no_data(self, temp_data_dir):
        """최근 데이터 없는 경우 실패 테스트"""
        # 빈 데이터 디렉토리로 새 로더 생성
        empty_loader = KRXDataLoader(temp_data_dir / "empty")
        with pytest.raises(FileNotFoundError, match="최근 1일 내 사용 가능한 데이터가 없습니다"):
            empty_loader.load_latest_available(days_back=1)
    
    def test_get_available_dates_all_years(self, krx_loader):
        """전체 연도 사용 가능한 날짜 조회 테스트"""
        dates = krx_loader.get_available_dates()
        
        assert "20231215" in dates
        assert "20231216" in dates
        assert len(dates) >= 2  # 최소 2개 이상 (현재 날짜 포함 가능)
        assert dates == sorted(dates)  # 정렬 확인
    
    def test_get_available_dates_specific_year(self, krx_loader):
        """특정 연도 사용 가능한 날짜 조회 테스트"""
        dates = krx_loader.get_available_dates(year=2023)
        
        assert "20231215" in dates
        assert "20231216" in dates
        assert len(dates) == 2
    
    def test_get_available_dates_nonexistent_year(self, krx_loader):
        """존재하지 않는 연도 조회 테스트"""
        dates = krx_loader.get_available_dates(year=2022)
        
        assert len(dates) == 0
    
    def test_convert_dtypes(self, krx_loader):
        """데이터 타입 변환 테스트"""
        # 원시 데이터 로드
        df = krx_loader.load_single_date("20231215")
        
        # 날짜 타입 확인
        assert pd.api.types.is_datetime64_any_dtype(df['basDt'])
        
        # 수치형 타입 확인
        numeric_cols = ['clpr', 'vs', 'mkp', 'hipr', 'lopr', 'trqu', 'trPrc', 'lstgStCnt', 'mrktTotAmt']
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])
        
        # 등락률 변환 확인 (1.45% → 0.0145)
        assert df['fltRt'].iloc[0] == pytest.approx(0.0145, abs=1e-4)
        
        # 카테고리 타입 확인
        assert isinstance(df['mrktCtg'].dtype, pd.CategoricalDtype)
    
    def test_get_market_summary(self, krx_loader):
        """시장 요약 통계 테스트"""
        summary = krx_loader.get_market_summary("20231215")
        
        assert 'date' in summary
        assert 'total_stocks' in summary
        assert 'kospi' in summary
        assert 'kosdaq' in summary
        
        assert summary['total_stocks'] == 3
        assert summary['kospi']['count'] == 2
        assert summary['kosdaq']['count'] == 1
        
        # KOSPI 시가총액이 더 큰지 확인
        assert summary['kospi']['total_market_cap'] > summary['kosdaq']['total_market_cap']
    
    def test_numeric_conversion_with_commas(self, temp_data_dir):
        """쉼표가 포함된 숫자 변환 테스트"""
        # 쉼표가 포함된 데이터 생성
        data_with_commas = {
            "response": {
                "body": {
                    "items": {
                        "item": [{
                            "basDt": "20231215",
                            "srtnCd": "005930",
                            "itmsNm": "삼성전자",
                            "mrktCtg": "KOSPI",
                            "clpr": "70,000",  # 쉼표 포함
                            "vs": "1,000",
                            "fltRt": "1.45",
                            "mkp": "69,500",
                            "hipr": "70,500",
                            "lopr": "69,000",
                            "trqu": "10,000,000",
                            "trPrc": "700,000,000,000",
                            "lstgStCnt": "5,969,782,550",
                            "mrktTotAmt": "417,834,978,500,000"
                        }]
                    }
                }
            }
        }
        
        # 파일 생성
        price_dir = temp_data_dir / "Price" / "STOCK" / "2023"
        price_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = price_dir / "20231215.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_with_commas, f, ensure_ascii=False)
        
        # 로더 테스트
        loader = KRXDataLoader(temp_data_dir)
        df = loader.load_single_date("20231215")
        
        # 쉼표가 제거되고 숫자로 변환되었는지 확인
        assert df['clpr'].iloc[0] == 70000
        assert df['trqu'].iloc[0] == 10000000