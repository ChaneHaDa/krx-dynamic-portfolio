"""KRX JSON 데이터 로더"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class KRXDataLoader:
    """KRX JSON 데이터 로더
    
    /path/to/krx-json-data/Price/STOCK/YYYY/YYYYMMDD.json 형태의
    KRX JSON 파일을 pandas DataFrame으로 로드
    """
    
    def __init__(self, data_root: Union[str, Path]):
        """
        Args:
            data_root: KRX JSON 데이터 루트 경로
                      예: "/home/chan/code/vscode/python3/krx-json-data"
        """
        self.data_root = Path(data_root)
        self.price_path = self.data_root / "Price" / "STOCK"
    
    def load_single_date(self, date: Union[str, datetime]) -> pd.DataFrame:
        """단일 날짜의 주식 데이터를 로드
        
        Args:
            date: 날짜 (YYYYMMDD 또는 datetime)
            
        Returns:
            해당 날짜의 주식 데이터 DataFrame
        """
        if isinstance(date, datetime):
            date_str = date.strftime("%Y%m%d")
            year = date.year
        else:
            date_str = str(date)
            year = int(date_str[:4])
        
        file_path = self.price_path / str(year) / f"{date_str}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 구조에서 실제 데이터 추출
        items = data['response']['body']['items']['item']
        
        # DataFrame 변환
        df = pd.DataFrame(items)
        
        # 컬럼 타입 변환
        df = self._convert_dtypes(df)
        
        return df
    
    def load_date_range(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """날짜 범위의 주식 데이터를 로드
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            기간 내 모든 주식 데이터를 합친 DataFrame
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y%m%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d")
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                daily_data = self.load_single_date(current_date)
                all_data.append(daily_data)
                print(f"로드 완료: {current_date.strftime('%Y%m%d')} ({len(daily_data)} 종목)")
            except FileNotFoundError:
                print(f"파일 없음: {current_date.strftime('%Y%m%d')} (건너뜀)")
            
            current_date += timedelta(days=1)
        
        if not all_data:
            raise ValueError(f"지정된 기간에 데이터가 없습니다: {start_date} ~ {end_date}")
        
        # 모든 데이터 합치기
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 날짜별 정렬
        combined_df = combined_df.sort_values(['basDt', 'srtnCd']).reset_index(drop=True)
        
        return combined_df
    
    def load_latest_available(self, days_back: int = 30) -> pd.DataFrame:
        """최근 사용 가능한 데이터를 로드
        
        Args:
            days_back: 과거 몇 일까지 검색할지
            
        Returns:
            가장 최근 사용 가능한 날짜의 주식 데이터
        """
        today = datetime.now()
        
        for i in range(days_back):
            check_date = today - timedelta(days=i)
            try:
                return self.load_single_date(check_date)
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError(f"최근 {days_back}일 내 사용 가능한 데이터가 없습니다")
    
    def get_available_dates(self, year: Optional[int] = None) -> List[str]:
        """사용 가능한 날짜 목록을 반환
        
        Args:
            year: 특정 연도만 조회 (None이면 모든 연도)
            
        Returns:
            사용 가능한 날짜 목록 (YYYYMMDD 형태)
        """
        available_dates = []
        
        if year:
            years = [year]
        else:
            years = [int(p.name) for p in self.price_path.iterdir() if p.is_dir()]
            years.sort()
        
        for year in years:
            year_path = self.price_path / str(year)
            if year_path.exists():
                json_files = list(year_path.glob("*.json"))
                dates = [f.stem for f in json_files]
                available_dates.extend(dates)
        
        return sorted(available_dates)
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 변환"""
        # 날짜 변환
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        
        # 수치형 컬럼들
        numeric_cols = [
            'clpr', 'vs', 'fltRt', 'mkp', 'hipr', 'lopr',
            'trqu', 'trPrc', 'lstgStCnt', 'mrktTotAmt'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # 쉼표 제거 후 숫자 변환
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # 등락률은 float으로 명시적 변환
        if 'fltRt' in df.columns:
            df['fltRt'] = df['fltRt'] / 100.0  # 퍼센트를 소수로 변환
        
        # 카테고리형
        if 'mrktCtg' in df.columns:
            df['mrktCtg'] = df['mrktCtg'].astype('category')
        
        return df
    
    def get_market_summary(self, date: Union[str, datetime]) -> Dict:
        """시장 전체 요약 통계
        
        Args:
            date: 조회 날짜
            
        Returns:
            시장 요약 정보
        """
        df = self.load_single_date(date)
        
        kospi_df = df[df['mrktCtg'] == 'KOSPI']
        kosdaq_df = df[df['mrktCtg'] == 'KOSDAQ']
        
        summary = {
            'date': df['basDt'].iloc[0],
            'total_stocks': len(df),
            'kospi': {
                'count': len(kospi_df),
                'total_market_cap': kospi_df['mrktTotAmt'].sum() if len(kospi_df) > 0 else 0,
                'total_volume': kospi_df['trqu'].sum() if len(kospi_df) > 0 else 0,
            },
            'kosdaq': {
                'count': len(kosdaq_df),
                'total_market_cap': kosdaq_df['mrktTotAmt'].sum() if len(kosdaq_df) > 0 else 0,
                'total_volume': kosdaq_df['trqu'].sum() if len(kosdaq_df) > 0 else 0,
            }
        }
        
        return summary