"""KRX 데이터 전처리 모듈"""

from typing import Optional

import numpy as np
import pandas as pd


class KRXPreprocessor:
    """KRX 데이터 전처리 클래스

    원시 KRX JSON 데이터를 포트폴리오 분석에 적합한 형태로 전처리
    """

    def __init__(self, min_market_cap: float = 100e8, min_volume: int = 1000):
        """
        Args:
            min_market_cap: 최소 시가총액 필터 (단위: 원)
            min_volume: 최소 거래량 필터 (단위: 주)
        """
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 데이터 정제

        Args:
            df: 원시 KRX 데이터프레임

        Returns:
            정제된 데이터프레임
        """
        df = df.copy()

        # 1. 결측치 처리
        df = self._handle_missing_values(df)

        # 2. 이상치 필터링
        df = self._filter_outliers(df)

        # 3. 기본 필터 적용
        df = self._apply_basic_filters(df)

        # 4. 파생 변수 생성
        df = self._create_derived_features(df)

        return df

    def create_ohlcv_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV 패널 데이터 생성

        시계열 분석을 위한 종목별 OHLCV 데이터 구조화

        Args:
            df: 정제된 KRX 데이터프레임

        Returns:
            인덱스: (date, stock_code), 컬럼: OHLCV 데이터
        """
        ohlcv_cols = ["mkp", "hipr", "lopr", "clpr", "trqu", "trPrc"]

        # 필요한 컬럼만 선택
        panel_df = df[["basDt", "srtnCd", "itmsNm"] + ohlcv_cols].copy()

        # 컬럼명 표준화
        panel_df.columns = [
            "date",
            "stock_code",
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]

        # 멀티인덱스 설정
        panel_df = panel_df.set_index(["date", "stock_code"])

        return panel_df

    def create_returns_matrix(
        self, df: pd.DataFrame, period: str = "1D"
    ) -> pd.DataFrame:
        """수익률 매트릭스 생성

        포트폴리오 최적화에 사용할 수익률 매트릭스 생성

        Args:
            df: OHLCV 패널 데이터
            period: 수익률 계산 주기 ('1D', '5D', '1M')

        Returns:
            인덱스: 날짜, 컬럼: 종목코드, 값: 수익률
        """
        # 종가 데이터만 추출
        close_prices = df.reset_index().pivot(
            index="date", columns="stock_code", values="close"
        )

        # 수익률 계산
        if period == "1D":
            returns = close_prices.pct_change()
        elif period == "5D":
            returns = close_prices.pct_change(periods=5)
        elif period == "1M":
            returns = close_prices.pct_change(periods=20)  # 약 1개월 (20 거래일)
        else:
            raise ValueError(f"지원하지 않는 주기입니다: {period}")

        return returns.dropna()

    def create_market_cap_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """시가총액 기준 가중치 생성

        Args:
            df: 원시 KRX 데이터프레임

        Returns:
            인덱스: 날짜, 컬럼: 종목코드, 값: 시가총액 비중
        """
        # 시가총액 데이터 추출
        market_cap = df.pivot(index="basDt", columns="srtnCd", values="mrktTotAmt")

        # 각 날짜별로 전체 시가총액 대비 비중 계산
        weights = market_cap.div(market_cap.sum(axis=1), axis=0)

        return weights.fillna(0)

    def filter_investable_universe(
        self, df: pd.DataFrame, top_n: int = 200, market: Optional[str] = None
    ) -> list[str]:
        """투자 유니버스 필터링

        Args:
            df: 원시 KRX 데이터프레임
            top_n: 상위 N개 종목
            market: 시장 필터 ('KOSPI', 'KOSDAQ', None)

        Returns:
            선별된 종목코드 리스트
        """
        # 최신 날짜 데이터 사용
        latest_date = df["basDt"].max()
        latest_df = df[df["basDt"] == latest_date].copy()

        # 시장 필터
        if market:
            latest_df = latest_df[latest_df["mrktCtg"] == market]

        # 시가총액 기준 상위 N개 선택
        top_stocks = latest_df.nlargest(top_n, "mrktTotAmt")

        return top_stocks["srtnCd"].tolist()

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        df = df.copy()

        # 필수 컬럼 결측치 제거
        essential_cols = ["srtnCd", "basDt", "clpr", "mrktTotAmt"]
        df = df.dropna(subset=essential_cols)

        # 거래량/거래대금 0인 경우 결측 처리 후 전진/후진 보간
        volume_cols = ["trqu", "trPrc"]
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df.groupby("srtnCd")[col].ffill().bfill()

        return df

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상치 필터링"""
        df = df.copy()

        # 등락률 이상치 제거 (±30% 초과)
        if "fltRt" in df.columns:
            df = df[df["fltRt"].abs() <= 0.30]

        # 가격 이상치 제거 (0 이하)
        price_cols = ["clpr", "mkp", "hipr", "lopr"]
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # 시가총액 상위 0.1% 이상치 제거 (초대형주 제외)
        if "mrktTotAmt" in df.columns:
            cap_99_9 = df["mrktTotAmt"].quantile(0.999)
            df = df[df["mrktTotAmt"] <= cap_99_9]

        return df

    def _apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 필터 적용"""
        df = df.copy()

        # 최소 시가총액 필터
        if "mrktTotAmt" in df.columns:
            df = df[df["mrktTotAmt"] >= self.min_market_cap]

        # 최소 거래량 필터
        if "trqu" in df.columns:
            df = df[df["trqu"] >= self.min_volume]

        # 관리종목, 정리매매 제외 (종목명으로 판단)
        if "itmsNm" in df.columns:
            # 관리종목 표시 제외
            df = df[~df["itmsNm"].str.contains("관리", na=False)]
            # 정리매매 표시 제외
            df = df[~df["itmsNm"].str.contains("정리", na=False)]

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        df = df.copy()

        # 1. 가격 기반 지표
        if all(col in df.columns for col in ["hipr", "lopr", "clpr"]):
            # 일중 변동성
            df["intraday_volatility"] = (df["hipr"] - df["lopr"]) / df["clpr"]

            # 시가-종가 스프레드
            if "mkp" in df.columns:
                df["open_close_spread"] = (df["clpr"] - df["mkp"]) / df["mkp"]

        # 2. 거래량 기반 지표
        if all(col in df.columns for col in ["trqu", "lstgStCnt"]):
            # 거래회전율
            df["turnover_ratio"] = df["trqu"] / df["lstgStCnt"]

        if all(col in df.columns for col in ["trPrc", "mrktTotAmt"]):
            # 거래대금 비율
            df["trading_value_ratio"] = df["trPrc"] / df["mrktTotAmt"]

        # 3. 시장 지표
        if "mrktTotAmt" in df.columns:
            # 로그 시가총액
            df["log_market_cap"] = np.log(df["mrktTotAmt"])

            # 시가총액 분위수
            df["market_cap_rank"] = df.groupby("basDt")["mrktTotAmt"].rank(pct=True)

        return df

    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        """데이터 품질 보고서 생성"""
        report = {
            "total_records": len(df),
            "date_range": {
                "start": df["basDt"].min(),
                "end": df["basDt"].max(),
                "trading_days": df["basDt"].nunique(),
            },
            "universe_size": {
                "total_stocks": df["srtnCd"].nunique(),
                "kospi_stocks": (
                    len(df[df["mrktCtg"] == "KOSPI"]["srtnCd"].unique())
                    if "mrktCtg" in df.columns
                    else 0
                ),
                "kosdaq_stocks": (
                    len(df[df["mrktCtg"] == "KOSDAQ"]["srtnCd"].unique())
                    if "mrktCtg" in df.columns
                    else 0
                ),
            },
            "missing_data": {
                col: df[col].isnull().sum()
                for col in df.columns
                if df[col].isnull().sum() > 0
            },
            "market_coverage": {
                "avg_daily_stocks": df.groupby("basDt").size().mean(),
                "min_daily_stocks": df.groupby("basDt").size().min(),
                "max_daily_stocks": df.groupby("basDt").size().max(),
            },
        }

        return report
