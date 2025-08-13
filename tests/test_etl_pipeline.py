"""ETL 파이프라인 통합 테스트"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from krx_portfolio.etl.main import (
    print_quality_report,
    run_etl_pipeline,
    setup_cache_directory,
)


class TestETLPipeline:
    """ETL 파이프라인 통합 테스트"""

    def test_setup_cache_directory(self):
        """캐시 디렉토리 설정 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"

            setup_cache_directory(cache_path)

            assert cache_path.exists()
            assert (cache_path / "raw").exists()
            assert (cache_path / "processed").exists()
            assert (cache_path / "features").exists()

    def test_print_quality_report(self, capsys):
        """데이터 품질 보고서 출력 테스트"""
        report = {
            "total_records": 10000,
            "date_range": {
                "start": pd.Timestamp("2023-12-01"),
                "end": pd.Timestamp("2023-12-15"),
                "trading_days": 11,
            },
            "universe_size": {
                "total_stocks": 2500,
                "kospi_stocks": 800,
                "kosdaq_stocks": 1700,
            },
            "missing_data": {"volume": 5, "price": 2},
            "market_coverage": {
                "avg_daily_stocks": 2450.5,
                "min_daily_stocks": 2400,
                "max_daily_stocks": 2500,
            },
        }

        print_quality_report(report)

        captured = capsys.readouterr()
        output = captured.out

        assert "전체 레코드: 10,000건" in output
        assert "2023-12-01" in output and "2023-12-15" in output
        assert "11일" in output
        assert "전체 2,500개" in output
        assert "KOSPI 800개" in output
        assert "KOSDAQ 1,700개" in output
        assert "일평균 종목 수: 2450개" in output
        assert "결측치: 2개 컬럼" in output

    def test_print_quality_report_no_missing_data(self, capsys):
        """결측치가 없는 경우 품질 보고서 출력 테스트"""
        report = {
            "total_records": 5000,
            "date_range": {
                "start": pd.Timestamp("2023-12-01"),
                "end": pd.Timestamp("2023-12-05"),
                "trading_days": 5,
            },
            "universe_size": {
                "total_stocks": 1000,
                "kospi_stocks": 400,
                "kosdaq_stocks": 600,
            },
            "missing_data": {},  # 결측치 없음
            "market_coverage": {
                "avg_daily_stocks": 1000.0,
                "min_daily_stocks": 1000,
                "max_daily_stocks": 1000,
            },
        }

        print_quality_report(report)

        captured = capsys.readouterr()
        output = captured.out

        assert "결측치: 없음 ✅" in output

    @patch("krx_portfolio.etl.main.datetime")
    def test_run_etl_pipeline_with_cache(
        self, mock_datetime, temp_data_dir, temp_cache_dir
    ):
        """캐시가 있는 경우 ETL 파이프라인 실행 테스트"""
        # datetime.now() 모킹
        mock_datetime.now.return_value.strftime.return_value = "20231215"
        mock_datetime.strptime.side_effect = lambda x, y: pd.to_datetime(
            x, format=y
        ).to_pydatetime()

        # 캐시 파일 생성
        cache_file = temp_cache_dir / "raw" / "krx_data_20231201_20231215.parquet"
        sample_data = pd.DataFrame(
            {
                "basDt": [pd.Timestamp("2023-12-15")] * 3,
                "srtnCd": ["005930", "000660", "035420"],
                "itmsNm": ["삼성전자", "SK하이닉스", "NAVER"],
                "mrktCtg": ["KOSPI", "KOSPI", "KOSDAQ"],
                "clpr": [70000, 125000, 200000],
                "vs": [1000, -2000, 3000],
                "fltRt": [0.0145, -0.0157, 0.0152],
                "mkp": [69500, 127000, 198000],
                "hipr": [70500, 128000, 201000],
                "lopr": [69000, 124500, 197500],
                "trqu": [10000000, 5000000, 1000000],
                "trPrc": [7.0e11, 6.25e11, 2.0e11],
                "lstgStCnt": [5969782550, 728002365, 164831339],
                "mrktTotAmt": [4.178e14, 9.1e13, 3.297e13],
            }
        )
        sample_data.to_parquet(cache_file)

        # 출력 캡처
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            run_etl_pipeline(
                data_root=str(temp_data_dir),
                start_date="20231201",
                end_date="20231215",
                cache_path=str(temp_cache_dir),
                force_reload=False,
            )

        output = captured_output.getvalue()

        # 캐시 사용 메시지 확인
        assert "캐시에서 원시 데이터 로드" in output
        assert "ETL 파이프라인 완료" in output

        # 결과 파일들이 생성되었는지 확인
        assert (
            temp_cache_dir / "features" / "ohlcv_panel_20231201_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "daily_returns_20231201_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "market_cap_weights_20231201_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "investment_universe_20231215.json"
        ).exists()

    def test_run_etl_pipeline_fresh_data(self, temp_data_dir, temp_cache_dir):
        """새로운 데이터로 ETL 파이프라인 실행 테스트"""
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            run_etl_pipeline(
                data_root=str(temp_data_dir),
                start_date="20231215",
                end_date="20231215",
                cache_path=str(temp_cache_dir),
                force_reload=False,
            )

        output = captured_output.getvalue()

        # 데이터 로딩 메시지 확인
        assert "KRX JSON 데이터 로딩 중" in output
        assert "로드 완료" in output
        assert "전처리 완료" in output
        assert "OHLCV 패널 생성" in output
        assert "일일 수익률 매트릭스 생성" in output
        assert "시가총액 가중치 생성" in output
        assert "투자 유니버스 생성" in output

        # 캐시 파일들이 생성되었는지 확인
        assert (temp_cache_dir / "raw" / "krx_data_20231215_20231215.parquet").exists()
        assert (
            temp_cache_dir / "processed" / "krx_processed_20231215_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "ohlcv_panel_20231215_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "daily_returns_20231215_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "market_cap_weights_20231215_20231215.parquet"
        ).exists()
        assert (
            temp_cache_dir / "features" / "investment_universe_20231215.json"
        ).exists()

    def test_run_etl_pipeline_force_reload(self, temp_data_dir, temp_cache_dir):
        """강제 리로드로 ETL 파이프라인 실행 테스트"""
        # 기존 캐시 파일 생성
        cache_file = temp_cache_dir / "raw" / "krx_data_20231215_20231215.parquet"
        old_data = pd.DataFrame({"dummy": [1, 2, 3]})
        old_data.to_parquet(cache_file)

        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            run_etl_pipeline(
                data_root=str(temp_data_dir),
                start_date="20231215",
                end_date="20231215",
                cache_path=str(temp_cache_dir),
                force_reload=True,
            )

        output = captured_output.getvalue()

        # 강제 리로드 시 캐시를 무시하고 새로 로드
        assert "KRX JSON 데이터 로딩 중" in output

        # 새로운 데이터가 캐시에 저장됨
        reloaded_data = pd.read_parquet(cache_file)
        assert "basDt" in reloaded_data.columns  # 실제 KRX 데이터 구조
        assert "dummy" not in reloaded_data.columns  # 이전 더미 데이터 제거

    def test_run_etl_pipeline_no_date_range(self, temp_data_dir, temp_cache_dir):
        """날짜 범위 없이 ETL 파이프라인 실행 테스트 (기본값 사용)"""
        captured_output = io.StringIO()

        with (
            patch("sys.stdout", captured_output),
            patch("krx_portfolio.etl.main.datetime") as mock_datetime,
        ):

            # 현재 날짜 모킹
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20231220"
            mock_datetime.now.return_value = mock_now

            # strptime 모킹
            def mock_strptime(date_str, format_str):
                return pd.to_datetime(date_str, format=format_str).to_pydatetime()

            mock_datetime.strptime.side_effect = mock_strptime

            try:
                run_etl_pipeline(
                    data_root=str(temp_data_dir), cache_path=str(temp_cache_dir)
                )
            except SystemExit:
                # 데이터가 없어서 실패할 수 있음 (정상적인 동작)
                pass

        output = captured_output.getvalue()

        # 기본 날짜 범위 사용 확인 (30일 전부터 오늘까지)
        assert "데이터 로딩 기간" in output

    def test_run_etl_pipeline_data_loading_failure(self, temp_cache_dir):
        """데이터 로딩 실패 시 ETL 파이프라인 테스트"""
        # 존재하지 않는 데이터 경로
        nonexistent_path = "/nonexistent/path"

        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output), pytest.raises(SystemExit):

            run_etl_pipeline(
                data_root=nonexistent_path,
                start_date="20231215",
                end_date="20231215",
                cache_path=str(temp_cache_dir),
                force_reload=False,
            )

        output = captured_output.getvalue()

        # 에러 메시지 확인
        assert "데이터 로딩 실패" in output

    def test_investment_universe_generation(self, temp_data_dir, temp_cache_dir):
        """투자 유니버스 생성 테스트"""
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            run_etl_pipeline(
                data_root=str(temp_data_dir),
                start_date="20231215",
                end_date="20231215",
                cache_path=str(temp_cache_dir),
            )

        # 투자 유니버스 파일 확인
        universe_file = (
            temp_cache_dir / "features" / "investment_universe_20231215.json"
        )
        assert universe_file.exists()

        with open(universe_file, encoding="utf-8") as f:
            universe_data = json.load(f)

        assert "date" in universe_data
        assert "kospi_top100" in universe_data
        assert "kosdaq_top50" in universe_data
        assert "combined_top150" in universe_data

        # KOSPI 종목들 확인 (시가총액 순으로 정렬되므로 상위 종목 확인)
        kospi_stocks = universe_data["kospi_top100"]
        assert len(kospi_stocks) > 0
        assert isinstance(kospi_stocks, list)

        # KOSDAQ 종목들 확인
        kosdaq_stocks = universe_data["kosdaq_top50"]
        assert "035420" in kosdaq_stocks  # NAVER

    def test_generated_files_content_validation(self, temp_data_dir, temp_cache_dir):
        """생성된 파일들의 내용 검증 테스트"""
        run_etl_pipeline(
            data_root=str(temp_data_dir),
            start_date="20231215",
            end_date="20231215",
            cache_path=str(temp_cache_dir),
        )

        # OHLCV 패널 데이터 검증
        ohlcv_file = (
            temp_cache_dir / "features" / "ohlcv_panel_20231215_20231215.parquet"
        )
        ohlcv_df = pd.read_parquet(ohlcv_file)

        assert isinstance(ohlcv_df.index, pd.MultiIndex)
        assert ohlcv_df.index.names == ["date", "stock_code"]
        expected_cols = [
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        assert all(col in ohlcv_df.columns for col in expected_cols)

        # 수익률 매트릭스 검증
        returns_file = (
            temp_cache_dir / "features" / "daily_returns_20231215_20231215.parquet"
        )
        returns_df = pd.read_parquet(returns_file)

        assert isinstance(returns_df.index, pd.DatetimeIndex)
        assert len(returns_df.columns) > 0

        # 시가총액 가중치 검증
        weights_file = (
            temp_cache_dir / "features" / "market_cap_weights_20231215_20231215.parquet"
        )
        weights_df = pd.read_parquet(weights_file)

        assert isinstance(weights_df.index, pd.DatetimeIndex)
        # 각 날짜별 가중치 합이 1에 가까운지 확인
        for date in weights_df.index:
            row_sum = weights_df.loc[date].sum()
            assert abs(row_sum - 1.0) < 1e-10  # 부동소수점 오차 고려
