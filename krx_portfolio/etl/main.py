"""ETL 파이프라인 메인 실행 스크립트"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import KRXDataLoader
from .preprocessor import KRXPreprocessor


def setup_cache_directory(cache_path: Path) -> None:
    """캐시 디렉토리 설정"""
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "raw").mkdir(exist_ok=True)
    (cache_path / "processed").mkdir(exist_ok=True)
    (cache_path / "features").mkdir(exist_ok=True)


def run_etl_pipeline(
    data_root: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_path: str = "./data/cache",
    force_reload: bool = False
) -> None:
    """ETL 파이프라인 실행
    
    Args:
        data_root: KRX JSON 데이터 루트 경로
        start_date: 시작 날짜 (YYYYMMDD, None이면 최근 30일)
        end_date: 종료 날짜 (YYYYMMDD, None이면 오늘)
        cache_path: 캐시 저장 경로
        force_reload: 강제 리로드 여부
    """
    
    print("🚀 KRX Dynamic Portfolio ETL Pipeline 시작")
    print("=" * 60)
    
    # 1. 초기화
    cache_dir = Path(cache_path)
    setup_cache_directory(cache_dir)
    
    loader = KRXDataLoader(data_root)
    preprocessor = KRXPreprocessor(min_market_cap=100e8, min_volume=1000)
    
    # 2. 날짜 범위 설정
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if not start_date:
        start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=30)
        start_date = start_dt.strftime("%Y%m%d")
    
    print(f"📅 데이터 로딩 기간: {start_date} ~ {end_date}")
    
    # 3. 캐시 확인
    raw_cache_file = cache_dir / "raw" / f"krx_data_{start_date}_{end_date}.parquet"
    
    if raw_cache_file.exists() and not force_reload:
        print(f"📂 캐시에서 원시 데이터 로드: {raw_cache_file}")
        raw_df = pd.read_parquet(raw_cache_file)
    else:
        # 4. 원시 데이터 로딩
        print("📥 KRX JSON 데이터 로딩 중...")
        try:
            raw_df = loader.load_date_range(start_date, end_date)
            print(f"✅ 로드 완료: {len(raw_df):,}건")
            
            # 원시 데이터 캐시 저장
            raw_df.to_parquet(raw_cache_file)
            print(f"💾 원시 데이터 캐시 저장: {raw_cache_file}")
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            sys.exit(1)
    
    # 5. 데이터 품질 보고서
    print("\n📊 원시 데이터 품질 보고서:")
    quality_report = preprocessor.get_data_quality_report(raw_df)
    print_quality_report(quality_report)
    
    # 6. 데이터 전처리
    print("\n🧹 데이터 전처리 중...")
    processed_cache_file = cache_dir / "processed" / f"krx_processed_{start_date}_{end_date}.parquet"
    
    if processed_cache_file.exists() and not force_reload:
        print(f"📂 캐시에서 전처리 데이터 로드: {processed_cache_file}")
        processed_df = pd.read_parquet(processed_cache_file)
    else:
        processed_df = preprocessor.clean_data(raw_df)
        print(f"✅ 전처리 완료: {len(processed_df):,}건 (제거: {len(raw_df) - len(processed_df):,}건)")
        
        # 전처리 데이터 캐시 저장
        processed_df.to_parquet(processed_cache_file)
        print(f"💾 전처리 데이터 캐시 저장: {processed_cache_file}")
    
    # 7. 분석용 데이터셋 생성
    print("\n🔧 분석용 데이터셋 생성 중...")
    
    # OHLCV 패널 데이터
    ohlcv_file = cache_dir / "features" / f"ohlcv_panel_{start_date}_{end_date}.parquet"
    if not ohlcv_file.exists() or force_reload:
        ohlcv_df = preprocessor.create_ohlcv_panel(processed_df)
        ohlcv_df.to_parquet(ohlcv_file)
        print(f"✅ OHLCV 패널 생성: {ohlcv_df.shape}")
    else:
        print(f"📂 OHLCV 패널 캐시 사용: {ohlcv_file}")
    
    # 수익률 매트릭스
    returns_file = cache_dir / "features" / f"daily_returns_{start_date}_{end_date}.parquet"
    if not returns_file.exists() or force_reload:
        ohlcv_df = pd.read_parquet(ohlcv_file)
        returns_df = preprocessor.create_returns_matrix(ohlcv_df, period='1D')
        returns_df.to_parquet(returns_file)
        print(f"✅ 일일 수익률 매트릭스 생성: {returns_df.shape}")
    else:
        print(f"📂 수익률 매트릭스 캐시 사용: {returns_file}")
    
    # 시가총액 가중치
    weights_file = cache_dir / "features" / f"market_cap_weights_{start_date}_{end_date}.parquet"
    if not weights_file.exists() or force_reload:
        weights_df = preprocessor.create_market_cap_weights(processed_df)
        weights_df.to_parquet(weights_file)
        print(f"✅ 시가총액 가중치 생성: {weights_df.shape}")
    else:
        print(f"📂 시가총액 가중치 캐시 사용: {weights_file}")
    
    # 8. 투자 유니버스 생성
    universe_file = cache_dir / "features" / f"investment_universe_{end_date}.json"
    if not universe_file.exists() or force_reload:
        # KOSPI 상위 100개
        kospi_universe = preprocessor.filter_investable_universe(
            processed_df, top_n=100, market='KOSPI'
        )
        # KOSDAQ 상위 50개
        kosdaq_universe = preprocessor.filter_investable_universe(
            processed_df, top_n=50, market='KOSDAQ'
        )
        
        universe_data = {
            'date': end_date,
            'kospi_top100': kospi_universe,
            'kosdaq_top50': kosdaq_universe,
            'combined_top150': kospi_universe + kosdaq_universe[:50]
        }
        
        import json
        with open(universe_file, 'w', encoding='utf-8') as f:
            json.dump(universe_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 투자 유니버스 생성: KOSPI {len(kospi_universe)}개, KOSDAQ {len(kosdaq_universe)}개")
    else:
        print(f"📂 투자 유니버스 캐시 사용: {universe_file}")
    
    print(f"\n🎉 ETL 파이프라인 완료!")
    print(f"📁 캐시 위치: {cache_dir.absolute()}")
    print("=" * 60)


def print_quality_report(report: dict) -> None:
    """데이터 품질 보고서 출력"""
    print(f"  • 전체 레코드: {report['total_records']:,}건")
    print(f"  • 데이터 기간: {report['date_range']['start']} ~ {report['date_range']['end']} "
          f"({report['date_range']['trading_days']}일)")
    print(f"  • 종목 수: 전체 {report['universe_size']['total_stocks']:,}개 "
          f"(KOSPI {report['universe_size']['kospi_stocks']:,}개, "
          f"KOSDAQ {report['universe_size']['kosdaq_stocks']:,}개)")
    print(f"  • 일평균 종목 수: {report['market_coverage']['avg_daily_stocks']:.0f}개")
    
    if report['missing_data']:
        print(f"  • 결측치: {len(report['missing_data'])}개 컬럼")
    else:
        print(f"  • 결측치: 없음 ✅")


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description="KRX Dynamic Portfolio ETL Pipeline")
    parser.add_argument(
        '--data-root', 
        required=True,
        help="KRX JSON 데이터 루트 경로 (예: /home/chan/code/vscode/python3/krx-json-data)"
    )
    parser.add_argument('--start-date', help="시작 날짜 (YYYYMMDD)")
    parser.add_argument('--end-date', help="종료 날짜 (YYYYMMDD)")
    parser.add_argument('--cache-path', default="./data/cache", help="캐시 저장 경로")
    parser.add_argument('--force-reload', action='store_true', help="강제 리로드")
    
    args = parser.parse_args()
    
    run_etl_pipeline(
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_path=args.cache_path,
        force_reload=args.force_reload
    )


if __name__ == "__main__":
    main()