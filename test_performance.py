#!/usr/bin/env python3
"""
실제 KRX 데이터로 성능 병목점 분석
"""

import time
import pandas as pd
import psutil
from pathlib import Path
import gc


def analyze_performance():
    """성능 병목점 분석"""
    print("🚀 KRX 데이터 성능 병목점 분석 시작")
    print("=" * 60)
    
    # 시스템 정보
    print("🖥️  시스템 정보:")
    print(f"   • CPU 코어: {psutil.cpu_count()}")
    print(f"   • 메모리: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"   • 사용 가능 메모리: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # 1. ETL 성능 테스트
    print("\\n📊 ETL 파이프라인 성능 테스트")
    
    data_sizes = [
        ("1개월", "--start-date=20240101 --end-date=20240131"),
        ("3개월", "--start-date=20240101 --end-date=20240331"),
        ("6개월", "--start-date=20240101 --end-date=20240630"),
    ]
    
    for period, date_args in data_sizes:
        print(f"\\n   📅 {period} 데이터 처리 중...")
        
        # 메모리 사용량 측정 시작
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        start_time = time.time()
        
        # ETL 실행 (캐시 강제 리로드)
        import subprocess
        cmd = ["python", "-m", "krx_portfolio.etl.main",
               "--data-root=/home/ind/code/st-pro/krx-json-data",
               "--cache-path=./data/cache",
               "--force-reload"] + date_args.split()
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="./", 
                              env={"PYTHONPATH": "."})
        
        elapsed_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_used = mem_after - mem_before
        
        print(f"      ⏱️  처리 시간: {elapsed_time:.2f}초")
        print(f"      💾 메모리 사용량: {mem_used:.1f} MB")
        
        if result.returncode == 0:
            # 생성된 파일 크기 확인
            cache_dir = Path("./data/cache")
            if cache_dir.exists():
                total_size = sum(f.stat().st_size for f in cache_dir.rglob('*.parquet'))
                print(f"      📁 생성된 캐시 크기: {total_size / 1024**2:.1f} MB")
            print(f"      ✅ 성공")
        else:
            print(f"      ❌ 실패: {result.stderr[:100]}...")
        
        # 메모리 정리
        gc.collect()
        time.sleep(2)  # 시스템 안정화
    
    # 2. 데이터 로딩 성능 테스트
    print("\\n📈 데이터 로딩 성능 테스트")
    
    cache_files = [
        ("원시 데이터", "raw/krx_data_*.parquet"),
        ("전처리 데이터", "processed/krx_processed_*.parquet"), 
        ("OHLCV 패널", "features/ohlcv_panel_*.parquet"),
        ("일일 수익률", "features/daily_returns_*.parquet"),
    ]
    
    for name, pattern in cache_files:
        cache_dir = Path("./data/cache")
        files = list(cache_dir.glob(pattern))
        
        if files:
            file_path = files[0]  # 가장 최근 파일 사용
            file_size = file_path.stat().st_size / 1024**2  # MB
            
            print(f"\\n   📊 {name} 로딩 테스트:")
            print(f"      📁 파일 크기: {file_size:.1f} MB")
            
            # 로딩 시간 측정
            start_time = time.time()
            df = pd.read_parquet(file_path)
            load_time = time.time() - start_time
            
            print(f"      ⏱️  로딩 시간: {load_time:.3f}초")
            print(f"      📊 데이터 형태: {df.shape}")
            print(f"      🚀 처리 속도: {file_size/load_time:.1f} MB/초")
            
            # 메모리 사용량
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            print(f"      💾 메모리 사용량: {memory_usage:.1f} MB")
            
            del df
            gc.collect()
    
    # 3. 대용량 데이터 처리 성능 시뮬레이션
    print("\\n🔍 대용량 데이터 처리 성능 시뮬레이션")
    
    # 가상의 대용량 데이터 생성
    import numpy as np
    
    data_sizes = [1000, 5000, 10000, 50000]  # 종목 수
    
    for n_stocks in data_sizes:
        print(f"\\n   📈 {n_stocks}개 종목 시뮬레이션:")
        
        # 데이터 생성
        start_time = time.time()
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        n_days = len(dates)
        
        # 랜덤 수익률 매트릭스 생성
        returns_data = np.random.normal(0.001, 0.02, size=(n_days, n_stocks))
        returns_df = pd.DataFrame(returns_data, index=dates, 
                                columns=[f'STOCK_{i:06d}' for i in range(n_stocks)])
        
        creation_time = time.time() - start_time
        memory_usage = returns_df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"      📊 데이터 형태: {returns_df.shape}")
        print(f"      ⏱️  생성 시간: {creation_time:.3f}초")
        print(f"      💾 메모리 사용량: {memory_usage:.1f} MB")
        
        # 기본 연산 성능 테스트
        start_time = time.time()
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        calc_time = time.time() - start_time
        
        print(f"      🧮 통계 계산 시간: {calc_time:.3f}초")
        print(f"      📊 공분산 행렬 크기: {cov_matrix.shape}")
        
        # 예상 최적화 시간 (대략적)
        optimization_time_est = (n_stocks / 1000) ** 2 * 0.5  # 대략적 추정
        print(f"      🎯 예상 최적화 시간: {optimization_time_est:.2f}초")
        
        del returns_df, mean_returns, cov_matrix
        gc.collect()
        
        if n_stocks >= 10000:
            print("      ⚠️  대용량 데이터: 샘플링 또는 차원 축소 권장")
    
    # 4. 성능 권장사항
    print("\\n💡 성능 최적화 권장사항")
    print("=" * 40)
    print("📊 데이터 크기별 권장사항:")
    print("   • ~1,000 종목: 전체 데이터 처리 가능")
    print("   • ~5,000 종목: 샘플링 또는 배치 처리 권장")
    print("   • ~10,000+ 종목: 차원 축소 또는 분산 처리 필수")
    print()
    print("⚡ 성능 최적화 방법:")
    print("   • 데이터 캐싱: Parquet 포맷 사용")
    print("   • 메모리 관리: 배치 처리, 가비지 컬렉션")
    print("   • 병렬 처리: multiprocessing 활용")
    print("   • 샘플링: 상위 유동성 종목 우선 선택")
    
    print("\\n✅ 성능 분석 완료!")


if __name__ == "__main__":
    analyze_performance()