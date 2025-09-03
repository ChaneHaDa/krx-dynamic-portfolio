#!/usr/bin/env python3
"""
실제 KRX 데이터로 포트폴리오 최적화 테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from krx_portfolio.models.mpt import MPTOptimizer
from krx_portfolio.models.risk import RiskModel


def test_portfolio_optimization():
    """실제 데이터로 포트폴리오 최적화 테스트"""
    print("🚀 실제 KRX 데이터 포트폴리오 최적화 테스트 시작")
    print("=" * 60)
    
    # 1. 캐시된 데이터 로드
    cache_dir = Path("./data/cache")
    returns_file = cache_dir / "features" / "daily_returns_20240101_20240131.parquet"
    
    if not returns_file.exists():
        print("❌ 캐시된 수익률 데이터가 없습니다. 먼저 ETL을 실행하세요.")
        return False
    
    print("📊 캐시된 수익률 데이터 로딩 중...")
    returns_df = pd.read_parquet(returns_file)
    print(f"   • 데이터 형태: {returns_df.shape}")
    print(f"   • 기간: {returns_df.index[0]} ~ {returns_df.index[-1]}")
    print(f"   • 종목 수: {returns_df.shape[1]}")
    
    # 결측치 제거 및 데이터 품질 확인
    returns_clean = returns_df.dropna(axis=1, how='any')
    print(f"   • 결측치 제거 후 종목 수: {returns_clean.shape[1]}")
    
    # 상위 100개 종목만 선택 (계산 시간 단축)
    if returns_clean.shape[1] > 100:
        # 거래량 기준으로 상위 종목 선택
        vol_sum = returns_clean.abs().sum().sort_values(ascending=False)
        top_stocks = vol_sum.head(100).index
        returns_clean = returns_clean[top_stocks]
        print(f"   • 상위 100개 종목 선택: {returns_clean.shape}")
    
    # 2. 리스크 모델 피팅
    print("\n📈 리스크 모델 피팅 중...")
    start_time = time.time()
    
    risk_model = RiskModel(method="ledoit_wolf")
    risk_model.fit(returns_clean)
    
    fit_time = time.time() - start_time
    print(f"   • 리스크 모델 피팅 완료: {fit_time:.2f}초")
    
    # 공분산 행렬 확인
    cov_matrix = risk_model.cov()
    print(f"   • 공분산 행렬 형태: {cov_matrix.shape}")
    
    # 3. 포트폴리오 최적화
    print("\n🎯 포트폴리오 최적화 실행 중...")
    
    # 기대 수익률 계산 (단순 평균)
    expected_returns = returns_clean.mean() * 252  # 연율화
    print(f"   • 기대 수익률 범위: {expected_returns.min():.3f} ~ {expected_returns.max():.3f}")
    
    # MPT 최적화
    optimizer = MPTOptimizer()
    
    # 최대 샤프 비율 포트폴리오
    start_time = time.time()
    weights_sharpe = optimizer.max_sharpe(
        mu=expected_returns.values,
        sigma=cov_matrix,
        max_weight=0.1  # 최대 10% 제한
    )
    sharpe_time = time.time() - start_time
    
    print(f"   • 최대 샤프 비율 최적화 완료: {sharpe_time:.2f}초")
    print(f"   • 포트폴리오 종목 수: {np.sum(weights_sharpe > 0.001)}")
    print(f"   • 최대 비중: {weights_sharpe.max():.3f}")
    
    # 최소 분산 포트폴리오
    start_time = time.time()
    weights_minvol = optimizer.min_variance(
        sigma=cov_matrix,
        max_weight=0.1
    )
    minvol_time = time.time() - start_time
    
    print(f"   • 최소 분산 최적화 완료: {minvol_time:.2f}초")
    
    # 4. 포트폴리오 성과 분석
    print("\n📊 포트폴리오 성과 분석")
    
    # 최대 샤프 비율 포트폴리오 성과
    portfolio_return_sharpe = np.dot(weights_sharpe, expected_returns)
    portfolio_vol_sharpe = np.sqrt(np.dot(weights_sharpe, np.dot(cov_matrix, weights_sharpe)))
    sharpe_ratio = portfolio_return_sharpe / portfolio_vol_sharpe
    
    print(f"   🎯 최대 샤프 비율 포트폴리오:")
    print(f"      • 기대 수익률: {portfolio_return_sharpe:.3f} ({portfolio_return_sharpe*100:.1f}%)")
    print(f"      • 변동성: {portfolio_vol_sharpe:.3f} ({portfolio_vol_sharpe*100:.1f}%)")
    print(f"      • 샤프 비율: {sharpe_ratio:.3f}")
    
    # 최소 분산 포트폴리오 성과
    portfolio_return_minvol = np.dot(weights_minvol, expected_returns)
    portfolio_vol_minvol = np.sqrt(np.dot(weights_minvol, np.dot(cov_matrix, weights_minvol)))
    
    print(f"   📉 최소 분산 포트폴리오:")
    print(f"      • 기대 수익률: {portfolio_return_minvol:.3f} ({portfolio_return_minvol*100:.1f}%)")
    print(f"      • 변동성: {portfolio_vol_minvol:.3f} ({portfolio_vol_minvol*100:.1f}%)")
    print(f"      • 샤프 비율: {portfolio_return_minvol/portfolio_vol_minvol:.3f}")
    
    # 상위 보유 종목 출력
    print(f"\n📈 최대 샤프 비율 포트폴리오 상위 종목:")
    weights_series = pd.Series(weights_sharpe, index=returns_clean.columns)
    top_holdings = weights_series.sort_values(ascending=False).head(10)
    for symbol, weight in top_holdings.items():
        if weight > 0.001:  # 0.1% 이상만 표시
            print(f"      • {symbol}: {weight:.3f} ({weight*100:.1f}%)")
    
    print("\n✅ 포트폴리오 최적화 테스트 완료!")
    return True


if __name__ == "__main__":
    test_portfolio_optimization()