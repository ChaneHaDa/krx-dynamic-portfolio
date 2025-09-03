#!/usr/bin/env python3
"""
실제 KRX 데이터로 포트폴리오 최적화 테스트 (수정버전)
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
    
    # 1. OHLCV 데이터 로드
    cache_dir = Path("./data/cache")
    ohlcv_file = cache_dir / "features" / "ohlcv_panel_20240101_20240131.parquet"
    
    if not ohlcv_file.exists():
        print("❌ OHLCV 데이터가 없습니다. 먼저 ETL을 실행하세요.")
        return False
    
    print("📊 OHLCV 데이터 로딩 중...")
    ohlcv_df = pd.read_parquet(ohlcv_file)
    print(f"   • 데이터 형태: {ohlcv_df.shape}")
    
    # 2. 수익률 매트릭스 생성
    print("📈 수익률 매트릭스 생성 중...")
    
    # 종가 데이터를 피벗하여 매트릭스 형태로 변환
    close_prices = ohlcv_df.reset_index().pivot(
        index='date', columns='stock_code', values='close'
    )
    
    print(f"   • 종가 매트릭스 형태: {close_prices.shape}")
    print(f"   • 기간: {close_prices.index[0]} ~ {close_prices.index[-1]}")
    
    # 수익률 계산
    returns = close_prices.pct_change().dropna()
    print(f"   • 수익률 매트릭스 형태: {returns.shape}")
    
    # 결측치가 너무 많은 종목 제거 (80% 이상 데이터가 있는 종목만)
    min_observations = int(len(returns) * 0.8)
    valid_stocks = returns.count() >= min_observations
    returns_clean = returns.loc[:, valid_stocks]
    print(f"   • 결측치 제거 후 종목 수: {returns_clean.shape[1]}")
    
    # 나머지 결측치는 0으로 채움 (거래정지 등)
    returns_clean = returns_clean.fillna(0)
    
    # 상위 50개 종목만 선택 (계산 시간 단축)
    if returns_clean.shape[1] > 50:
        # 변동성 기준으로 상위 종목 선택 (더 활발한 거래)
        volatility = returns_clean.std().sort_values(ascending=False)
        top_stocks = volatility.head(50).index
        returns_clean = returns_clean[top_stocks]
        print(f"   • 상위 50개 종목 선택: {returns_clean.shape}")
    
    # 3. 리스크 모델 피팅
    print("\\n📈 리스크 모델 피팅 중...")
    start_time = time.time()
    
    try:
        risk_model = RiskModel(method="ledoit_wolf")
        risk_model.fit(returns_clean)
        
        fit_time = time.time() - start_time
        print(f"   • 리스크 모델 피팅 완료: {fit_time:.2f}초")
        
        # 공분산 행렬 확인
        cov_matrix = risk_model.cov()
        print(f"   • 공분산 행렬 형태: {cov_matrix.shape}")
        
    except Exception as e:
        print(f"   ❌ 리스크 모델 피팅 실패: {e}")
        print("   📊 샘플 공분산 행렬 사용")
        cov_matrix = returns_clean.cov().values
    
    # 4. 포트폴리오 최적화
    print("\\n🎯 포트폴리오 최적화 실행 중...")
    
    # 기대 수익률 계산 (단순 평균)
    expected_returns = returns_clean.mean() * 252  # 연율화
    print(f"   • 기대 수익률 범위: {expected_returns.min():.3f} ~ {expected_returns.max():.3f}")
    
    # MPT 최적화
    optimizer = MPTOptimizer()
    
    try:
        # 최대 샤프 비율 포트폴리오
        print("   ⏱️ 최대 샤프 비율 최적화 중...")
        start_time = time.time()
        weights_sharpe = optimizer.max_sharpe(
            mu=expected_returns.values,
            sigma=cov_matrix,
            max_weight=0.2  # 최대 20% 제한
        )
        sharpe_time = time.time() - start_time
        
        print(f"   • 최대 샤프 비율 최적화 완료: {sharpe_time:.2f}초")
        print(f"   • 포트폴리오 종목 수: {np.sum(weights_sharpe > 0.01)}")
        print(f"   • 최대 비중: {weights_sharpe.max():.3f}")
        
    except Exception as e:
        print(f"   ❌ 최대 샤프 비율 최적화 실패: {e}")
        # 동일 가중 포트폴리오로 대체
        weights_sharpe = np.ones(len(expected_returns)) / len(expected_returns)
        print("   📊 동일 가중 포트폴리오 사용")
    
    try:
        # 최소 분산 포트폴리오
        print("   ⏱️ 최소 분산 최적화 중...")
        start_time = time.time()
        weights_minvol = optimizer.min_variance(
            sigma=cov_matrix,
            max_weight=0.2
        )
        minvol_time = time.time() - start_time
        
        print(f"   • 최소 분산 최적화 완료: {minvol_time:.2f}초")
        
    except Exception as e:
        print(f"   ❌ 최소 분산 최적화 실패: {e}")
        weights_minvol = np.ones(len(expected_returns)) / len(expected_returns)
        print("   📊 동일 가중 포트폴리오 사용")
    
    # 5. 포트폴리오 성과 분석
    print("\\n📊 포트폴리오 성과 분석")
    
    # 최대 샤프 비율 포트폴리오 성과
    portfolio_return_sharpe = np.dot(weights_sharpe, expected_returns)
    portfolio_vol_sharpe = np.sqrt(np.dot(weights_sharpe, np.dot(cov_matrix, weights_sharpe)))
    sharpe_ratio = portfolio_return_sharpe / portfolio_vol_sharpe if portfolio_vol_sharpe > 0 else 0
    
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
    print(f"\\n📈 최대 샤프 비율 포트폴리오 상위 종목:")
    weights_series = pd.Series(weights_sharpe, index=returns_clean.columns)
    top_holdings = weights_series.sort_values(ascending=False).head(10)
    for symbol, weight in top_holdings.items():
        if weight > 0.01:  # 1% 이상만 표시
            print(f"      • {symbol}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 6. 간단한 백테스트
    print("\\n🔄 간단한 백테스트 실행")
    portfolio_daily_returns = (returns_clean * weights_sharpe).sum(axis=1)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    
    total_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_daily_returns)) - 1
    annual_vol = portfolio_daily_returns.std() * np.sqrt(252)
    
    print(f"   📊 백테스트 결과 (기간: {returns_clean.index[0].strftime('%Y-%m-%d')} ~ {returns_clean.index[-1].strftime('%Y-%m-%d')}):")
    print(f"      • 총 수익률: {total_return:.3f} ({total_return*100:.1f}%)")
    print(f"      • 연평균 수익률: {annual_return:.3f} ({annual_return*100:.1f}%)")
    print(f"      • 연변동성: {annual_vol:.3f} ({annual_vol*100:.1f}%)")
    print(f"      • 실현 샤프 비율: {annual_return/annual_vol:.3f}")
    
    print("\\n✅ 포트폴리오 최적화 테스트 완료!")
    return True


if __name__ == "__main__":
    test_portfolio_optimization()