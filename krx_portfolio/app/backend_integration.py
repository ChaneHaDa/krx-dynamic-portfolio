"""
대시보드-백엔드 통합 모듈
===========================

이 모듈은 Streamlit 대시보드와 실제 백엔드 로직 간의 연동을 담당합니다.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BackendIntegration:
    """백엔드 통합 클래스"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run_etl_pipeline(self, data_root: str, force_reload: bool = False, 
                        days_back: int = 30) -> Tuple[bool, str]:
        """
        ETL 파이프라인 실행
        
        Parameters
        ----------
        data_root : str
            KRX JSON 데이터 루트 경로
        force_reload : bool
            강제 리로드 여부
        days_back : int
            과거 몇 일의 데이터를 처리할지
            
        Returns
        -------
        Tuple[bool, str]
            (성공 여부, 메시지)
        """
        try:
            # 실제 ETL 모듈 import 시도
            from krx_portfolio.etl.main import run_etl_pipeline
            
            # 날짜 범위 설정
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            
            # ETL 실행
            run_etl_pipeline(
                data_root=data_root,
                start_date=start_date,
                end_date=end_date,
                cache_path=str(self.cache_dir),
                force_reload=force_reload
            )
            
            return True, f"ETL 파이프라인 실행 완료 ({start_date} ~ {end_date})"
            
        except ImportError as e:
            return False, f"ETL 모듈을 찾을 수 없습니다: {e}"
        except FileNotFoundError as e:
            return False, f"데이터 파일을 찾을 수 없습니다: {e}"
        except Exception as e:
            logger.error(f"ETL 파이프라인 실행 실패: {e}")
            return False, f"ETL 실행 중 오류 발생: {e}"
    
    def run_portfolio_optimization(self, 
                                 optimization_method: str = "max_sharpe",
                                 risk_model: str = "ledoit_wolf",
                                 max_weight: float = 0.1,
                                 min_weight: float = 0.0,
                                 lookback_days: int = 252) -> Tuple[bool, Dict[str, Any]]:
        """
        포트폴리오 최적화 실행
        
        Parameters
        ----------
        optimization_method : str
            최적화 방법 (max_sharpe, min_variance, mean_variance)
        risk_model : str
            리스크 모델 (sample, ledoit_wolf, oas, ewma)
        max_weight : float
            최대 비중 제한
        min_weight : float
            최소 비중 제한
        lookback_days : int
            과거 데이터 기간
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (성공 여부, 결과 딕셔너리)
        """
        try:
            # 캐시된 데이터 확인
            returns_files = list(self.cache_dir.glob("features/daily_returns_*.parquet"))
            if not returns_files:
                return False, {"error": "ETL 데이터가 없습니다. 먼저 ETL 파이프라인을 실행해주세요."}
            
            # 최신 수익률 데이터 로드
            latest_returns_file = max(returns_files, key=lambda p: p.stat().st_mtime)
            returns_df = pd.read_parquet(latest_returns_file)
            
            # 시가총액 가중치 데이터 로드
            weights_files = list(self.cache_dir.glob("features/market_cap_weights_*.parquet"))
            if weights_files:
                latest_weights_file = max(weights_files, key=lambda p: p.stat().st_mtime)
                market_weights = pd.read_parquet(latest_weights_file)
            else:
                market_weights = None
            
            # 실제 최적화 모듈 import 시도
            from krx_portfolio.models.pipeline import PortfolioOptimizationPipeline
            from krx_portfolio.utils import get_default_config
            
            # 설정 생성
            config = get_default_config()
            config["optimization"].update({
                "method": optimization_method,
                "risk_model": risk_model,
                "max_weight": max_weight,
                "min_weight": min_weight,
                "lookback_days": lookback_days
            })
            
            # 최적화 파이프라인 실행
            pipeline = PortfolioOptimizationPipeline(config=config)
            
            # 최적화 실행 (최근 lookback_days 기간의 데이터 사용)
            recent_returns = returns_df.tail(lookback_days)
            mu = recent_returns.mean()
            
            # 섹터 맵 생성 (필요한 경우)
            sector_map = {col: "General" for col in recent_returns.columns}
            
            # 포트폴리오 최적화 실행
            optimization_results = pipeline.build_weights(mu, recent_returns, sector_map)
            
            # 결과 구성
            results = {
                "weights": optimization_results.get("weights", pd.Series()),
                "metrics": {
                    "expected_return": optimization_results.get("expected_return", 0) * 100,
                    "volatility": optimization_results.get("volatility", 0) * 100,
                    "sharpe_ratio": optimization_results.get("sharpe_ratio", 0),
                    "method": optimization_method,
                    "risk_model": risk_model
                },
                "config": config["optimization"],
                "data_period": {
                    "start": recent_returns.index[0].strftime("%Y-%m-%d"),
                    "end": recent_returns.index[-1].strftime("%Y-%m-%d"),
                    "days": len(recent_returns)
                }
            }
            
            return True, results
            
        except ImportError as e:
            return False, {"error": f"최적화 모듈을 찾을 수 없습니다: {e}"}
        except FileNotFoundError as e:
            return False, {"error": f"데이터 파일을 찾을 수 없습니다: {e}"}
        except Exception as e:
            logger.error(f"포트폴리오 최적화 실행 실패: {e}")
            return False, {"error": f"최적화 실행 중 오류 발생: {e}"}
    
    def run_backtesting(self,
                       start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       initial_capital: int = 100_000_000,
                       transaction_cost_bps: float = 25.0,
                       rebalance_frequency: str = "monthly") -> Tuple[bool, Dict[str, Any]]:
        """
        백테스팅 실행
        
        Parameters
        ----------
        start_date : Union[str, datetime]
            백테스트 시작일
        end_date : Union[str, datetime]
            백테스트 종료일
        initial_capital : int
            초기 자본
        transaction_cost_bps : float
            거래비용 (basis points)
        rebalance_frequency : str
            리밸런싱 주기
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (성공 여부, 결과 딕셔너리)
        """
        try:
            # 날짜 변환
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 캐시된 데이터 확인
            returns_files = list(self.cache_dir.glob("features/daily_returns_*.parquet"))
            weights_files = list(self.cache_dir.glob("features/market_cap_weights_*.parquet"))
            
            if not returns_files:
                return False, {"error": "ETL 데이터가 없습니다. 먼저 ETL 파이프라인을 실행해주세요."}
            
            # 최신 데이터 로드
            latest_returns_file = max(returns_files, key=lambda p: p.stat().st_mtime)
            returns_df = pd.read_parquet(latest_returns_file)
            
            if weights_files:
                latest_weights_file = max(weights_files, key=lambda p: p.stat().st_mtime)
                weights_df = pd.read_parquet(latest_weights_file)
            else:
                # 동일가중 포트폴리오 생성
                weights_df = pd.DataFrame(
                    1/len(returns_df.columns), 
                    index=returns_df.index, 
                    columns=returns_df.columns
                )
            
            # 백테스팅 모듈 import 시도
            from krx_portfolio.backtesting.engine import BacktestEngine
            from krx_portfolio.models.rebalance import Rebalancer
            from krx_portfolio.utils import get_default_config
            
            # 설정 생성
            config = get_default_config()
            config["backtesting"].update({
                "initial_capital": initial_capital,
                "transaction_cost_bps": transaction_cost_bps
            })
            
            # 리밸런서 설정
            rebalancer = Rebalancer(
                schedule=rebalance_frequency,
                tc_bps=transaction_cost_bps
            )
            
            # 백테스트 엔진 생성
            engine = BacktestEngine(
                initial_capital=initial_capital,
                transaction_cost_bps=transaction_cost_bps,
                rebalancer=rebalancer
            )
            
            # 백테스트 실행
            backtest_results = engine.run_backtest(
                weights=weights_df,
                returns=returns_df,
                start_date=start_date,
                end_date=end_date
            )
            
            return True, backtest_results
            
        except ImportError as e:
            return False, {"error": f"백테스팅 모듈을 찾을 수 없습니다: {e}"}
        except FileNotFoundError as e:
            return False, {"error": f"데이터 파일을 찾을 수 없습니다: {e}"}
        except Exception as e:
            logger.error(f"백테스팅 실행 실패: {e}")
            return False, {"error": f"백테스팅 실행 중 오류 발생: {e}"}
    
    def get_available_data_info(self) -> Dict[str, Any]:
        """
        사용 가능한 데이터 정보 조회
        
        Returns
        -------
        Dict[str, Any]
            데이터 정보
        """
        info = {
            "etl_data_available": False,
            "latest_etl_date": None,
            "available_symbols": [],
            "data_period": None,
            "cache_size_mb": 0
        }
        
        try:
            # 캐시 디렉토리 크기 계산
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # 수익률 데이터 확인
            returns_files = list(self.cache_dir.glob("features/daily_returns_*.parquet"))
            if returns_files:
                info["etl_data_available"] = True
                latest_file = max(returns_files, key=lambda p: p.stat().st_mtime)
                
                # 파일명에서 날짜 추출
                filename = latest_file.stem
                if "_" in filename:
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        info["latest_etl_date"] = parts[-1]
                
                # 심볼 정보 로드
                try:
                    returns_df = pd.read_parquet(latest_file)
                    info["available_symbols"] = list(returns_df.columns)
                    info["data_period"] = {
                        "start": returns_df.index[0].strftime("%Y-%m-%d"),
                        "end": returns_df.index[-1].strftime("%Y-%m-%d"),
                        "days": len(returns_df)
                    }
                except Exception as e:
                    logger.warning(f"데이터 파일 읽기 실패: {e}")
            
        except Exception as e:
            logger.error(f"데이터 정보 조회 실패: {e}")
        
        return info
    
    def clear_cache(self) -> Tuple[bool, str]:
        """
        캐시 삭제
        
        Returns
        -------
        Tuple[bool, str]
            (성공 여부, 메시지)
        """
        try:
            import shutil
            
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                return True, "캐시가 성공적으로 삭제되었습니다."
            else:
                return True, "삭제할 캐시가 없습니다."
                
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {e}")
            return False, f"캐시 삭제 중 오류 발생: {e}"


# 전역 백엔드 통합 인스턴스
_backend_integration = None


def get_backend_integration(cache_dir: str = "./data/cache") -> BackendIntegration:
    """
    백엔드 통합 인스턴스 반환 (싱글톤 패턴)
    
    Parameters
    ----------
    cache_dir : str
        캐시 디렉토리 경로
        
    Returns
    -------
    BackendIntegration
        백엔드 통합 인스턴스
    """
    global _backend_integration
    
    if _backend_integration is None:
        _backend_integration = BackendIntegration(cache_dir)
    
    return _backend_integration


# 편의 함수들
def run_etl_pipeline_safe(data_root: str, force_reload: bool = False, 
                         days_back: int = 30) -> Tuple[bool, str]:
    """ETL 파이프라인 실행 (안전한 래퍼)"""
    backend = get_backend_integration()
    return backend.run_etl_pipeline(data_root, force_reload, days_back)


def run_portfolio_optimization_safe(**kwargs) -> Tuple[bool, Dict[str, Any]]:
    """포트폴리오 최적화 실행 (안전한 래퍼)"""
    backend = get_backend_integration()
    return backend.run_portfolio_optimization(**kwargs)


def run_backtesting_safe(**kwargs) -> Tuple[bool, Dict[str, Any]]:
    """백테스팅 실행 (안전한 래퍼)"""
    backend = get_backend_integration()
    return backend.run_backtesting(**kwargs)