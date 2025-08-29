"""
유틸리티 함수들
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Parameters
    ----------
    config_path : str, optional
        설정 파일 경로. None이면 기본 경로 사용
    
    Returns
    -------
    Dict[str, Any]
        설정 딕셔너리
    """
    if config_path is None:
        # 기본 설정 파일 경로
        config_path = Path(__file__).parent.parent / "configs" / "portfolio.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # 기본 설정 반환
        return get_default_config()
    except Exception as e:
        raise ValueError(f"설정 파일 로드 실패: {e}")


def get_default_config() -> Dict[str, Any]:
    """
    기본 설정 반환
    
    Returns
    -------
    Dict[str, Any]
        기본 설정 딕셔너리
    """
    return {
        "optimization": {
            "method": "max_sharpe",
            "risk_model": "ledoit_wolf",
            "max_weight": 0.1,
            "min_weight": 0.0,
            "lookback_days": 252
        },
        "rebalancing": {
            "schedule": "monthly",
            "turnover_budget": 0.5,
            "transaction_cost_bps": 25.0
        },
        "backtesting": {
            "initial_capital": 100000000,
            "transaction_cost_bps": 25.0,
            "market_impact_model": "linear",
            "cash_rate": 0.025
        },
        "risk": {
            "var_confidence": 0.05,
            "cvar_confidence": 0.05,
            "lookback_days": 252
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    설정 파일 저장
    
    Parameters
    ----------
    config : Dict[str, Any]
        저장할 설정 딕셔너리
    config_path : str
        저장할 파일 경로
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 딕셔너리 병합
    
    Parameters
    ----------
    base_config : Dict[str, Any]
        기본 설정
    update_config : Dict[str, Any]  
        업데이트할 설정
    
    Returns
    -------
    Dict[str, Any]
        병합된 설정
    """
    result = base_config.copy()
    
    for key, value in update_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result