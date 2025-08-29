"""
성능 최적화 모듈
=================

대시보드 성능을 개선하기 위한 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import psutil
import gc
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


def monitor_memory(func):
    """메모리 사용량 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 실행 전 메모리 체크
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # 실행 후 메모리 체크
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = memory_after - memory_before
        
        logger.info(f"{func.__name__}: {execution_time:.2f}s, Memory: {memory_diff:+.1f}MB")
        
        # 메모리 사용량이 너무 크면 가비지 컬렉션 실행
        if memory_diff > 100:  # 100MB 초과
            gc.collect()
            logger.info(f"Garbage collection executed for {func.__name__}")
            
        return result
    return wrapper


class DataSampler:
    """대용량 데이터 샘플링 및 청크 처리"""
    
    @staticmethod
    def smart_sample(df: pd.DataFrame, target_size: int = 1000, method: str = "uniform") -> pd.DataFrame:
        """
        지능형 데이터 샘플링
        
        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터프레임
        target_size : int
            목표 크기
        method : str
            샘플링 방법 ('uniform', 'stratified', 'recent')
        
        Returns
        -------
        pd.DataFrame
            샘플링된 데이터프레임
        """
        if len(df) <= target_size:
            return df.copy()
        
        if method == "uniform":
            # 균등 간격 샘플링
            step = len(df) // target_size
            return df.iloc[::step].copy()
            
        elif method == "recent":
            # 최근 데이터 우선 샘플링 (시계열 데이터용)
            recent_ratio = 0.7  # 최근 70% 데이터에서 더 많이 샘플링
            recent_size = int(target_size * recent_ratio)
            old_size = target_size - recent_size
            
            split_idx = len(df) // 2
            recent_data = df.iloc[split_idx:].sample(min(recent_size, len(df) - split_idx))
            old_data = df.iloc[:split_idx].sample(min(old_size, split_idx))
            
            return pd.concat([old_data, recent_data]).sort_index()
            
        elif method == "stratified":
            # 계층적 샘플링 (카테고리가 있는 경우)
            if hasattr(df, 'sector') or 'sector' in df.columns:
                return df.groupby('sector').apply(
                    lambda x: x.sample(min(len(x), target_size // df['sector'].nunique()))
                ).reset_index(drop=True)
            else:
                return df.sample(target_size)
        
        return df.sample(target_size)
    
    @staticmethod
    def chunk_processor(data: pd.DataFrame, chunk_size: int = 100, 
                       process_func=None) -> List[Any]:
        """
        청크 단위 데이터 처리
        
        Parameters
        ----------
        data : pd.DataFrame
            처리할 데이터
        chunk_size : int
            청크 크기
        process_func : callable
            각 청크에 적용할 함수
        
        Returns
        -------
        List[Any]
            처리된 결과 리스트
        """
        if process_func is None:
            process_func = lambda x: x
            
        results = []
        total_chunks = (len(data) - 1) // chunk_size + 1
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            result = process_func(chunk)
            results.append(result)
            
            # 진행률 표시 (Streamlit에서)
            if hasattr(st, 'progress'):
                progress = (i // chunk_size + 1) / total_chunks
                st.progress(progress)
        
        return results


class StreamlitOptimizer:
    """Streamlit 특화 최적화"""
    
    @staticmethod
    def lazy_load_dataframe(data_loader_func, key: str, refresh_button: bool = True):
        """
        지연 로딩 데이터프레임
        
        Parameters
        ----------
        data_loader_func : callable
            데이터 로딩 함수
        key : str
            세션 상태 키
        refresh_button : bool
            새로고침 버튼 표시 여부
        
        Returns
        -------
        pd.DataFrame or None
            로딩된 데이터프레임
        """
        # 새로고침 버튼
        if refresh_button:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 새로고침", key=f"refresh_{key}"):
                    if key in st.session_state:
                        del st.session_state[key]
        
        # 데이터 로딩
        if key not in st.session_state:
            with st.spinner("데이터 로딩 중..."):
                try:
                    st.session_state[key] = data_loader_func()
                except Exception as e:
                    st.error(f"데이터 로딩 실패: {e}")
                    return None
        
        return st.session_state[key]
    
    @staticmethod
    def optimized_dataframe_display(df: pd.DataFrame, max_rows: int = 100):
        """
        최적화된 데이터프레임 표시
        
        Parameters
        ----------
        df : pd.DataFrame
            표시할 데이터프레임
        max_rows : int
            최대 표시 행 수
        """
        if len(df) > max_rows:
            st.info(f"📊 전체 {len(df):,}행 중 상위 {max_rows}행만 표시됩니다.")
            
            # 페이지네이션 옵션
            show_pagination = st.checkbox("페이지네이션 사용")
            
            if show_pagination:
                page_size = st.selectbox("페이지 크기", [50, 100, 200], index=1)
                total_pages = (len(df) - 1) // page_size + 1
                page = st.selectbox("페이지", range(1, total_pages + 1))
                
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                display_df = df.iloc[start_idx:end_idx]
            else:
                display_df = df.head(max_rows)
        else:
            display_df = df
        
        st.dataframe(display_df, use_container_width=True)
        
        # 다운로드 버튼
        if len(df) > 0:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv_data,
                file_name=f"data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


class CacheManager:
    """Streamlit 캐시 관리"""
    
    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """캐시 정보 조회"""
        try:
            # Streamlit 캐시 통계 (가능한 경우)
            cache_stats = {
                "total_cached_functions": 0,
                "cache_hits": "N/A",
                "cache_misses": "N/A",
                "memory_usage": "N/A"
            }
            
            # 시스템 메모리 정보
            memory = psutil.virtual_memory()
            cache_stats.update({
                "system_memory_total": f"{memory.total / 1024**3:.1f} GB",
                "system_memory_available": f"{memory.available / 1024**3:.1f} GB",
                "system_memory_percent": f"{memory.percent:.1f}%"
            })
            
            return cache_stats
        except Exception as e:
            logger.error(f"Cache info retrieval failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def clear_all_caches():
        """모든 Streamlit 캐시 클리어"""
        try:
            st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            # 가비지 컬렉션 실행
            gc.collect()
            
            return True, "모든 캐시가 클리어되었습니다."
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return False, f"캐시 클리어 실패: {e}"
    
    @staticmethod
    def selective_cache_clear(cache_keys: List[str]):
        """선택적 캐시 클리어"""
        cleared_count = 0
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]
                cleared_count += 1
        
        if cleared_count > 0:
            gc.collect()
        
        return f"{cleared_count}개 캐시 항목이 클리어되었습니다."


class PerformanceProfiler:
    """성능 프로파일링 도구"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        
    def start(self):
        """프로파일링 시작"""
        self.start_time = time.time()
        self.checkpoints = []
        
    def checkpoint(self, name: str):
        """체크포인트 추가"""
        if self.start_time is None:
            self.start()
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.checkpoints:
            interval = current_time - self.checkpoints[-1]['timestamp']
        else:
            interval = elapsed
            
        self.checkpoints.append({
            'name': name,
            'elapsed_total': elapsed,
            'elapsed_interval': interval,
            'timestamp': current_time,
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
    
    def get_report(self) -> pd.DataFrame:
        """성능 리포트 반환"""
        if not self.checkpoints:
            return pd.DataFrame()
            
        return pd.DataFrame(self.checkpoints)
    
    def display_report(self):
        """Streamlit에서 성능 리포트 표시"""
        report_df = self.get_report()
        
        if report_df.empty:
            st.warning("성능 데이터가 없습니다.")
            return
            
        st.subheader("🔍 성능 프로파일 리포트")
        
        # 요약 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 실행 시간", f"{report_df['elapsed_total'].iloc[-1]:.2f}s")
        with col2:
            st.metric("평균 메모리", f"{report_df['memory_mb'].mean():.1f} MB")
        with col3:
            st.metric("최대 메모리", f"{report_df['memory_mb'].max():.1f} MB")
        
        # 상세 리포트
        display_df = report_df[['name', 'elapsed_interval', 'elapsed_total', 'memory_mb']].copy()
        display_df.columns = ['단계', '구간 시간(s)', '누적 시간(s)', '메모리(MB)']
        display_df['구간 시간(s)'] = display_df['구간 시간(s)'].round(3)
        display_df['누적 시간(s)'] = display_df['누적 시간(s)'].round(3)
        display_df['메모리(MB)'] = display_df['메모리(MB)'].round(1)
        
        st.dataframe(display_df, use_container_width=True)


# 전역 프로파일러 인스턴스
_performance_profiler = None

def get_profiler() -> PerformanceProfiler:
    """전역 프로파일러 인스턴스 반환"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler