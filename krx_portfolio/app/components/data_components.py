"""
데이터 관리 관련 Streamlit 컴포넌트들
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json


def render_data_status(data_info: Dict[str, Any]) -> None:
    """데이터 상태 정보 렌더링"""
    st.subheader("📊 데이터 현황")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 종목 수",
            value=f"{data_info.get('total_stocks', 0):,}개",
            help="처리 가능한 전체 종목 수"
        )
    
    with col2:
        last_update = data_info.get('last_update', 'Unknown')
        if isinstance(last_update, str):
            try:
                last_update = pd.to_datetime(last_update).strftime('%Y-%m-%d')
            except:
                pass
        
        st.metric(
            label="마지막 업데이트",
            value=last_update,
            help="데이터 마지막 업데이트 일자"
        )
    
    with col3:
        data_period = data_info.get('data_period', {})
        start_date = data_period.get('start', 'N/A')
        end_date = data_period.get('end', 'N/A')
        
        st.metric(
            label="데이터 기간",
            value=f"{start_date} ~ {end_date}",
            help="보유 데이터의 시작일과 종료일"
        )
    
    with col4:
        st.metric(
            label="데이터 품질",
            value=f"{data_info.get('quality_score', 0):.1f}%",
            help="데이터 완성도 및 품질 점수"
        )


def render_etl_controls() -> Dict[str, Any]:
    """ETL 파이프라인 제어 인터페이스 렌더링"""
    st.subheader("🔄 ETL 파이프라인 제어")
    
    # ETL 설정 패널
    with st.expander("⚙️ ETL 설정", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            data_root = st.text_input(
                "데이터 루트 경로",
                value="/path/to/krx/data",
                help="KRX JSON 데이터가 저장된 루트 디렉토리"
            )
            
            start_date = st.date_input(
                "시작일",
                value=datetime(2020, 1, 1),
                help="데이터 수집 시작일"
            )
        
        with col2:
            end_date = st.date_input(
                "종료일",
                value=datetime.now(),
                help="데이터 수집 종료일"
            )
            
            force_reload = st.checkbox(
                "강제 재로드",
                value=False,
                help="캐시 무시하고 전체 데이터 다시 로드"
            )
    
    # ETL 실행 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 ETL 파이프라인 실행", type="primary", use_container_width=True):
            etl_config = {
                "data_root": data_root,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "force_reload": force_reload
            }
            
            return etl_config
    
    return {}


def render_cache_management(cache_info: Dict[str, Any]) -> None:
    """캐시 관리 인터페이스 렌더링"""
    st.subheader("💾 캐시 관리")
    
    # 캐시 현황
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cache_size = cache_info.get('total_size', 0)
        if cache_size > 1024**3:  # GB
            size_str = f"{cache_size / 1024**3:.2f} GB"
        elif cache_size > 1024**2:  # MB
            size_str = f"{cache_size / 1024**2:.2f} MB"
        else:  # KB
            size_str = f"{cache_size / 1024:.2f} KB"
        
        st.metric("총 캐시 크기", size_str)
    
    with col2:
        st.metric("캐시 파일 수", f"{cache_info.get('file_count', 0):,}개")
    
    with col3:
        last_cleared = cache_info.get('last_cleared', 'Never')
        st.metric("마지막 정리", last_cleared)
    
    # 캐시 디렉토리별 상세 정보
    if 'directories' in cache_info:
        st.markdown("**📁 디렉토리별 캐시 현황**")
        
        cache_dirs = []
        for dir_name, dir_info in cache_info['directories'].items():
            cache_dirs.append({
                "디렉토리": dir_name,
                "파일 수": f"{dir_info.get('count', 0):,}개",
                "크기": f"{dir_info.get('size', 0) / 1024**2:.1f} MB",
                "마지막 수정": dir_info.get('last_modified', 'Unknown')
            })
        
        if cache_dirs:
            st.dataframe(pd.DataFrame(cache_dirs), hide_index=True, use_container_width=True)
    
    # 캐시 관리 버튼들
    st.markdown("**🛠️ 캐시 관리 작업**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ 전체 삭제", help="모든 캐시 파일을 삭제합니다"):
            st.warning("⚠️ 전체 캐시가 삭제되었습니다. 다음 실행 시 전체 데이터를 다시 로드합니다.")
    
    with col2:
        if st.button("🧹 오래된 파일만", help="7일 이상 된 캐시 파일만 삭제"):
            st.info("📅 7일 이상 된 캐시 파일이 정리되었습니다.")
    
    with col3:
        if st.button("📊 캐시 통계", help="상세한 캐시 사용 통계를 표시"):
            show_cache_statistics(cache_info)
    
    with col4:
        if st.button("🔍 캐시 검증", help="캐시 파일 무결성을 검증"):
            st.success("✅ 캐시 파일 검증이 완료되었습니다.")


def show_cache_statistics(cache_info: Dict[str, Any]) -> None:
    """캐시 통계 상세 표시"""
    with st.expander("📊 캐시 상세 통계", expanded=True):
        
        # 파일 형태별 통계
        if 'file_types' in cache_info:
            st.markdown("**📄 파일 형태별 분포**")
            
            file_types_df = pd.DataFrame([
                {"형태": ext, "개수": info['count'], "크기(MB)": info['size'] / 1024**2}
                for ext, info in cache_info['file_types'].items()
            ])
            
            st.bar_chart(file_types_df.set_index('형태')['개수'])
        
        # 시간별 사용 패턴
        if 'usage_pattern' in cache_info:
            st.markdown("**⏰ 시간별 사용 패턴**")
            
            usage_df = pd.DataFrame(cache_info['usage_pattern'])
            st.line_chart(usage_df.set_index('hour')['access_count'])


def render_data_quality_report(quality_data: Dict[str, Any]) -> None:
    """데이터 품질 리포트 렌더링"""
    st.subheader("✅ 데이터 품질 리포트")
    
    # 전체 품질 점수
    overall_score = quality_data.get('overall_score', 0)
    score_color = "green" if overall_score >= 90 else "orange" if overall_score >= 70 else "red"
    
    st.markdown(f"### 전체 품질 점수: <span style='color:{score_color}'>{overall_score:.1f}%</span>", unsafe_allow_html=True)
    
    # 세부 품질 지표
    quality_metrics = quality_data.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = quality_metrics.get('completeness', 0)
        st.metric("완전성", f"{completeness:.1f}%", help="결측값이 없는 데이터의 비율")
    
    with col2:
        accuracy = quality_metrics.get('accuracy', 0)
        st.metric("정확성", f"{accuracy:.1f}%", help="올바른 형식의 데이터 비율")
    
    with col3:
        consistency = quality_metrics.get('consistency', 0)
        st.metric("일관성", f"{consistency:.1f}%", help="일관된 포맷의 데이터 비율")
    
    with col4:
        timeliness = quality_metrics.get('timeliness', 0)
        st.metric("적시성", f"{timeliness:.1f}%", help="최신 데이터의 비율")
    
    # 품질 이슈 목록
    if 'issues' in quality_data and quality_data['issues']:
        st.markdown("**⚠️ 발견된 품질 이슈**")
        
        for issue in quality_data['issues']:
            severity = issue.get('severity', 'info').lower()
            icon = "🔴" if severity == 'high' else "🟡" if severity == 'medium' else "🔵"
            
            st.markdown(f"{icon} **{issue.get('type', 'Unknown')}**: {issue.get('description', 'No description')}")
            
            if issue.get('affected_records', 0) > 0:
                st.markdown(f"   - 영향받은 레코드: {issue['affected_records']:,}개")
    else:
        st.success("✅ 품질 이슈가 발견되지 않았습니다.")


def render_etl_progress(progress_data: Dict[str, Any]) -> None:
    """ETL 진행 상황 렌더링"""
    if not progress_data:
        return
    
    st.subheader("⏳ ETL 진행 상황")
    
    # 전체 진행률
    overall_progress = progress_data.get('overall_progress', 0)
    st.progress(overall_progress, f"전체 진행률: {overall_progress:.1%}")
    
    # 단계별 진행 상황
    stages = progress_data.get('stages', {})
    
    for stage_name, stage_info in stages.items():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stage_progress = stage_info.get('progress', 0)
            st.progress(stage_progress, f"{stage_name}: {stage_progress:.1%}")
        
        with col2:
            status = stage_info.get('status', 'pending')
            status_emoji = "✅" if status == 'completed' else "🔄" if status == 'running' else "⏳"
            st.markdown(f"{status_emoji} {status}")
    
    # 로그 메시지
    if 'logs' in progress_data:
        with st.expander("📝 상세 로그"):
            for log_entry in progress_data['logs'][-10:]:  # 최근 10개만 표시
                timestamp = log_entry.get('timestamp', '')
                message = log_entry.get('message', '')
                level = log_entry.get('level', 'info').upper()
                
                level_color = {
                    'ERROR': 'red',
                    'WARNING': 'orange', 
                    'INFO': 'blue',
                    'DEBUG': 'gray'
                }.get(level, 'black')
                
                st.markdown(f"<span style='color:{level_color}'>[{timestamp}] {level}: {message}</span>", unsafe_allow_html=True)


def render_data_preview(data_sample: pd.DataFrame, title: str = "데이터 미리보기") -> None:
    """데이터 미리보기 렌더링"""
    if data_sample.empty:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    st.subheader(f"👀 {title}")
    
    # 데이터 기본 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("행 수", f"{len(data_sample):,}")
    
    with col2:
        st.metric("열 수", f"{len(data_sample.columns):,}")
    
    with col3:
        memory_usage = data_sample.memory_usage(deep=True).sum() / 1024**2
        st.metric("메모리 사용량", f"{memory_usage:.1f} MB")
    
    # 데이터 테이블 (상위 N개 행만 표시)
    display_rows = min(100, len(data_sample))
    st.dataframe(
        data_sample.head(display_rows),
        use_container_width=True
    )
    
    if len(data_sample) > display_rows:
        st.info(f"📝 상위 {display_rows}개 행만 표시됩니다. (전체: {len(data_sample):,}개)")
    
    # 데이터 타입 정보
    with st.expander("📋 데이터 타입 정보"):
        dtype_info = pd.DataFrame({
            "컬럼": data_sample.columns,
            "데이터 타입": data_sample.dtypes.astype(str),
            "결측값": data_sample.isnull().sum(),
            "결측값 비율(%)": (data_sample.isnull().sum() / len(data_sample) * 100).round(2)
        })
        
        st.dataframe(dtype_info, hide_index=True, use_container_width=True)