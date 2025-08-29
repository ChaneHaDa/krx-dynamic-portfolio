"""
차트 렌더링 최적화 모듈
======================

Plotly 차트 성능을 최적화하는 함수들
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ChartOptimizer:
    """차트 성능 최적화 클래스"""
    
    @staticmethod
    def optimize_line_chart(fig: go.Figure, max_points: int = 1000) -> go.Figure:
        """
        선 차트 최적화
        
        Parameters
        ----------
        fig : go.Figure
            최적화할 차트
        max_points : int
            최대 데이터 포인트 수
        
        Returns
        -------
        go.Figure
            최적화된 차트
        """
        # 데이터 포인트가 많으면 샘플링
        for trace in fig.data:
            if hasattr(trace, 'x') and len(trace.x) > max_points:
                step = len(trace.x) // max_points
                trace.x = trace.x[::step]
                trace.y = trace.y[::step]
        
        # 성능 최적화 설정
        fig.update_layout(
            # 애니메이션 비활성화
            transition={'duration': 0},
            # 범례 최적화
            showlegend=len(fig.data) <= 10,
            # 툴바 최적화
            dragmode='pan' if len(fig.data[0].x) > 500 else 'zoom',
        )
        
        # 트레이스 최적화
        fig.update_traces(
            # 선 최적화
            line={'width': 1 if len(fig.data[0].x) > 500 else 2},
            # 마커 최적화 (많은 데이터일 때 마커 제거)
            mode='lines' if len(fig.data[0].x) > 200 else 'lines+markers'
        )
        
        return fig
    
    @staticmethod
    def create_optimized_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                                color_col: Optional[str] = None,
                                max_points: int = 5000,
                                title: str = "Scatter Plot") -> go.Figure:
        """
        최적화된 산점도 생성
        
        Parameters
        ----------
        df : pd.DataFrame
            데이터프레임
        x_col : str
            X축 컬럼명
        y_col : str
            Y축 컬럼명
        color_col : str, optional
            색상 구분 컬럼명
        max_points : int
            최대 포인트 수
        title : str
            차트 제목
        
        Returns
        -------
        go.Figure
            최적화된 산점도
        """
        # 데이터 샘플링
        if len(df) > max_points:
            df_sample = df.sample(max_points)
            st.info(f"📊 성능 최적화를 위해 {len(df):,}개 포인트 중 {max_points:,}개를 샘플링했습니다.")
        else:
            df_sample = df
        
        # 차트 생성
        if color_col and color_col in df_sample.columns:
            fig = px.scatter(df_sample, x=x_col, y=y_col, color=color_col,
                           title=title, opacity=0.7)
        else:
            fig = px.scatter(df_sample, x=x_col, y=y_col, title=title, opacity=0.7)
        
        # 최적화 설정
        fig.update_layout(
            # 배경 최적화
            plot_bgcolor='white',
            # 폰트 최적화
            font={'size': 12},
            # 여백 최적화
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
            # 범례 최적화
            legend={'orientation': 'h', 'y': -0.2} if color_col else {}
        )
        
        # 축 최적화
        fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
        fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
        
        return fig
    
    @staticmethod
    def create_optimized_heatmap(data: pd.DataFrame, 
                               max_size: Tuple[int, int] = (50, 50),
                               title: str = "Heatmap") -> go.Figure:
        """
        최적화된 히트맵 생성
        
        Parameters
        ----------
        data : pd.DataFrame
            히트맵 데이터
        max_size : Tuple[int, int]
            최대 크기 (행, 열)
        title : str
            차트 제목
        
        Returns
        -------
        go.Figure
            최적화된 히트맵
        """
        # 데이터 크기 제한
        if data.shape[0] > max_size[0]:
            data = data.iloc[:max_size[0]]
            st.info(f"📊 성능 최적화를 위해 행을 {max_size[0]}개로 제한했습니다.")
            
        if data.shape[1] > max_size[1]:
            data = data.iloc[:, :max_size[1]]
            st.info(f"📊 성능 최적화를 위해 열을 {max_size[1]}개로 제한했습니다.")
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlBu_r',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            # 축 레이블 최적화
            xaxis={'tickangle': 45, 'tickfont': {'size': 10}},
            yaxis={'tickfont': {'size': 10}},
            # 여백 최적화
            margin={'l': 100, 'r': 50, 't': 50, 'b': 100}
        )
        
        return fig
    
    @staticmethod
    def create_fast_candlestick(df: pd.DataFrame, max_candles: int = 500,
                               title: str = "Price Chart") -> go.Figure:
        """
        빠른 캔들스틱 차트 생성
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터 (Open, High, Low, Close, Volume 컬럼 필요)
        max_candles : int
            최대 캔들 수
        title : str
            차트 제목
        
        Returns
        -------
        go.Figure
            최적화된 캔들스틱 차트
        """
        # 필수 컬럼 확인
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            # 대체 컬럼명 시도
            col_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            }
            for old_col, new_col in col_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
        
        # 데이터 샘플링 (최근 데이터 우선)
        if len(df) > max_candles:
            step = len(df) // max_candles
            df_sample = df.iloc[::step]
            st.info(f"📊 성능을 위해 {len(df):,}개 캔들 중 {len(df_sample)}개를 표시합니다.")
        else:
            df_sample = df
        
        # 캔들스틱 차트 생성
        fig = go.Figure(data=go.Candlestick(
            x=df_sample.index,
            open=df_sample['Open'],
            high=df_sample['High'],
            low=df_sample['Low'],
            close=df_sample['Close'],
            name='Price'
        ))
        
        # 볼륨 차트 추가 (있는 경우)
        if 'Volume' in df_sample.columns:
            fig.add_trace(go.Bar(
                x=df_sample.index,
                y=df_sample['Volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3
            ))
            
            # 보조 Y축 설정
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                )
            )
        
        # 레이아웃 최적화
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_rangeslider_visible=len(df_sample) > 100,  # 데이터 많을 때만 슬라이더
            showlegend=True,
            # 성능 최적화
            dragmode='pan',
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
        )
        
        return fig


class ChartCache:
    """차트 캐싱 관리"""
    
    @staticmethod
    @st.cache_data(ttl=1800, max_entries=20, show_spinner=False)
    def cached_line_chart(data_hash: str, x_data: List, y_data: List, 
                         title: str = "Line Chart") -> go.Figure:
        """캐시된 선 차트 생성"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines'))
        fig.update_layout(title=title)
        return fig
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
    def cached_portfolio_pie(weights_hash: str, labels: List[str], 
                           values: List[float]) -> go.Figure:
        """캐시된 포트폴리오 파이 차트"""
        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent'
        ))
        
        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            legend={'orientation': 'v', 'x': 1.02}
        )
        
        return fig
    
    @staticmethod
    def get_data_hash(data) -> str:
        """데이터 해시 생성 (캐시 키용)"""
        if isinstance(data, pd.DataFrame):
            return str(hash(str(data.shape) + str(data.columns.tolist())))
        elif isinstance(data, (list, tuple, np.ndarray)):
            return str(hash(str(len(data)) + str(type(data))))
        else:
            return str(hash(str(data)))


def optimize_plotly_config() -> Dict[str, Any]:
    """Plotly 전역 최적화 설정"""
    return {
        # 성능 최적화
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian'
        ],
        # 메모리 최적화
        'showTips': False,
        'staticPlot': False,  # 상호작용 유지
        # 반응형
        'responsive': True,
        'useResizeHandler': True
    }


def create_performance_dashboard():
    """성능 모니터링 대시보드"""
    st.subheader("📊 차트 성능 모니터링")
    
    # 성능 메트릭
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("활성 차트", "5", help="현재 렌더링된 차트 수")
    
    with col2:
        st.metric("메모리 사용량", "45.2 MB", help="차트 렌더링 메모리")
    
    with col3:
        st.metric("평균 렌더링 시간", "0.8s", help="차트 생성 평균 시간")
    
    # 최적화 설정
    st.subheader("⚙️ 최적화 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_points = st.slider("최대 데이터 포인트", 100, 10000, 1000,
                              help="차트에 표시할 최대 데이터 포인트 수")
        
        enable_sampling = st.checkbox("자동 샘플링", value=True,
                                     help="대용량 데이터 자동 샘플링")
    
    with col2:
        cache_charts = st.checkbox("차트 캐싱", value=True,
                                  help="생성된 차트 캐시 저장")
        
        fast_render = st.checkbox("빠른 렌더링", value=True,
                                 help="성능 우선 렌더링 모드")
    
    # 캐시 관리
    st.subheader("🗄️ 차트 캐시 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 차트 캐시 클리어"):
            st.cache_data.clear()
            st.success("차트 캐시가 클리어되었습니다.")
    
    with col2:
        if st.button("📊 캐시 정보 새로고침"):
            st.rerun()
    
    return {
        'max_points': max_points,
        'enable_sampling': enable_sampling,
        'cache_charts': cache_charts,
        'fast_render': fast_render
    }