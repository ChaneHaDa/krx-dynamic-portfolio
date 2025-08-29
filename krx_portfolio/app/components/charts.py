"""
차트 생성 함수들
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st


def create_cumulative_return_chart(
    returns_data: Dict[str, pd.Series], 
    title: str = "누적 수익률",
    height: int = 500
) -> go.Figure:
    """누적 수익률 차트 생성"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (name, returns) in enumerate(returns_data.items()):
        cumulative = (1 + returns).cumprod()
        
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=cumulative,
            mode='lines',
            name=name,
            line=dict(
                width=2,
                color=colors[i % len(colors)]
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         '날짜: %{x}<br>' +
                         '누적수익률: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="누적 수익률",
        height=height,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_drawdown_chart(
    returns: pd.Series,
    title: str = "최대 낙폭 (Drawdown)",
    height: int = 400
) -> go.Figure:
    """드로다운 차트 생성"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100
    
    fig = go.Figure()
    
    # 드로다운 영역 차트
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill='tonexty',
        mode='none',
        name='Drawdown',
        fillcolor='rgba(255, 0, 0, 0.3)',
        hovertemplate='날짜: %{x}<br>' +
                     'Drawdown: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # 0% 기준선
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="Drawdown (%)",
        height=height,
        showlegend=False
    )
    
    return fig


def create_risk_return_scatter(
    portfolios_data: Dict[str, Dict[str, float]],
    title: str = "위험-수익률 분포",
    height: int = 500
) -> go.Figure:
    """위험-수익률 산점도 생성"""
    fig = go.Figure()
    
    names = list(portfolios_data.keys())
    returns = [data['return'] for data in portfolios_data.values()]
    risks = [data['risk'] for data in portfolios_data.values()]
    sharpe_ratios = [data.get('sharpe', 0) for data in portfolios_data.values()]
    
    # 샤프 비율에 따른 색상 매핑
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers+text',
        text=names,
        textposition="top center",
        marker=dict(
            size=15,
            color=sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="샤프 비율")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                     '위험: %{x:.2f}%<br>' +
                     '수익률: %{y:.2f}%<br>' +
                     '샤프 비율: %{marker.color:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="위험 (변동성, %)",
        yaxis_title="수익률 (%)",
        height=height
    )
    
    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "상관관계 히트맵",
    height: int = 600
) -> go.Figure:
    """상관관계 히트맵 생성"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hovertemplate='행: %{y}<br>' +
                     '열: %{x}<br>' +
                     '상관계수: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig


def create_rolling_metrics_chart(
    returns: pd.Series,
    window: int = 252,
    title: str = "이동 지표",
    height: int = 500
) -> go.Figure:
    """이동 지표 차트 생성 (샤프 비율, 변동성 등)"""
    # 이동 통계 계산
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('이동 연평균 수익률 (%)', '이동 변동성 (%)', '이동 샤프 비율'),
        vertical_spacing=0.08
    )
    
    # 수익률
    fig.add_trace(
        go.Scatter(
            x=rolling_return.index,
            y=rolling_return * 100,
            mode='lines',
            name='연평균 수익률',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # 변동성
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol * 100,
            mode='lines',
            name='변동성',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # 샤프 비율
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name='샤프 비율',
            line=dict(color='green', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False
    )
    
    return fig


def create_returns_distribution(
    returns: pd.Series,
    title: str = "수익률 분포",
    height: int = 400
) -> go.Figure:
    """수익률 분포 히스토그램 생성"""
    fig = go.Figure()
    
    # 히스토그램
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='수익률 분포',
        opacity=0.7,
        marker_color='blue'
    ))
    
    # 평균선
    mean_return = returns.mean() * 100
    fig.add_vline(
        x=mean_return, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"평균: {mean_return:.2f}%"
    )
    
    # VaR 선들
    var_95 = np.percentile(returns * 100, 5)
    var_99 = np.percentile(returns * 100, 1)
    
    fig.add_vline(
        x=var_95, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"VaR 95%: {var_95:.2f}%"
    )
    
    fig.add_vline(
        x=var_99, 
        line_dash="dash", 
        line_color="darkred",
        annotation_text=f"VaR 99%: {var_99:.2f}%"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="일간 수익률 (%)",
        yaxis_title="빈도",
        height=height,
        showlegend=False
    )
    
    return fig


def create_sector_performance_chart(
    sector_returns: Dict[str, float],
    title: str = "섹터별 성과",
    height: int = 400
) -> go.Figure:
    """섹터별 성과 막대 차트"""
    sectors = list(sector_returns.keys())
    returns = list(sector_returns.values())
    
    # 양수/음수에 따른 색상 설정
    colors = ['green' if r >= 0 else 'red' for r in returns]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sectors,
        y=returns,
        marker_color=colors,
        text=[f"{r:.2f}%" for r in returns],
        textposition='outside',
        hovertemplate='섹터: %{x}<br>' +
                     '수익률: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="섹터",
        yaxis_title="수익률 (%)",
        height=height,
        showlegend=False
    )
    
    # 0% 기준선
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
    
    return fig


def create_rebalancing_chart(
    weights_history: pd.DataFrame,
    title: str = "리밸런싱 이력",
    height: int = 500
) -> go.Figure:
    """리밸런싱 이력 스택 차트"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, asset in enumerate(weights_history.columns):
        fig.add_trace(go.Scatter(
            x=weights_history.index,
            y=weights_history[asset] * 100,
            mode='lines',
            name=asset,
            stackgroup='one',
            line=dict(width=0),
            fillcolor=colors[i % len(colors)],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         '날짜: %{x}<br>' +
                         '비중: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="비중 (%)",
        height=height,
        hovermode='x unified'
    )
    
    return fig


def create_performance_attribution_chart(
    attribution_data: pd.DataFrame,
    title: str = "성과 기여도 분석",
    height: int = 500
) -> go.Figure:
    """성과 기여도 분석 차트"""
    fig = go.Figure()
    
    # 양수/음수 기여도에 따른 색상
    colors = ['green' if x >= 0 else 'red' for x in attribution_data['contribution']]
    
    fig.add_trace(go.Bar(
        x=attribution_data.index,
        y=attribution_data['contribution'] * 100,
        marker_color=colors,
        text=[f"{x:.2f}%" for x in attribution_data['contribution'] * 100],
        textposition='outside',
        hovertemplate='자산: %{x}<br>' +
                     '기여도: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="자산",
        yaxis_title="기여도 (%)",
        height=height,
        showlegend=False
    )
    
    # 0% 기준선
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
    
    return fig