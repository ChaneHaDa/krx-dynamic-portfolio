"""
포트폴리오 관련 Streamlit 컴포넌트들
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any


def render_portfolio_summary(portfolio_data: Dict[str, Any]) -> None:
    """포트폴리오 요약 정보 렌더링"""
    st.subheader("📊 포트폴리오 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 자산",
            value=f"₩{portfolio_data.get('total_value', 0):,.0f}",
            delta=f"{portfolio_data.get('daily_change_pct', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            label="보유 종목 수",
            value=f"{portfolio_data.get('num_holdings', 0)}개",
            help="현재 포트폴리오 보유 종목 수"
        )
    
    with col3:
        st.metric(
            label="예상 수익률",
            value=f"{portfolio_data.get('expected_return', 0):.2f}%",
            help="연간 예상 수익률"
        )
    
    with col4:
        st.metric(
            label="변동성",
            value=f"{portfolio_data.get('volatility', 0):.2f}%",
            help="연간 변동성"
        )


def render_allocation_chart(weights: pd.Series, names: Optional[List[str]] = None) -> None:
    """포트폴리오 자산배분 차트 렌더링"""
    st.subheader("🥧 자산 배분")
    
    if names is None:
        names = weights.index.tolist()
    
    # 상위 10개 종목만 표시, 나머지는 기타로 합계
    if len(weights) > 10:
        top_10 = weights.nlargest(10)
        others_sum = weights.iloc[10:].sum()
        
        display_weights = pd.concat([top_10, pd.Series([others_sum], index=["기타"])])
        display_names = top_10.index.tolist() + ["기타"]
    else:
        display_weights = weights
        display_names = names[:len(weights)]
    
    # 파이 차트 생성
    fig = px.pie(
        values=display_weights.values,
        names=display_names,
        title="포트폴리오 구성 비중",
        hole=0.3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>' +
                     '비중: %{percent}<br>' +
                     '가치: %{value:.2f}%<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 테이블
    with st.expander("📋 상세 보유 현황"):
        allocation_df = pd.DataFrame({
            "종목": weights.index,
            "비중(%)": weights.values * 100,
            "가치(원)": weights.values * st.session_state.get('total_portfolio_value', 100000000)
        }).round(2)
        
        st.dataframe(
            allocation_df,
            use_container_width=True,
            hide_index=True
        )


def render_performance_metrics(metrics: Dict[str, float]) -> None:
    """성과 지표 메트릭 렌더링"""
    st.subheader("📈 성과 지표")
    
    # 주요 지표들을 그룹으로 나누어 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 수익률 지표**")
        st.metric("총 수익률", f"{metrics.get('total_return', 0):.2f}%")
        st.metric("연평균 수익률", f"{metrics.get('annualized_return', 0):.2f}%")
        st.metric("월평균 수익률", f"{metrics.get('monthly_return', 0):.2f}%")
    
    with col2:
        st.markdown("**⚡ 위험조정 수익률**")
        st.metric("샤프 비율", f"{metrics.get('sharpe_ratio', 0):.3f}")
        st.metric("소르티노 비율", f"{metrics.get('sortino_ratio', 0):.3f}")
        st.metric("칼마 비율", f"{metrics.get('calmar_ratio', 0):.3f}")
    
    with col3:
        st.markdown("**⚠️ 위험 지표**")
        st.metric("최대 낙폭", f"{metrics.get('max_drawdown', 0):.2f}%")
        st.metric("변동성", f"{metrics.get('volatility', 0):.2f}%")
        st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2f}%")


def render_sector_allocation(sector_weights: pd.Series) -> None:
    """섹터별 자산배분 렌더링"""
    st.subheader("🏭 섹터별 배분")
    
    # 수평 막대 차트
    fig = px.bar(
        x=sector_weights.values * 100,
        y=sector_weights.index,
        orientation='h',
        title="섹터별 비중 (%)",
        labels={'x': '비중 (%)', 'y': '섹터'}
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_risk_metrics(risk_data: Dict[str, float]) -> None:
    """위험 지표 상세 렌더링"""
    st.subheader("⚠️ 위험 분석")
    
    # 위험 지표 테이블
    risk_df = pd.DataFrame([
        {"지표": "Value at Risk (95%)", "값": f"{risk_data.get('var_95', 0):.2f}%", "설명": "95% 신뢰구간 최대 손실"},
        {"지표": "Conditional VaR (95%)", "값": f"{risk_data.get('cvar_95', 0):.2f}%", "설명": "VaR 초과 시 평균 손실"},
        {"지표": "베타", "값": f"{risk_data.get('beta', 0):.3f}", "설명": "시장 대비 민감도"},
        {"지표": "추적 오차", "값": f"{risk_data.get('tracking_error', 0):.2f}%", "설명": "벤치마크 대비 추적 오차"},
        {"지표": "정보 비율", "값": f"{risk_data.get('information_ratio', 0):.3f}", "설명": "초과수익/추적오차"},
        {"지표": "하방 편차", "값": f"{risk_data.get('downside_deviation', 0):.2f}%", "설명": "목표 대비 하방 위험"}
    ])
    
    st.dataframe(risk_df, use_container_width=True, hide_index=True)


def render_rebalancing_schedule(rebalance_dates: List[str], next_rebalance: str) -> None:
    """리밸런싱 일정 렌더링"""
    st.subheader("⚖️ 리밸런싱 일정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📅 다음 리밸런싱: {next_rebalance}")
        st.info(f"🔄 총 리밸런싱 횟수: {len(rebalance_dates)}회")
    
    with col2:
        if rebalance_dates:
            with st.expander("🗓️ 리밸런싱 이력"):
                for date in rebalance_dates[-10:]:  # 최근 10회만 표시
                    st.text(date)


def render_portfolio_comparison(portfolios: Dict[str, pd.Series]) -> None:
    """여러 포트폴리오 비교 렌더링"""
    st.subheader("🔍 포트폴리오 비교")
    
    if len(portfolios) < 2:
        st.warning("비교할 포트폴리오가 부족합니다.")
        return
    
    # 성과 비교 테이블
    comparison_data = []
    for name, returns in portfolios.items():
        total_return = (returns + 1).prod() - 1
        volatility = returns.std() * np.sqrt(252)  # 연환산
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        max_dd = ((returns + 1).cumprod() / (returns + 1).cumprod().expanding().max() - 1).min()
        
        comparison_data.append({
            "포트폴리오": name,
            "총수익률(%)": f"{total_return * 100:.2f}",
            "변동성(%)": f"{volatility * 100:.2f}",
            "샤프비율": f"{sharpe:.3f}",
            "최대낙폭(%)": f"{max_dd * 100:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # 누적 수익률 차트
    fig = go.Figure()
    
    for name, returns in portfolios.items():
        cumulative = (returns + 1).cumprod()
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=cumulative,
            mode='lines',
            name=name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="포트폴리오별 누적 수익률 비교",
        xaxis_title="날짜",
        yaxis_title="누적 수익률",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)