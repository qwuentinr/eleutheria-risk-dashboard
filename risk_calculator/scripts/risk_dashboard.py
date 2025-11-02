# -*- coding: utf-8 -*-
"""
SOVEREIGN RISK DASHBOARD - Streamlit Interface
Visualizes default probability assessments from JSON outputs
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import base64
from PIL import Image

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()  # risk_calculator directory
JSON_DIR = PROJECT_DIR / "json"
IMAGES_DIR = PROJECT_DIR / "images"

# Get logo for page icon
logo_path = IMAGES_DIR / "logo_1.webp"
page_icon = None
if logo_path.exists():
    try:
        page_icon = Image.open(logo_path)
    except Exception:
        page_icon = "E"  # Fallback to "E" for Eleutheria if image can't be loaded
else:
    page_icon = "E"  # Fallback to "E" for Eleutheria if logo doesn't exist

# Configure page
st.set_page_config(
    page_title="Eleutheria - Sovereign Risk Dashboard",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)
# All Eurozone members
AVAILABLE_COUNTRIES = [
    "Austria",
    "Belgium",
    "Croatia",
    "Cyprus",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Portugal",
    "Slovakia",
    "Slovenia",
    "Spain"
]


def load_assessment(country: str) -> Optional[Dict[str, Any]]:
    """Load assessment JSON for a given country."""
    json_path = JSON_DIR / f"{country}_default_probability_assessment.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading assessment for {country}: {e}")
        return None


def get_rating_color(rating: str) -> str:
    """Get color for rating badge."""
    rating_map = {
        'AAA': '#00AA00',
        'AA+': '#44AA00', 'AA': '#66AA00', 'AA-': '#88AA00',
        'A+': '#AAAA00', 'A': '#CCAA00', 'A-': '#EEAA00',
        'BBB+': '#FFBB00', 'BBB': '#FF9900', 'BBB-': '#FF7700',
        'BB+': '#FF5500', 'BB': '#FF3300', 'BB-': '#FF1100',
        'B+': '#FF0000', 'B': '#DD0000', 'B-': '#BB0000',
        'CCC+': '#990000', 'CCC': '#770000', 'CCC-': '#550000',
        'CC': '#330000', 'C': '#110000', 'D': '#000000'
    }
    return rating_map.get(rating, '#666666')


def get_status_color(status: str) -> str:
    """Get color for threshold status."""
    status_map = {
        'safe': '#00AA00',
        'warning': '#FFA500',
        'critical': '#FF0000'
    }
    return status_map.get(status, '#666666')


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


@st.cache_data
def load_all_assessments() -> Dict[str, Dict[str, Any]]:
    """Load all assessments and cache them."""
    assessments = {}
    for country in AVAILABLE_COUNTRIES:
        assessment = load_assessment(country)
        if assessment:
            assessments[country] = assessment
    return assessments


def load_time_series(country: str) -> Optional[Dict[str, Any]]:
    """Load time series analysis JSON for a given country."""
    json_path = JSON_DIR / f"{country}_time_series_analysis.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading time series for {country}: {e}")
        return None


@st.cache_data
def load_all_time_series() -> Dict[str, Dict[str, Any]]:
    """Load all time series analyses and cache them."""
    time_series = {}
    for country in AVAILABLE_COUNTRIES:
        ts_data = load_time_series(country)
        if ts_data:
            time_series[country] = ts_data
    return time_series


def render_overview_tab(assessment: Dict[str, Any]):
    """Render the overview tab with key metrics."""
    overall = assessment['overall_assessment']
    metadata = assessment['metadata']
    country_name = metadata.get('country', 'Country')
    
    st.header(f"Overview - {country_name}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "5-Year Default Probability",
            f"{overall['default_probability_5yr']:.2f}%",
            delta=None
        )
    
    with col2:
        rating = overall['rating']
        rating_color = get_rating_color(rating)
        st.markdown(
            f"""
            <div style="background-color: {rating_color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {rating_color};">
                <h3 style="margin: 0; color: {rating_color};">{rating}</h3>
                <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">Rating</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        outlook = overall['rating_outlook']
        st.metric(
            "Rating Outlook",
            outlook,
            delta=None
        )
    
    st.divider()
    
    # Metadata (without country)
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Assessment Date:** {metadata['assessment_date']}")
    with col2:
        st.info(f"**Forecast Year:** {metadata['forecast_year']}")
    
    # Uncertainty visualization
    st.subheader("Uncertainty Analysis")
    uncertainty = assessment['uncertainty']
    ci = uncertainty['confidence_interval_90']
    median = uncertainty['median_probability']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Median Probability", f"{median:.2f}%")
    with col2:
        st.metric(
            "90% Confidence Interval",
            f"[{ci['lower']:.2f}%, {ci['upper']:.2f}%]"
        )
    with col3:
        st.metric("Volatility", f"{uncertainty['volatility']:.2f}")


def render_scenarios_tab(assessment: Dict[str, Any]):
    """Render the scenario analysis tab."""
    st.header("Scenario Analysis")
    
    scenarios = assessment['scenario_analysis']
    
    # Scenario comparison chart
    scenario_data = {
        'Scenario': ['Base Case', 'Upside Case', 'Downside Case'],
        'Probability (%)': [
            scenarios['base_case']['probability'],
            scenarios['upside_case']['probability'],
            scenarios['downside_case']['probability']
        ],
        'Weight': [
            scenarios['base_case']['weight'],
            scenarios['upside_case']['weight'],
            scenarios['downside_case']['weight']
        ]
    }
    
    df_scenarios = pd.DataFrame(scenario_data)
    
    # Create bar chart
    fig = px.bar(
        df_scenarios,
        x='Scenario',
        y='Probability (%)',
        color='Scenario',
        color_discrete_map={
            'Base Case': '#3498db',
            'Upside Case': '#2ecc71',
            'Downside Case': '#e74c3c'
        },
        title="Scenario Probabilities",
        text='Probability (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Default Probability (%)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Base Case")
        st.metric(
            "Probability",
            f"{scenarios['base_case']['probability']:.2f}%",
            delta=None
        )
        st.caption(f"Weight: {scenarios['base_case']['weight']*100:.0f}%")
    
    with col2:
        st.subheader("Upside Case")
        upside_diff = scenarios['upside_case']['probability'] - scenarios['base_case']['probability']
        st.metric(
            "Probability",
            f"{scenarios['upside_case']['probability']:.2f}%",
            delta=f"{upside_diff:.2f}%"
        )
        st.caption(f"Weight: {scenarios['upside_case']['weight']*100:.0f}%")
    
    with col3:
        st.subheader("Downside Case")
        downside_diff = scenarios['downside_case']['probability'] - scenarios['base_case']['probability']
        st.metric(
            "Probability",
            f"{scenarios['downside_case']['probability']:.2f}%",
            delta=f"{downside_diff:.2f}%"
        )
        st.caption(f"Weight: {scenarios['downside_case']['weight']*100:.0f}%")
    
    st.divider()
    
    # Expected probability
    st.metric(
        "Expected Probability (Weighted Average)",
        f"{scenarios['expected_probability']:.2f}%",
        delta=None
    )


def render_risk_factors_tab(assessment: Dict[str, Any]):
    """Render the key risk factors tab."""
    st.header("Key Risk Drivers")
    
    risk_factors = assessment['key_drivers']['top_risk_factors']
    
    if not risk_factors:
        st.warning("No risk factor data available.")
        return
    
    # Create DataFrame
    df_risk = pd.DataFrame(risk_factors)
    
    # Sort by absolute contribution
    df_risk['abs_contribution'] = df_risk['contribution'].abs()
    df_risk = df_risk.sort_values('abs_contribution', ascending=True)
    
    # Create horizontal bar chart
    colors = ['#e74c3c' if d == 'negative' else '#2ecc71' for d in df_risk['direction']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_risk['feature'],
        x=df_risk['contribution'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{x:.3f}" for x in df_risk['contribution']],
        textposition='outside',
        name='Contribution'
    ))
    
    fig.update_layout(
        title="Top 10 Risk Factor Contributions",
        xaxis_title="Contribution to Default Probability",
        yaxis_title="Risk Factor",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Breakdown")
    
    # Format for display
    display_df = df_risk[['feature', 'contribution', 'direction']].copy()
    display_df['contribution'] = display_df['contribution'].apply(lambda x: f"{x:.3f}")
    display_df.columns = ['Risk Factor', 'Contribution', 'Direction']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_thresholds_tab(assessment: Dict[str, Any]):
    """Render the threshold analysis tab."""
    st.header("Threshold Analysis")
    
    thresholds = assessment['threshold_analysis']
    
    if not thresholds:
        st.warning("No threshold data available.")
        return
    
    # Create cards for each threshold
    threshold_names = {
        'debt_gdp': 'Debt / GDP',
        'deficit_gdp': 'Deficit / GDP',
        'current_account_gdp': 'Current Account / GDP',
        'inflation_5yr': 'Inflation (5yr avg)',
        'reserves_months': 'Reserves (months)'
    }
    
    # Group thresholds into columns
    cols = st.columns(len(thresholds))
    
    for idx, (key, data) in enumerate(thresholds.items()):
        with cols[idx % len(cols)]:
            status = data['status']
            status_color = get_status_color(status)
            display_name = threshold_names.get(key, key.replace('_', ' ').title())
            
            st.markdown(
                f"""
                <div style="background-color: {status_color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {status_color}; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 10px 0;">{display_name}</h4>
                    <p style="font-size: 1.5em; margin: 0; font-weight: bold; color: {status_color};">
                        {data['value']:.2f}
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #666;">
                        Status: <strong style="color: {status_color};">{status.upper()}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # Detailed threshold visualization
    st.subheader("Threshold Visualization")
    
    threshold_viz_data = []
    for key, data in thresholds.items():
        display_name = threshold_names.get(key, key.replace('_', ' ').title())
        value = data['value']
        thresh = data['thresholds']
        status = data['status']
        
        threshold_viz_data.append({
            'Metric': display_name,
            'Value': value,
            'Safe Threshold': thresh.get('safe', 0),
            'Warning Threshold': thresh.get('warning', 0),
            'Danger Threshold': thresh.get('danger', 0),
            'Status': status
        })
    
    df_thresholds = pd.DataFrame(threshold_viz_data)
    
    # Create visualization
    fig = go.Figure()
    
    for _, row in df_thresholds.iterrows():
        metric = row['Metric']
        value = row['Value']
        safe = row['Safe Threshold']
        warning = row['Warning Threshold']
        danger = row['Danger Threshold']
        status = row['Status']
        
        # Add zones
        fig.add_trace(go.Scatter(
            x=[metric, metric, metric, metric],
            y=[0, safe, warning, danger],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Current value
        color = get_status_color(status)
        fig.add_trace(go.Scatter(
            x=[metric],
            y=[value],
            mode='markers',
            marker=dict(size=15, color=color, symbol='circle'),
            name=metric,
            text=f"{value:.2f}",
            textposition='middle right'
        ))
    
    fig.update_layout(
        title="Threshold Status Across Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_features_tab(assessment: Dict[str, Any]):
    """Render the features tab showing raw and adjusted features."""
    st.header("Model Features")
    
    raw_features = assessment.get('raw_features', {})
    adjusted_features = assessment.get('adjusted_features', {})
    
    if not raw_features:
        st.warning("No feature data available.")
        return
    
    # Create comparison DataFrame
    feature_data = []
    for key in raw_features.keys():
        raw_val = raw_features.get(key)
        adj_val = adjusted_features.get(key) if adjusted_features else None
        
        feature_data.append({
            'Feature': key.replace('_', ' ').title(),
            'Raw Value': raw_val if raw_val is not None else None,
            'Adjusted Value': adj_val if adj_val is not None else None,
            'Change': (adj_val - raw_val) if (raw_val is not None and adj_val is not None) else None
        })
    
    df_features = pd.DataFrame(feature_data)
    
    # Display comparison
    st.subheader("Raw vs Adjusted Features")
    
    # Format for display
    display_df = df_features.copy()
    for col in ['Raw Value', 'Adjusted Value', 'Change']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
    
    display_df.columns = ['Feature', 'Raw Value', 'Adjusted Value', 'Change']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Feature comparison chart
    if len(df_features) > 0:
        st.subheader("Feature Comparison")
        
        # Select top N features for visualization
        # Create absolute value column for sorting
        df_features['Abs Raw Value'] = df_features['Raw Value'].apply(
            lambda x: abs(x) if x is not None else 0
        )
        top_features = df_features.nlargest(10, 'Abs Raw Value')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Raw Value',
            x=top_features['Feature'],
            y=top_features['Raw Value'],
            marker_color='lightblue'
        ))
        
        if adjusted_features:
            fig.add_trace(go.Bar(
                name='Adjusted Value',
                x=top_features['Feature'],
                y=top_features['Adjusted Value'],
                marker_color='lightcoral'
            ))
        
        fig.update_layout(
            title="Top 10 Features: Raw vs Adjusted",
            xaxis_title="Feature",
            yaxis_title="Value",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_time_series_tab(time_series_data: Optional[Dict[str, Any]], 
                          country: str, 
                          current_assessment: Dict[str, Any]):
    """Render the time series analysis tab."""
    st.header("Time Series Analysis")
    
    if not time_series_data:
        st.warning(f"No time series data found for {country}. Please run the analysis with time series option.")
        return
    
    ts_data = time_series_data.get('time_series', [])
    summary = time_series_data.get('summary', {})
    metadata = time_series_data.get('metadata', {})
    
    if not ts_data:
        st.warning("Time series data is empty.")
        return
    
    # Create DataFrame from time series
    df_ts = pd.DataFrame(ts_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Min PD",
            f"{summary.get('min_probability', 0):.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Max PD",
            f"{summary.get('max_probability', 0):.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Average PD",
            f"{summary.get('average_probability', 0):.2f}%",
            delta=None
        )
    
    with col4:
        trend = summary.get('trend', 'stable')
        st.metric(
            "Trend",
            f"{trend.title()}",
            delta=None
        )
    
    st.divider()
    
    # Main time series chart - Default Probability
    st.subheader("Default Probability Over Time")
    
    fig_prob = go.Figure()
    
    fig_prob.add_trace(go.Scatter(
        x=df_ts['year'],
        y=df_ts['default_probability'],
        mode='lines+markers',
        name='Default Probability',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8, color='#e74c3c'),
        hovertemplate='<b>Year:</b> %{x}<br>' +
                      '<b>PD:</b> %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add current assessment point if available
    current_year = metadata.get('end_year', 2024)
    if current_year in df_ts['year'].values:
        current_pd = df_ts[df_ts['year'] == current_year]['default_probability'].iloc[0]
        fig_prob.add_trace(go.Scatter(
            x=[current_year],
            y=[current_pd],
            mode='markers',
            name='Current (2024)',
            marker=dict(size=15, color='#2c3e50', symbol='star'),
            hovertemplate=f'<b>Year:</b> {current_year}<br>' +
                          f'<b>PD:</b> {current_pd:.2f}%<br>' +
                          '<extra></extra>'
        ))
    
    fig_prob.update_layout(
        title=f"{country}: 5-Year Default Probability (2015-{current_year})",
        xaxis_title="Year",
        yaxis_title="Default Probability (%)",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Time Series Data")
    
    display_df = df_ts[['year', 'default_probability', 'rating', 
                        'debt_gdp', 'deficit_gdp']].copy()
    display_df.columns = ['Year', 'PD (%)', 'Rating', 
                          'Debt/GDP (%)', 'Deficit/GDP (%)']
    
    # Convert Year to datetime for proper display
    display_df['Year'] = pd.to_datetime(display_df['Year'].astype(str), format='%Y').dt.year
    display_df['Year'] = display_df['Year'].astype(str)
    
    # Format columns
    for col in ['PD (%)', 'Debt/GDP (%)', 'Deficit/GDP (%)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary statistics
    if len(df_ts) > 1:
        st.subheader("Change Analysis")
        
        # Define rating order for comparisons
        rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                       'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
                       'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
        
        first_year = df_ts['year'].iloc[0]
        last_year = df_ts['year'].iloc[-1]
        first_pd = df_ts['default_probability'].iloc[0]
        last_pd = df_ts['default_probability'].iloc[-1]
        first_rating = df_ts['rating'].iloc[0]
        last_rating = df_ts['rating'].iloc[-1]
        
        # Comparison from first year to last year
        pd_change_full = last_pd - first_pd
        pd_change_full_display = -pd_change_full
        rating_change_full = rating_order.index(first_rating) - rating_order.index(last_rating) if last_rating in rating_order and first_rating in rating_order else 0
        
        # Comparison from previous year to last year
        prev_year = df_ts['year'].iloc[-2] if len(df_ts) > 1 else last_year
        prev_pd = df_ts['default_probability'].iloc[-2] if len(df_ts) > 1 else last_pd
        prev_rating = df_ts['rating'].iloc[-2] if len(df_ts) > 1 else last_rating
        pd_change_year = last_pd - prev_pd
        pd_change_year_display = -pd_change_year
        rating_change_year = rating_order.index(prev_rating) - rating_order.index(last_rating) if last_rating in rating_order and prev_rating in rating_order else 0
        
        # Helper function to render metric with custom delta color
        def render_metric_with_delta_color(label, value, delta_value, delta_label=""):
            if delta_value == 0:
                delta_color = "#6c757d"  # Neutral gray for zero change
                delta_class = "neutral"
            elif delta_value > 0:
                delta_color = "#28a745"  # Green for positive
                delta_class = "positive"
            else:
                delta_color = "#dc3545"  # Red for negative
                delta_class = "negative"
            
            delta_display = f"{delta_value:+.2f}%" if isinstance(delta_value, float) else f"{delta_value:+d} {delta_label}"
            if delta_value == 0:
                delta_display = "0.00%" if isinstance(delta_value, float) else f"0 {delta_label}"
            
            st.markdown(f"""
            <div style="background-color: #2c3e50; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #34495e; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <div style="font-size: 0.875rem; color: #ecf0f1; margin-bottom: 5px; font-weight: 500;">{label}</div>
                <div style="font-size: 1.875rem; font-weight: 600; margin-bottom: 5px; color: #ffffff;">{value}</div>
                <div style="font-size: 0.875rem; color: {delta_color}; font-weight: 600;">{delta_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Full period comparison (first year to last year)
        col1, col2 = st.columns(2)
        with col1:
            render_metric_with_delta_color(
                f"PD Change ({first_year} to {last_year})",
                f"{pd_change_full:+.2f}%",
                pd_change_full_display
            )
        with col2:
            render_metric_with_delta_color(
                f"Rating Change ({first_year} to {last_year})",
                f"{first_rating} → {last_rating}",
                rating_change_full,
                "notches"
            )
        
        # Previous year comparison
        col3, col4 = st.columns(2)
        with col3:
            render_metric_with_delta_color(
                f"PD Change ({prev_year} to {last_year})",
                f"{pd_change_year:+.2f}%",
                pd_change_year_display
            )
        with col4:
            render_metric_with_delta_color(
                f"Rating Change ({prev_year} to {last_year})",
                f"{prev_rating} → {last_rating}",
                rating_change_year,
                "notches"
            )


def render_about_page():
    """Render the About page."""
    st.header("About Eleutheria")
    
    st.markdown("""
    ## Our Mission
    
    **Eleutheria** challenges the Big 3 rating agencies (S&P, Moody's, Fitch) with real-time, 
    data-driven sovereign credit analysis. While traditional agencies are slow, reactive, and often 
    biased (paid by issuers), we provide transparent, predictive analysis using automated Python 
    models and publicly available data.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### The Problem
        
        - **Too Slow**: Rating agencies react to crises, they don't predict them
        - **Too Biased**: Paid by issuers, creating conflicts of interest
        - **Too Wrong**: Billions lost by investors trusting outdated ratings
        - **Not Predictive**: Downgrades come AFTER the damage is done
        """)
    
    with col2:
        st.markdown("""
        ### Our Solution
        
        - **Real-Time Analysis**: Data updated continuously
        - **Transparent Methodology**: Open about our models and sources
        - **Data-Driven**: Python automation, not opinions
        - **Predictive Focus**: Alert BEFORE crises, not after
        - **European Specialization**: Deep expertise in sovereign risk
        """)
    
    st.divider()
    
    st.markdown("""
    ## How It Works
    All data is sourced from public international organizations (IMF, World Bank, Eurostat) and 
    processed through automated Python scripts to generate default probability assessments and 
    credit ratings.
    """)
    
    st.divider()
    
    st.markdown("""
    ## Why Eleutheria?
    
    **Eleutheria** (ἐλευθερία) means "freedom" in ancient Greek. We believe investors deserve 
    **freedom from**:
    
    - Outdated, reactive ratings
    - Hidden biases and conflicts of interest  
    - Information asymmetry with institutions
    - Slow-moving traditional analysis
    
    And **freedom to**:
    
    - Make informed decisions with real-time data
    - Understand sovereign risk transparently
    - Access professional-grade analysis
    - Stay ahead of market-moving events
    """)
    
    st.divider()
    
    st.markdown("""
    ## Our Positioning

    > *"Independent, real-time sovereign risk analysis providing alternative perspectives to traditional rating agencies."*

    ### Why Alternative Analysis Matters

    Traditional credit rating agencies face structural challenges:
    - **Timing Lag**: Ratings typically reflect historical data and may not capture rapid changes
    - **Issuer-Pays Model**: Academic research (see: Bolton et al., 2012, Journal of Finance) documents potential conflicts in the issuer-pays business model
    - **Pro-Cyclical Bias**: Research shows ratings can amplify market cycles (see: IMF Working Papers)
    - **Historical Accuracy**: Academic studies document instances where rating changes lagged market developments (see: Reinhart & Rogoff, "This Time Is Different")

    **Our Approach:**
    - Higher update frequency
    - Transparent, open methodology
    - Independent funding model (no issuer payments)
    - Complementary perspective to established agencies

    """)

    
    st.divider()
    
    # Newsletter subscription in About page too
    render_newsletter_subscription()
    
    st.divider()
    
    st.info("""
    Our ratings are not comparable to or substitutes for ratings from regulated rating agencies. 
    These are independent analytical assessments for informational purposes only.
    """)


def render_comparison_view(assessments: Dict[str, Dict[str, Any]], all_time_series: Dict[str, Dict[str, Any]]):
    """Render comparison view across all countries."""
    st.header("Country Comparison")
    
    if not assessments:
        st.warning("No assessment data available for comparison.")
        return
    
    # Prepare comparison data
    comparison_data = []
    for country, assessment in assessments.items():
        overall = assessment['overall_assessment']
        comparison_data.append({
            'Country': country,
            'Default Probability (%)': overall['default_probability_5yr'],
            'Rating': overall['rating'],
            'Outlook': overall['rating_outlook']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Multi-country comparison: Line chart for trend visualization
    st.subheader("Multi-Country Default Probability Comparison")
    
    # Prepare data for line chart (if time series data available)
    # For now, show as grouped bar chart for better comparison
    fig_comparison = go.Figure()
    
    # Sort by default probability for better visualization
    df_sorted = df_comparison.sort_values('Default Probability (%)', ascending=False)
    
    fig_comparison.add_trace(go.Bar(
        x=df_sorted['Country'],
        y=df_sorted['Default Probability (%)'],
        marker=dict(
            color=df_sorted['Default Probability (%)'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="PD (%)")
        ),
        text=df_sorted['Default Probability (%)'].apply(lambda x: f"{x:.2f}%"),
        textposition='outside',
        customdata=df_sorted[['Rating', 'Outlook']],
        hovertemplate='<b>%{x}</b><br>' +
                      'Default Probability: %{y:.2f}%<br>' +
                      'Rating: %{customdata[0]}<br>' +
                      'Outlook: %{customdata[1]}<br>' +
                      '<extra></extra>'
    ))
    
    fig_comparison.update_layout(
        title="Default Probability Comparison Across All Countries",
        xaxis_title="Country",
        yaxis_title="Default Probability (%)",
        height=500,
        xaxis={'tickangle': -45},
        hovermode='closest'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Default Probability Over Time comparison
    st.subheader("Default Probability Over Time")
    
    # Country selector for time series comparison
    available_countries_ts = [country for country in list(assessments.keys()) if country in all_time_series]
    
    if available_countries_ts:
        selected_countries = st.multiselect(
            "Select countries to compare over time",
            options=available_countries_ts,
            default=available_countries_ts[:5] if len(available_countries_ts) >= 5 else available_countries_ts
        )
        
        if selected_countries:
            fig_ts_comparison = go.Figure()
            
            for country in selected_countries:
                ts_data = all_time_series[country]
                ts_records = ts_data.get('time_series', [])
                
                if ts_records:
                    df_country_ts = pd.DataFrame(ts_records)
                    
                    fig_ts_comparison.add_trace(go.Scatter(
                        x=df_country_ts['year'],
                        y=df_country_ts['default_probability'],
                        mode='lines+markers',
                        name=country,
                        line=dict(width=2),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{country}</b><br>' +
                                    'Year: %{x}<br>' +
                                    'PD: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    ))
            
            fig_ts_comparison.update_layout(
                title="Default Probability Evolution - Multi-Country Comparison",
                xaxis_title="Year",
                yaxis_title="Default Probability (%)",
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_ts_comparison, use_container_width=True)
        else:
            st.info("Please select at least one country to compare.")
    else:
        st.warning("No time series data available for comparison.")


def render_header():
    """Render branded header with logo and banner."""
    # Display banner
    banner_path = IMAGES_DIR / "banner_1.webp"
    if banner_path.exists():
        st.image(str(banner_path))
    
    # Title with branding and gradient
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #01757C 0%, #00D4C9 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; color: white; font-weight: 300; font-size: 2.5em; font-family: 'Cambria', serif;">Eleutheria - Sovereign Risk Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()


def render_newsletter_subscription():
    """Render newsletter subscription link."""
    st.markdown(
        """
        <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #01757C 0%, #00D4C9 100%); border-radius: 10px; margin: 30px 0;">
            <h3 style="color: white; margin-bottom: 15px;">Stay Ahead with Eleutheria</h3>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1em; margin-bottom: 20px;">
                Get real-time sovereign risk analysis before the Big 3 react. Subscribe to our newsletter on Substack.
            </p>
            <a href="https://substack.com/@eleutheriarc?utm_campaign=profile&utm_medium=profile-page" 
               target="_blank" 
               style="display: inline-block; padding: 12px 30px; background-color: white; color: #01757C; text-decoration: none; border-radius: 25px; font-weight: bold; font-size: 1.1em; transition: transform 0.2s;">
                Subscribe to Newsletter →
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_legal_notices_content():
    """Render legal notices content."""
    st.markdown("""
        ### Disclaimer
        
        **The information provided by Eleutheria ("we," "us," or "our") on this dashboard is for general informational purposes only.**
        
        - All information is provided in good faith, however we make no representation or warranty of any kind, express or implied, regarding the accuracy, adequacy, validity, reliability, availability, or completeness of any information on this dashboard.
        
        - **This is not investment advice.** The sovereign risk assessments, default probabilities, and credit ratings displayed are analytical outputs based on publicly available data and mathematical models. They should not be considered as recommendations to buy, sell, or hold any securities or financial instruments.
        
        - **You should not make any investment decisions solely based on the information contained in this dashboard.** Always consult with qualified financial advisors and conduct your own due diligence before making investment decisions.
        
        ### Limitation of Liability
        
        - Eleutheria and its operators will not be liable for any loss or damage of any nature incurred as a result of the use of this dashboard or reliance on any information provided herein.
        
        - We do not guarantee the accuracy, completeness, or timeliness of the data, and assume no responsibility for any errors or omissions in the content.
        
        ### Data Sources
        
        - Data is sourced from publicly available international organizations including IMF, World Bank, and Eurostat, among others. Data may contain errors, omissions, or may be outdated.
        
        - Ratings and assessments are calculated using proprietary methodologies and may differ significantly from those issued by S&P, Moody's, Fitch, or other rating agencies.
        
        ### Intellectual Property
        
        - All content, methodologies, and visualizations on this dashboard are the intellectual property of Eleutheria unless otherwise stated.
        
        - Unauthorized reproduction or distribution of this content is prohibited.
        
        ### Contact
        
        For questions regarding these legal notices, please contact us through the appropriate channels.
        
        **Last Updated:** """ + datetime.now().strftime('%Y-%m-%d') + """
        """)


def render_legal_notices():
    """Render comprehensive legal notices as full page."""
    st.header("Legal Notices & Disclaimers")
    
    # Make this VISIBLE by default on first visit
    if 'legal_acknowledged' not in st.session_state:
        st.session_state.legal_acknowledged = False
    
    if not st.session_state.legal_acknowledged:
        with st.container():
            st.error("""
            ### ⚠️ MANDATORY LEGAL DISCLAIMER - READ BEFORE PROCEEDING
            
            By using this dashboard, you acknowledge and agree to the following terms:
            """)
            
            st.markdown("""
            #### 1. NOT INVESTMENT ADVICE
            
            This dashboard provides **educational and informational content only**. Nothing on this platform constitutes:
            - Investment advice or recommendations
            - An offer to buy or sell securities
            - Professional financial, legal, or tax advice
            - A substitute for consultation with licensed professionals
            
            **You are solely responsible for your investment decisions.**
            
            #### 2. NOT A CREDIT RATING AGENCY
            
            Eleutheria is **NOT**:
            - A registered credit rating agency with ESMA (European Securities and Markets Authority)
            - A Nationally Recognized Statistical Rating Organization (NRSRO) with the U.S. SEC
            - Affiliated with, endorsed by, or comparable to S&P Global Ratings, Moody's Investors Service, Fitch Ratings, or any other recognized rating agency
            
            Our "ratings" are **analytical scores from a proprietary model** - they are NOT credit ratings as defined by EU Regulation (EC) No 1060/2009 or U.S. securities laws.
            
            #### 3. NO GUARANTEES OR WARRANTIES
            
            We make **NO representations or warranties** regarding:
            - Accuracy, completeness, or timeliness of data or analysis
            - Fitness for any particular purpose
            - Results from using this information
            - Absence of errors, bugs, or omissions
            
            **Data may be outdated, incomplete, or contain errors.** Public data sources (IMF, World Bank, Eurostat) may themselves contain inaccuracies.
            
            #### 4. LIMITATION OF LIABILITY
            
            **TO THE MAXIMUM EXTENT PERMITTED BY LAW:**
            
            Eleutheria, its operators, contributors, and affiliates shall NOT be liable for:
            - Any investment losses or financial damages
            - Direct, indirect, incidental, consequential, or punitive damages
            - Loss of profits, data, or business opportunities
            - Any damages arising from use of or inability to use this dashboard
            
            **Even if advised of the possibility of such damages.**
            
            #### 5. INTELLECTUAL PROPERTY
            
            - All content, methodologies, code, and visualizations are © Eleutheria (all rights reserved) unless otherwise stated
            - Our methodology is **independently developed** and does not replicate proprietary systems of S&P, Moody's, Fitch, or other agencies
            - Unauthorized reproduction, distribution, or commercial use is prohibited
            - Academic/research citations require prior written permission
            
            #### 6. THIRD-PARTY DATA
            
            Data sourced from IMF, World Bank, Eurostat, and other organizations:
            - Remains property of respective organizations
            - Subject to their own terms of use and licenses
            - May be subject to revisions or corrections
            - We do not guarantee accuracy of third-party data
            
            #### 7. REGULATORY STATUS
            
            - **France (AMF)**: Not registered as investment advisor (Conseiller en Investissements Financiers)
            - **EU (ESMA)**: Not registered as Credit Rating Agency under CRA Regulation
            - **United States (SEC/FINRA)**: Not registered as investment advisor or broker-dealer
            
            If your jurisdiction requires specific registrations for the services you seek, **DO NOT USE THIS DASHBOARD** for investment purposes.
            
            #### 8. BETA/PROOF-OF-CONCEPT STATUS
            
            This dashboard is a **proof of concept** demonstrating analytical capabilities. It may contain:
            - Software bugs or calculation errors
            - Incomplete or experimental features
            - Methodological limitations
            - Unvalidated models or assumptions
            
            **Use at your own risk.**
            
            #### 9. NO FIDUCIARY DUTY
            
            We owe you **NO fiduciary duty**. We do not act as your advisor, agent, or fiduciary in any capacity.
            
            #### 10. CHANGES TO TERMS
            
            We reserve the right to modify these terms at any time without notice. Continued use constitutes acceptance of updated terms.
            
            #### 11. GOVERNING LAW & JURISDICTION
            
            These terms are governed by **French law**. Any disputes shall be subject to the exclusive jurisdiction of French courts.
            
            #### 12. SEVERABILITY
            
            If any provision is found unenforceable, remaining provisions remain in full effect.
            
            **Last Updated:** """ + datetime.now().strftime('%B %d, %Y') + """
            
            ---
            """)
            
            accept = st.checkbox("I have read, understood, and agree to these terms and disclaimers")
            
            if st.button("Proceed to Dashboard", disabled=not accept):
                st.session_state.legal_acknowledged = True
                st.rerun()
            
            if not accept:
                st.stop()  # Prevent access until acknowledged
    
    # Also keep expandable version at bottom for reference
    with st.expander("📋 Full Legal Notices & Disclaimers", expanded=False):
        st.markdown("[Same content as above]")

def render_regulatory_disclosures():
    """Render required regulatory disclosures."""
    st.markdown("""
    ## Regulatory Disclosures
    
    ### Entity Information
    - **Operating Entity**: [Your legal name or "Individual/Sole Proprietorship"]
    - **Registration**: Not registered as investment advisor or credit rating agency
    - **Location**: France
    - **Business Model**: Educational content and analytical tools
    
    ### Non-Registration Disclosure
    
    **Eleutheria is NOT:**
    - Registered with the Autorité des Marchés Financiers (AMF) as a financial investment advisor
    - Registered with ESMA as a Credit Rating Agency under Regulation (EC) No 1060/2009
    - Registered with the U.S. SEC or any other financial regulatory body
    - Authorized to provide investment advice under French law (Code monétaire et financier)
    
    **What This Means:**
    - We do not provide personalized investment recommendations
    - We do not manage client assets
    - We do not offer credit ratings for regulatory purposes
    - Our content is for informational/educational purposes only
    
    ### Conflicts of Interest
    
    **Current Conflicts:** [Be honest - examples:]
    - Operator holds positions in European sovereign bonds [if applicable]
    - Operator may benefit from newsletter subscriptions/future paid services
    - Dashboard designed to attract attention to operator's content
    
    We commit to disclosing any material conflicts that arise.
    
    ### Data Processing (GDPR Compliance)
    
    **If you collect emails for newsletter:**
    - **Data Controller**: [Your name/entity]
    - **Purpose**: Newsletter delivery only
    - **Legal Basis**: Consent (GDPR Article 6(1)(a))
    - **Data Retention**: Until you unsubscribe
    - **Your Rights**: Access, rectification, erasure, portability, objection
    - **Contact**: [Email address]
    
    [If using Substack, note they are separate data controller for their platform]
    
    ### Cookies & Tracking
    
    This dashboard may use:
    - Streamlit's default session cookies (functional, required)
    - [Any analytics you add - must disclose]
    
    No third-party advertising cookies are used.
    """)

def render_terms_of_service():
    """Comprehensive Terms of Service."""
    st.markdown("""
    # Terms of Service
    
    **Effective Date:** [Date]
    
    ## 1. Acceptance of Terms
    
    By accessing this dashboard, you agree to be bound by these Terms of Service and all applicable laws.
    If you do not agree, you must immediately cease use.
    
    ## 2. License to Use
    
    We grant you a **limited, non-exclusive, non-transferable, revocable license** to access this dashboard for:
    - Personal, non-commercial use
    - Educational purposes
    - Research purposes
    
    **Prohibited Uses:**
    - Commercial redistribution of our analysis or data
    - Automated scraping or data harvesting
    - Reverse engineering our methodologies
    - Reselling or sublicensing access
    - Use for creating competing products
    - Misrepresenting source or affiliation
    
    ## 3. User Obligations
    
    You agree to:
    - Provide accurate information if registering
    - Maintain confidentiality of any account credentials
    - Not share access credentials
    - Comply with applicable laws
    - Not use dashboard for illegal purposes
    
    ## 4. Service Availability
    
    We provide this dashboard "AS IS" without guarantees of:
    - Continuous availability
    - Error-free operation
    - Specific uptime percentages
    - Data accuracy or completeness
    
    We may suspend or terminate service at any time without notice.
    
    ## 5. User-Generated Content
    
    [If you add comments or forums later]
    
    ## 6. Indemnification
    
    You agree to indemnify and hold harmless Eleutheria and its operators from any claims, losses, damages, or expenses (including legal fees) arising from:
    - Your use of the dashboard
    - Your violation of these terms
    - Your violation of any third-party rights
    - Investment decisions you make
    
    ## 7. Termination
    
    We may terminate your access immediately for:
    - Violation of these terms
    - Abusive behavior
    - Any reason or no reason
    
    ## 8. Dispute Resolution
    
    **Governing Law:** French law
    
    **Jurisdiction:** Exclusive jurisdiction of French courts
    
    **[Optional - consider adding]:**
    - Mandatory arbitration clause
    - Class action waiver
    - [These can be controversial but provide protection]
    
    ## 9. Changes to Terms
    
    We reserve the right to modify these terms at any time. Changes effective immediately upon posting.
    
    ## 10. Entire Agreement
    
    These terms constitute the entire agreement between you and Eleutheria regarding use of this dashboard.
    
    ## 11. Contact
    
    Questions: [Contact email]
    
    **Last Updated:** [Date]
    """)

def render_privacy_policy():
    """GDPR-compliant Privacy Policy."""
    st.markdown("""
    # Privacy Policy
    
    **Effective Date:** [Date]
    
    ## Introduction
    
    Eleutheria ("we," "us," "our") respects your privacy. This policy explains how we collect, use, and protect your personal data in compliance with the EU General Data Protection Regulation (GDPR).
    
    ## 1. Data Controller
    
    **Identity:** [Your legal name or business entity]
    **Contact:** [Email address]
    **Location:** France
    
    ## 2. Data We Collect
    
    ### Automatically Collected (via Streamlit):
    - IP address
    - Browser type
    - Session information
    - Access times
    - Pages viewed
    
    **Legal Basis:** Legitimate interest (operating the service)
    
    ### Voluntarily Provided:
    - Email address (if you subscribe to newsletter)
    - Name (if provided)
    - Any information you submit via contact forms
    
    **Legal Basis:** Consent (GDPR Article 6(1)(a))
    
    ## 3. How We Use Your Data
    
    - **Service Delivery:** To operate the dashboard
    - **Communications:** To send newsletters (if subscribed)
    - **Analytics:** To understand usage patterns and improve service
    - **Legal Compliance:** To comply with applicable laws
    
    ## 4. Data Sharing
    
    We do NOT sell your personal data.
    
    **We may share data with:**
    - **Substack:** If you subscribe to newsletter (they are separate data controller)
    - **Hosting Providers:** [e.g., Streamlit Cloud, AWS] - as data processors
    - **Legal Authorities:** If required by law
    
    ## 5. International Transfers
    
    Data may be transferred outside the EU to:
    - **United States** (Streamlit Cloud / Substack)
    
    **Safeguards:** We rely on:
    - Standard Contractual Clauses
    - Adequacy decisions
    - Service provider's compliance frameworks
    
    ## 6. Data Retention
    
    - **Analytics data:** 90 days
    - **Newsletter subscriptions:** Until you unsubscribe
    - **Contact form inquiries:** 2 years
    
    ## 7. Your Rights (GDPR)
    
    You have the right to:
    - **Access:** Request copy of your personal data
    - **Rectification:** Correct inaccurate data
    - **Erasure:** Request deletion ("right to be forgotten")
    - **Portability:** Receive your data in structured format
    - **Object:** Object to processing
    - **Restrict:** Request restricted processing
    - **Withdraw Consent:** Unsubscribe at any time
    
    **To exercise rights:** Contact [email]
    
    **Supervisory Authority:** You may lodge complaint with CNIL (France) or your local data protection authority.
    
    ## 8. Cookies
    
    We use:
    - **Essential cookies:** Required for dashboard functionality (no consent required under GDPR)
    - **Analytics cookies:** [If you add Google Analytics, etc. - requires consent]
    
    [If using non-essential cookies, add cookie banner]
    
    ## 9. Security
    
    We implement reasonable security measures including:
    - HTTPS encryption
    - Secure hosting
    - Access controls
    
    However, no system is 100% secure. Use at your own risk.
    
    ## 10. Children
    
    This service is not directed at children under 16. We do not knowingly collect data from children.
    
    ## 11. Changes to Privacy Policy
    
    We may update this policy. We'll notify you of material changes via dashboard banner or email (if subscribed).
    
    ## 12. Contact
    
    Privacy questions: [Email]
    
    **Last Updated:** [Date]
    """)


def add_custom_css():
    """Add custom CSS styling for Eleutheria branding."""
    # Get logo path for favicon and encode as base64
    logo_path = IMAGES_DIR / "logo_1.webp"
    favicon_script = ""
    if logo_path.exists():
        try:
            with open(logo_path, 'rb') as f:
                logo_data = f.read()
                logo_base64 = base64.b64encode(logo_data).decode('utf-8')
                # Use JavaScript to set favicon dynamically
                favicon_script = f"""
                <script>
                    (function() {{
                        var link = document.querySelector("link[rel*='icon']") || document.createElement('link');
                        link.type = 'image/webp';
                        link.rel = 'shortcut icon';
                        link.href = 'data:image/webp;base64,{logo_base64}';
                        document.getElementsByTagName('head')[0].appendChild(link);
                    }})();
                </script>
                """
        except Exception:
            # If encoding fails, just skip favicon
            pass
    
    st.markdown(f"""
    {favicon_script}
    <style>
        /* Main title styling */
        h1 {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 700;
        }}
        
        /* Metric cards styling */
        [data-testid="stMetricValue"] {{
            font-size: 1.8rem;
        }}
        
        /* Footer styling */
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }}
        
        /* Ensure sidebar elements are visible */
        section[data-testid="stSidebar"] > div {{
            background-color: transparent;
        }}
        
        /* Radio button styling for better visibility */
        div[data-testid="stRadio"] > div {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 5px;
        }}
        
        /* Logo sizing */
        [data-testid="stSidebar"] img {{
            max-width: 150px !important;
            height: auto !important;
            margin: 0 auto;
            display: block;
        }}
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Add custom CSS
    add_custom_css()
    
    # Render branded header
    render_header()
    
    # Load all assessments
    assessments = load_all_assessments()
    
    if not assessments:
        st.error("No assessment files found. Please run risk_calculator_2.py first.")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        # Logo (smaller size)
        logo_path = IMAGES_DIR / "logo_1.webp"
        if logo_path.exists():
            st.image(str(logo_path))
        
        st.markdown("---")
        st.header("Navigation")
        
        view_mode = st.radio(
            "Select View",
            ["Country Analysis", "Country Comparison", "About"],
            index=0,
            label_visibility="visible"
        )
        
        if view_mode == "Country Analysis":
            selected_country = st.selectbox(
                "Select Country",
                options=list(assessments.keys()),
                index=0
            )
            
            assessment = assessments[selected_country]
            
        elif view_mode == "Country Comparison":
            selected_country = None
            assessment = None
            
        else:  # About page
            selected_country = None
            assessment = None
    
    # Load time series data
    all_time_series = load_all_time_series()
    time_series_data = all_time_series.get(selected_country) if selected_country else None
    
    # Main content area
    if view_mode == "Country Analysis" and assessment:
        # Overview section (no tabs)
        render_overview_tab(assessment)
        
        # Time series at the bottom
        st.divider()
        render_time_series_tab(time_series_data, selected_country, assessment)
        
        # Newsletter subscription link
        render_newsletter_subscription()
    
    elif view_mode == "Country Comparison":
        render_comparison_view(assessments, all_time_series)
        
        # Newsletter subscription link
        render_newsletter_subscription()
    
    elif view_mode == "About":
        render_about_page()
    
    # Footer with legal notices (only for main pages)
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; color: #666;">
            <p><strong>Eleutheria</strong> - Challenging the Big 3 with Real-Time Data</p>
            <p style="font-size: 0.9em;">
                Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | 
                Data source: IMF, World Bank, Eurostat
            </p>
            <p style="font-size: 0.8em; color: #888;">
                © """ + str(datetime.now().year) + """ Eleutheria. All rights reserved.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Legal notices at the bottom of each page (as expander with all legal content)
    with st.expander("Legal Notices & Disclaimers", expanded=False):
        render_legal_notices_content()
        
        st.divider()
        
        # Terms of Service section
        st.markdown("## Terms of Service")
        
        st.markdown("""
**Effective Date:** [Date]

### 1. Acceptance of Terms

By accessing this dashboard, you agree to be bound by these Terms of Service and all applicable laws.
If you do not agree, you must immediately cease use.

### 2. License to Use

We grant you a **limited, non-exclusive, non-transferable, revocable license** to access this dashboard for:
- Personal, non-commercial use
- Educational purposes
- Research purposes

**Prohibited Uses:**
- Commercial redistribution of our analysis or data
- Automated scraping or data harvesting
- Reverse engineering our methodologies
- Reselling or sublicensing access
- Use for creating competing products
- Misrepresenting source or affiliation

### 3. User Obligations

You agree to:
- Provide accurate information if registering
- Maintain confidentiality of any account credentials
- Not share access credentials
- Comply with applicable laws
- Not use dashboard for illegal purposes

### 4. Service Availability

We provide this dashboard "AS IS" without guarantees of:
- Continuous availability
- Error-free operation
- Specific uptime percentages
- Data accuracy or completeness

We may suspend or terminate service at any time without notice.

### 5. Indemnification

You agree to indemnify and hold harmless Eleutheria and its operators from any claims, losses, damages, or expenses (including legal fees) arising from:
- Your use of the dashboard
- Your violation of these terms
- Your violation of any third-party rights
- Investment decisions you make

### 6. Termination

We may terminate your access immediately for:
- Violation of these terms
- Abusive behavior
- Any reason or no reason

### 7. Dispute Resolution

**Governing Law:** French law

**Jurisdiction:** Exclusive jurisdiction of French courts

### 8. Changes to Terms

We reserve the right to modify these terms at any time. Changes effective immediately upon posting.

### 9. Contact

Questions: [Contact email]

**Last Updated:** """ + datetime.now().strftime('%Y-%m-%d') + """
""")
        
        st.divider()
        
        # Privacy Policy section
        st.markdown("## Privacy Policy")
        
        st.markdown("""
**Effective Date:** [Date]

### Introduction

Eleutheria ("we," "us," "our") respects your privacy. This policy explains how we collect, use, and protect your personal data in compliance with the EU General Data Protection Regulation (GDPR).

### 1. Data Controller

**Identity:** [Your legal name or business entity]
**Contact:** [Email address]
**Location:** France

### 2. Data We Collect

- **Usage data:** Analytics on dashboard access (if enabled)
- **Newsletter subscriptions:** Email addresses (if subscribed via Substack)
- **Technical data:** IP addresses, browser type (minimal, functional)

### 3. How We Use Your Data

- To provide and improve the dashboard
- To send newsletter updates (with consent)
- For analytics and performance monitoring

### 4. Legal Basis (GDPR)

- **Consent:** For newsletter subscriptions
- **Legitimate interest:** For basic analytics and service improvement
- **Contract:** For providing the dashboard service

### 5. Data Sharing

We do not sell your data. We may share with:
- Substack (for newsletter delivery) - separate data controller
- Hosting providers (technical infrastructure)

### 6. Your Rights (GDPR)

You have the right to:
- **Access:** Request copy of your personal data
- **Rectification:** Correct inaccurate data
- **Erasure:** Request deletion ("right to be forgotten")
- **Portability:** Receive your data in structured format
- **Object:** Object to processing
- **Withdraw Consent:** Unsubscribe at any time

**To exercise rights:** Contact [email]

### 7. Data Retention

- **Analytics data:** 90 days
- **Newsletter subscriptions:** Until you unsubscribe
- **Contact form inquiries:** 2 years

### 8. Security

We implement reasonable security measures including HTTPS encryption and secure hosting. However, no system is 100% secure.

### 9. Contact

Privacy questions: [Email]

**Last Updated:** """ + datetime.now().strftime('%Y-%m-%d') + """
""")


if __name__ == "__main__":
    main()

