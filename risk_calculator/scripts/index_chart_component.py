"""
Reusable Index Chart Component
Can be imported into other Streamlit dashboards

Usage:
    from index_chart_component import render_index_chart_tab, create_index_chart, load_index_data
    
    # Option 1: Use the complete tab component
    render_index_chart_tab(output_dir="output")
    
    # Option 2: Just get the chart figure
    perf_dict = load_index_data("output")
    fig = create_index_chart(perf_dict)
    st.plotly_chart(fig, use_container_width=True)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


@st.cache_data(ttl=3600)
def load_index_data(output_dir="output"):
    """
    Load index performance data from output directory.
    
    Parameters:
    -----------
    output_dir : str or Path
        Path to the output directory containing CSV files
    
    Returns:
    --------
    dict
        Dictionary with keys like 'Equal Weighted', 'Market Cap Weighted'
        and values as DataFrames with 'date' and 'cumulative_index' columns
    """
    output_path = Path(output_dir)
    data_dict = {}
    
    try:
        # Load Equal Weighted
        if (output_path / "performance_equal_weighted.csv").exists():
            perf_eq = pd.read_csv(output_path / "performance_equal_weighted.csv")
            perf_eq['date'] = pd.to_datetime(perf_eq['date'])
            perf_eq = perf_eq.sort_values('date').reset_index(drop=True)
            data_dict['Equal Weighted'] = perf_eq
            
        # Load Market Cap Weighted
        if (output_path / "performance_market_cap_weighted.csv").exists():
            perf_mc = pd.read_csv(output_path / "performance_market_cap_weighted.csv")
            perf_mc['date'] = pd.to_datetime(perf_mc['date'])
            perf_mc = perf_mc.sort_values('date').reset_index(drop=True)
            data_dict['Market Cap Weighted'] = perf_mc
            
        return data_dict
    except Exception as e:
        st.error(f"Error loading index data: {e}")
        return {}


def create_index_chart(perf_dict, show_base_line=True, height=500):
    """
    Create the index performance chart.
    
    Parameters:
    -----------
    perf_dict : dict
        Dictionary with keys like 'Equal Weighted', 'Market Cap Weighted'
        and values as DataFrames with 'date' and 'cumulative_index' columns
    show_base_line : bool
        Whether to show the base level (100) line
    height : int
        Chart height in pixels
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The configured Plotly figure
    """
    # Brand colors
    COLOR_EQUAL = "#0E2841"  # Primary Dark Blue
    COLOR_MCAP = "#035159"   # Teal/Green
    
    fig_main = go.Figure()
    
    # Add Equal Weighted line if available
    if 'Equal Weighted' in perf_dict:
        perf_eq = perf_dict['Equal Weighted']
        fig_main.add_trace(go.Scatter(
            x=perf_eq['date'],
            y=perf_eq['cumulative_index'],
            mode='lines',
            name='Equal Weighted',
            line=dict(color=COLOR_EQUAL, width=3),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Index Level: %{y:.2f}<extra></extra>'
        ))
    
    # Add Market Cap Weighted line if available
    if 'Market Cap Weighted' in perf_dict:
        perf_mc = perf_dict['Market Cap Weighted']
        fig_main.add_trace(go.Scatter(
            x=perf_mc['date'],
            y=perf_mc['cumulative_index'],
            mode='lines',
            name='Market Cap Weighted',
            line=dict(color=COLOR_MCAP, width=3),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Index Level: %{y:.2f}<extra></extra>'
        ))
    
    # Set Y-axis range
    if len(fig_main.data) > 0:
        all_values = []
        for trace in fig_main.data:
            all_values.extend(trace.y)
        
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range_start = 50
            y_range_end = max(y_max * 1.05, 150)
            fig_main.update_yaxes(range=[y_range_start, y_range_end])
    
    # Add base level line
    if show_base_line:
        fig_main.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="#E0E0E0", 
            opacity=0.5, 
            annotation_text="Base Level (100)", 
            annotation_position="right",
            annotation_font=dict(color='black')
        )
    
    # Update layout
    fig_main.update_layout(
        title="",  # Explicitly set empty title to avoid "undefined"
        xaxis_title="Date",
        yaxis_title="Index Level (Base = 100)",
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1, 
            font=dict(family="Arial", color='black', size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color='black'),
        xaxis=dict(
            gridcolor='#E0E0E0', 
            title_font=dict(family="Arial", color='black', size=11), 
            tickfont=dict(family="Arial", color='black', size=10)
        ),
        yaxis=dict(
            gridcolor='#E0E0E0', 
            title_font=dict(family="Arial", color='black', size=11), 
            tickfont=dict(family="Arial", color='black', size=10)
        )
    )
    
    return fig_main


def render_index_chart_tab(output_dir="output", show_subheader=True, show_info=True):
    """
    Render the complete index chart tab component.
    Can be called from another dashboard.
    
    Parameters:
    -----------
    output_dir : str or Path
        Path to the output directory containing CSV files
    show_subheader : bool
        Whether to show the "DD Bond Index Performance" subheader
    show_info : bool
        Whether to show data range info at the top
    
    Returns:
    --------
    bool
        True if chart was rendered successfully, False otherwise
    """
    output_path = Path(output_dir)
    
    # Load data
    perf_dict = load_index_data(str(output_path))
    
    if not perf_dict:
        st.warning("⚠️ No performance data found.")
        st.info(f"Looking for data in: {output_path.absolute()}")
        st.code("Make sure you've run: python run_dual_index.py", language="bash")
        return False
    
    # Show data range info if requested
    if show_info and perf_dict:
        # Use the first available dataset for info
        first_perf = list(perf_dict.values())[0]
        data_start = first_perf['date'].min().date()
        data_end = first_perf['date'].max().date()
        data_years = (first_perf['date'].max() - first_perf['date'].min()).days / 365.25
        
        # Determine frequency
        if len(first_perf) > 1:
            days_between = (first_perf['date'].iloc[1] - first_perf['date'].iloc[0]).days
            freq_label = "weekly" if days_between <= 7 else "monthly"
        else:
            freq_label = "period"
        
        weighting_method = list(perf_dict.keys())[0]
        st.info(f"📅 **Data Range:** {data_start} to {data_end} ({data_years:.1f} years, {len(first_perf)} {freq_label} observations) | **Method:** {weighting_method}")
    
    # Show subheader if requested
    if show_subheader:
        st.subheader("DD Bond Index Performance")
    
    # Create and display chart
    fig = create_index_chart(perf_dict)
    st.plotly_chart(fig, use_container_width=True)
    
    return True

