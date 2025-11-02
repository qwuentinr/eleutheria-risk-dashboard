import io
import pandas as pd
import streamlit as st
from scipy.stats import rankdata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import os

st.set_page_config(page_title="📊 GPR Analytics Platform", layout="wide")

# --------------------------
# Utilities
# --------------------------
@st.cache_data(show_spinner=False)
def load_excel(src, sheet_name=0):
    """Load .xls/.xlsx into a clean monthly dataframe indexed by date."""
    try:
        df = pd.read_excel(src, sheet_name=sheet_name, engine="xlrd")
    except Exception:
        df = pd.read_excel(src, sheet_name=sheet_name)

    # Handle the date column - looking for 'month' which appears to be the first column
    candidates = [c for c in df.columns if str(c).strip().lower() in ("date","month","time","period")]
    date_col = candidates[0] if candidates else df.columns[0]

    # Convert date column - handle the 01/01/1900 format
    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    
    # If that fails, try other formats
    if df[date_col].isna().all():
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Drop rows with invalid dates and set index
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Convert numeric columns, being more careful about data types
    for c in df.columns:
        if c not in ['var_name', 'var_label']:  # Skip text columns
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop all-empty columns and text columns that aren't needed
    df = df.dropna(axis=1, how="all")
    text_cols = ['var_name', 'var_label']
    df = df.drop(columns=[col for col in text_cols if col in df.columns])
    
    return df


def extract_country_code(col_name):
    """Extract 3-letter country code from column name like GPRC_USA or GPRHC_FRA"""
    if '_' in col_name and len(col_name.split('_')) >= 2:
        return col_name.split('_')[-1]
    return None


def get_country_name(country_code):
    """Convert 3-letter country code to full name"""
    country_mapping = {
        'USA': '🇺🇸 United States',
        'DEU': '🇩🇪 Germany', 
        'FRA': '🇫🇷 France',
        'GBR': '🇬🇧 United Kingdom',
        'JPN': '🇯🇵 Japan',
        'CHN': '🇨🇳 China',
        'BRA': '🇧🇷 Brazil',
        'IND': '🇮🇳 India',
        'RUS': '🇷🇺 Russia',
        'CAN': '🇨🇦 Canada',
        'AUS': '🇦🇺 Australia',
        'ITA': '🇮🇹 Italy',
        'ESP': '🇪🇸 Spain',
        'NLD': '🇳🇱 Netherlands',
        'CHE': '🇨🇭 Switzerland',
        'SWE': '🇸🇪 Sweden',
        'NOR': '🇳🇴 Norway',
        'DNK': '🇩🇰 Denmark',
        'FIN': '🇫🇮 Finland',
        'BEL': '🇧🇪 Belgium',
        'POL': '🇵🇱 Poland',
        'HUN': '🇭🇺 Hungary',
        'PRT': '🇵🇹 Portugal',
        'KOR': '🇰🇷 South Korea',
        'MEX': '🇲🇽 Mexico',
        'ARG': '🇦🇷 Argentina',
        'CHL': '🇨🇱 Chile',
        'PER': '🇵🇪 Peru',
        'COL': '🇨🇴 Colombia',
        'VEN': '🇻🇪 Venezuela',
        'TUR': '🇹🇷 Turkey',
        'ISR': '🇮🇱 Israel',
        'EGY': '🇪🇬 Egypt',
        'SAU': '🇸🇦 Saudi Arabia',
        'THA': '🇹🇭 Thailand',
        'MYS': '🇲🇾 Malaysia',
        'IDN': '🇮🇩 Indonesia',
        'PHL': '🇵🇭 Philippines',
        'VNM': '🇻🇳 Vietnam',
        'HKG': '🇭🇰 Hong Kong',
        'TWN': '🇹🇼 Taiwan',
        'UKR': '🇺🇦 Ukraine',
        'TUN': '🇹🇳 Tunisia',
        'ZAF': '🇿🇦 South Africa',
    }
    return country_mapping.get(country_code, f"📍 {country_code}")


def pct_rank(series: pd.Series, ref: pd.Series) -> pd.Series:
    """Percentile rank of series values relative to the reference distribution ref."""
    # Clean the inputs
    series_clean = series.dropna()
    ref_clean = ref.dropna()
    
    if len(ref_clean) < 3 or len(series_clean) == 0:
        return pd.Series(index=series.index, dtype=float)
    
    ref_values = ref_clean.values
    out = []
    
    for v in series.values:
        if pd.isna(v):
            out.append(float('nan'))
        else:
            r = rankdata(list(ref_values) + [v], method="average")[-1]
            out.append(100.0 * (r - 1) / len(ref_values))
    
    return pd.Series(out, index=series.index)


def bucket_from_score(x, edges=(25, 50, 75, 90)):
    if pd.isna(x):
        return "N/A"
    a, b, c, d = edges
    if x <= a: return "Very Low"
    if x <= b: return "Low" 
    if x <= c: return "Moderate"
    if x <= d: return "High"
    return "Extreme"


def to_month_start(idx):
    return pd.to_datetime(idx).to_period("M").to_timestamp()


def compute_credit_score(gpr_series, gpr_ref, chg_horizon, chg_lookback, w_level, ewma_span, overlay_series=None, overlay_params=None):
    """Compute credit score for a given GPR series"""
    # Level percentile
    pct_level = pct_rank(gpr_series, gpr_ref)
    
    # Change over horizon
    chg = gpr_series.diff(chg_horizon)
    
    # Change percentile (rolling window)
    pct_chg = pd.Series(index=chg.index, dtype=float)
    lookback_periods = chg_lookback * 12
    
    for i in range(len(chg)):
        if pd.isna(chg.iloc[i]):
            continue
        start_idx = max(0, i - lookback_periods)
        window_data = chg.iloc[start_idx:i+1]
        if len(window_data.dropna()) >= 6:
            pct_chg.iloc[i] = pct_rank(pd.Series([chg.iloc[i]]), window_data).iloc[0]
    
    # Raw score
    score_raw = w_level * pct_level + (1 - w_level) * pct_chg
    score_raw = pd.to_numeric(score_raw, errors='coerce').dropna()
    
    # Event overlay
    overlay = pd.Series(0.0, index=score_raw.index)
    if overlay_series is not None and overlay_params is not None:
        overlay_thresh, overlay_boost, overlay_decay, ref_start = overlay_params
        acts = overlay_series.reindex(score_raw.index)
        acts_clean = pd.to_numeric(acts, errors='coerce')
        
        if not acts_clean.empty and overlay_thresh > 0:
            acts_ref = acts_clean[acts_clean.index >= ref_start]
            if len(acts_ref) > 0:
                threshold = acts_ref.quantile(overlay_thresh / 100.0)
                events = acts_clean > threshold
                
                for idx in events[events].index:
                    end_date = idx + pd.DateOffset(months=overlay_decay)
                    mask = (overlay.index >= idx) & (overlay.index <= end_date)
                    overlay[mask] = overlay_boost
    
    # Final score with EWMA smoothing
    if len(score_raw) > 0:
        try:
            score = score_raw.ewm(span=ewma_span, adjust=False).mean()
            score = score + overlay.reindex(score.index, fill_value=0)
            score = score.clip(0, 100)
        except Exception as e:
            # Fallback without EWMA
            score = score_raw + overlay.reindex(score_raw.index, fill_value=0)
            score = score.clip(0, 100)
    else:
        score = pd.Series(dtype=float)
    
    return {
        'level_pct': pct_level,
        'change': chg,
        'change_pct': pct_chg,
        'score_raw': score_raw,
        'overlay': overlay,
        'score': score
    }


# --------------------------
# Sidebar – Data and Parameters
# --------------------------
st.sidebar.markdown("## 📁 Data Source")

uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xls","xlsx"])
default_path = "data_gpr_export.xls"

# Check if default file exists
file_exists = os.path.exists(default_path)
use_default = st.sidebar.checkbox(
    f"📂 Use local file: {default_path}", 
    value=True if uploaded is None and file_exists else False,
    disabled=not file_exists
)

if not file_exists and uploaded is None:
    st.sidebar.warning("⚠️ Local file not found. Please upload a file.")

# Load data
df_raw = None
try:
    if uploaded is not None and not use_default:
        df_raw = load_excel(uploaded)
        st.sidebar.success("✅ File uploaded successfully!")
    elif use_default and file_exists:
        df_raw = load_excel(default_path)
        st.sidebar.success("✅ Local file loaded successfully!")
    else:
        st.sidebar.error("❌ No data source selected")
        st.stop()
        
except Exception as e:
    st.sidebar.error(f"❌ Could not load file: {e}")
    st.stop()

# Normalize index to month start
df = df_raw.copy()
df.index = to_month_start(df.index)

# Display data info
st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Data Information")
st.sidebar.info(f"**Rows:** {len(df):,}\n**Columns:** {len(df.columns):,}\n**Date range:** {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")

# --------------------------
# Analysis Mode Selection
# --------------------------
st.sidebar.markdown("## 🎯 Analysis Mode")

analysis_mode = st.sidebar.radio(
    "Select Analysis Type",
    ["📊 GPR Index Dashboard", "📈 Credit Score Risk Assessment", "🌍 Multi-Country Comparison"],
    help="Choose your analysis focus"
)

# --------------------------
# Geography Selection
# --------------------------
st.sidebar.markdown("## 🌍 Geography Selection")

# Detect available countries
country_cols = [col for col in df.columns if col.startswith('GPRC_')]
country_risk_cols = [col for col in df.columns if col.startswith('GPRHC_')]
available_countries = list(set([extract_country_code(col) for col in country_cols + country_risk_cols if extract_country_code(col)]))
available_countries.sort()

if analysis_mode == "🌍 Multi-Country Comparison":
    # Multi-select for comparison
    if available_countries:
        country_options = {code: get_country_name(code) for code in available_countries}
        selected_countries = st.sidebar.multiselect(
            "🏳️ Select Countries to Compare", 
            options=available_countries,
            format_func=lambda x: country_options[x],
            default=available_countries[:3] if len(available_countries) >= 3 else available_countries,
            help="Choose multiple countries for comparison"
        )
        
        if not selected_countries:
            st.sidebar.error("❌ Please select at least one country for comparison")
            st.stop()
            
    else:
        st.sidebar.error("❌ No country-specific columns found")
        st.stop()
        
else:
    # Single country or global selection
    analysis_type = st.sidebar.radio(
        "📊 Data Scope",
        ["🌍 Global GPR (Aggregate)", "🏳️ Country-Specific GPR", "📈 Custom Column Selection"],
        help="Choose between global aggregate GPR or country-specific analysis"
    )

    if analysis_type == "🏳️ Country-Specific GPR":
        if available_countries:
            country_options = {code: get_country_name(code) for code in available_countries}
            selected_country_code = st.sidebar.selectbox(
                "🏳️ Select Country", 
                options=available_countries,
                format_func=lambda x: country_options[x],
                help="Choose a country for GPR analysis"
            )
            
            # Build column names based on selected country
            main_col_candidate = f"GPRC_{selected_country_code}"
            risk_col_candidate = f"GPRHC_{selected_country_code}"
            
            # Verify columns exist
            if main_col_candidate in df.columns:
                main_col = main_col_candidate
            else:
                st.sidebar.error(f"❌ Column {main_col_candidate} not found in data")
                st.stop()
                
            # Optional risk column
            acts_col = risk_col_candidate if risk_col_candidate in df.columns else "<none>"
            
            st.sidebar.success(f"✅ Selected: {get_country_name(selected_country_code)}")
            
            # Override other selections for country mode
            threats_col = "<none>"
            
        else:
            st.sidebar.error("❌ No country-specific columns found (GPRC_XXX pattern)")
            st.stop()

    elif analysis_type == "🌍 Global GPR (Aggregate)":
        # Use global aggregate columns
        cols = list(df.columns)
        gpr_like = [c for c in cols if "gpr" in str(c).lower() and len(str(c)) <= 5 and not c.startswith(('GPRC_', 'GPRHC_'))]
        gpra_like = [c for c in cols if "gpra" in str(c).lower() and not c.startswith(('GPRC_', 'GPRHC_'))]
        gprt_like = [c for c in cols if "gprt" in str(c).lower() and not c.startswith(('GPRC_', 'GPRHC_'))]
        
        if gpr_like:
            main_col = st.sidebar.selectbox("📈 Main GPR column", options=gpr_like, index=0)
        else:
            st.sidebar.error("❌ No global GPR columns found")
            st.stop()
            
        acts_col = st.sidebar.selectbox("⚡ Acts (GPRA) column (optional)", options=["<none>"] + gpra_like, index=0)
        threats_col = st.sidebar.selectbox("⚠️ Threats (GPRT) column (optional)", options=["<none>"] + gprt_like, index=0)

    else:  # Custom column selection
        cols = list(df.columns)
        
        main_col = st.sidebar.selectbox("📈 Main column", options=cols, index=0)
        acts_col = st.sidebar.selectbox("⚡ Acts column (optional)", options=["<none>"] + cols, index=0)
        threats_col = st.sidebar.selectbox("⚠️ Threats column (optional)", options=["<none>"] + cols, index=0)

# Parameters (only show for Credit Score mode)
if analysis_mode == "📈 Credit Score Risk Assessment" or analysis_mode == "🌍 Multi-Country Comparison":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Scoring Parameters")

    start_choice = st.sidebar.selectbox("📅 Reference window", ["1985 → present", "Full sample"])
    ref_start = pd.Timestamp("1985-01-01") if "1985" in start_choice else df.index.min()

    w_level = st.sidebar.slider("⚖️ Level weight", 0.0, 1.0, 0.7, 0.05)
    chg_horizon = st.sidebar.selectbox("📈 Change horizon (months)", [1,3,6,12], index=2)
    chg_lookback = st.sidebar.selectbox("👀 Change lookback (years)", [5,10,15], index=1)
    ewma_span = st.sidebar.slider("📊 EWMA smoothing span", 1, 24, 6)

    # Event overlay parameters
    st.sidebar.markdown("**🚨 Event Overlay**")
    use_overlay = st.sidebar.checkbox("Enable event overlay", value=False)
    if use_overlay:
        overlay_thresh = st.sidebar.slider("Event threshold (percentile)", 50, 99, 95)
        overlay_boost = st.sidebar.slider("Event boost points", 1, 20, 5)
        overlay_decay = st.sidebar.slider("Event persistence (months)", 1, 12, 3)

    # Bucket edges
    st.sidebar.markdown("**🏷️ Risk Buckets**")
    edges = (
        st.sidebar.slider("Very Low → Low", 10, 40, 25),
        st.sidebar.slider("Low → Moderate", 30, 60, 50),
        st.sidebar.slider("Moderate → High", 60, 85, 75),
        st.sidebar.slider("High → Extreme", 80, 95, 90)
    )

# --------------------------
# Main Layout Based on Analysis Mode
# --------------------------

if analysis_mode == "📊 GPR Index Dashboard":
    # GPR Index Dashboard
    st.title("📊 Geopolitical Risk Index Dashboard")
    st.markdown("*Historical trends and current levels of geopolitical risk indicators*")
    
    # Get main series
    gpr = df[main_col].dropna()
    
    if len(gpr) == 0:
        st.error("❌ Selected column has no valid data")
        st.stop()
    
    # Latest metrics
    latest_dt = gpr.index.max()
    latest_value = float(gpr.iloc[-1])
    avg_value = float(gpr.mean())
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if analysis_type == "🏳️ Country-Specific GPR":
            st.metric("🏳️ Country", selected_country_code)
        else:
            st.metric("📈 Index", main_col)
    with col2:
        st.metric("📅 Latest Date", latest_dt.strftime("%Y-%m"))
    with col3:
        change_1y = gpr.iloc[-1] - gpr.iloc[-13] if len(gpr) >= 13 else 0
        st.metric("📊 Current Level", f"{latest_value:.1f}", delta=f"{change_1y:.1f} vs 1Y ago")
    with col4:
        st.metric("📈 Historical Avg", f"{avg_value:.1f}")
    
    # Main Chart - Index Level
    st.markdown("---")
    
    fig = go.Figure()
    
    series_name = f"📊 {main_col}"
    if analysis_type == "🏳️ Country-Specific GPR":
        series_name = f"📊 {get_country_name(selected_country_code)} GPR"
    
    fig.add_trace(go.Scatter(
        x=gpr.index, y=gpr.values, 
        name=series_name,
        mode="lines", 
        line=dict(color="#1f77b4", width=2),
        fill='tonexty',
        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Level: %{y:.2f}<extra></extra>"
    ))
    
    # Add average line
    fig.add_hline(y=avg_value, line_dash="dash", line_color="red", 
                  annotation_text=f"Historical Average: {avg_value:.1f}")
    
    fig.update_layout(
        title=f"📈 {series_name} - Historical Levels",
        height=500,
        xaxis=dict(title="📅 Date"),
        yaxis=dict(title="📊 GPR Level"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts for acts/threats if available
    if acts_col != "<none>" and acts_col in df.columns:
        acts_series = df[acts_col].dropna()
        if len(acts_series) > 0:
            st.markdown("### ⚡ Geopolitical Risk Acts")
            fig_acts = go.Figure()
            fig_acts.add_trace(go.Scatter(
                x=acts_series.index, y=acts_series.values,
                name="⚡ GPR Acts",
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
            ))
            fig_acts.update_layout(
                height=400,
                xaxis=dict(title="📅 Date"),
                yaxis=dict(title="⚡ Acts Level"),
            )
            st.plotly_chart(fig_acts, use_container_width=True)

elif analysis_mode == "📈 Credit Score Risk Assessment":
    # Credit Score Risk Assessment (single country/index)
    if analysis_type == "🏳️ Country-Specific GPR":
        title = f"📈 {get_country_name(selected_country_code)} - Credit Score Risk Assessment"
        subtitle = f"*Credit risk scoring based on {get_country_name(selected_country_code)} geopolitical risk*"
    elif analysis_type == "🌍 Global GPR (Aggregate)":
        title = "📈 Global Credit Score Risk Assessment"
        subtitle = "*Credit risk scoring based on global geopolitical risk aggregate*"
    else:
        title = "📈 Custom Credit Score Risk Assessment"
        subtitle = f"*Credit risk scoring based on {main_col}*"
    
    st.title(title)
    st.markdown(subtitle)
    
    # Get main series
    gpr = df[main_col].dropna()
    
    if len(gpr) == 0:
        st.error("❌ Selected column has no valid data")
        st.stop()
    
    # Reference window
    ref_mask = gpr.index >= ref_start
    gpr_ref = gpr[ref_mask]
    
    # Prepare overlay parameters
    overlay_series = None
    overlay_params = None
    if use_overlay and acts_col != "<none>" and acts_col in df.columns:
        overlay_series = df[acts_col]
        overlay_params = (overlay_thresh, overlay_boost, overlay_decay, ref_start)
    
    # Compute credit score
    results = compute_credit_score(gpr, gpr_ref, chg_horizon, chg_lookback, w_level, ewma_span, overlay_series, overlay_params)
    
    score = results['score']
    buckets = score.apply(lambda x: bucket_from_score(x, edges=edges))
    
    # Latest metrics
    if len(score) > 0:
        latest_dt = score.index.max()
        latest_score = float(score.iloc[-1])
        latest_bucket = bucket_from_score(latest_score, edges=edges)
    else:
        latest_dt = None
        latest_score = float('nan')
        latest_bucket = "N/A"
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if analysis_type == "🏳️ Country-Specific GPR":
            st.metric("🏳️ Country", selected_country_code)
        else:
            st.metric("📈 Main Series", main_col)
    with col2:
        latest_date_str = latest_dt.strftime("%Y-%m") if latest_dt else "—"
        st.metric("📅 Latest Date", latest_date_str)
    with col3:
        latest_score_str = f"{latest_score:.1f}" if not pd.isna(latest_score) else "—"
        st.metric("🎯 Credit Score", latest_score_str)
    with col4:
        bucket_emojis = {"Very Low": "🟢", "Low": "🔵", "Moderate": "🟡", "High": "🟠", "Extreme": "🔴", "N/A": "⚪"}
        bucket_display = f"{bucket_emojis.get(latest_bucket, '⚪')} {latest_bucket}"
        st.metric("🏷️ Risk Level", bucket_display)
    
    # Main Chart - Credit Score
    st.markdown("---")
    
    if len(score) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # GPR Level (secondary y-axis)
        fig.add_trace(
            go.Scatter(x=gpr.index, y=gpr.values, name="📊 GPR Level", 
                      line=dict(color="#1f77b4", width=2), opacity=0.7),
            secondary_y=True
        )
        
        # Credit Score (primary y-axis)
        fig.add_trace(
            go.Scatter(x=score.index, y=score.values, name="🎯 Credit Score",
                      line=dict(color="#ff7f0e", width=3)),
            secondary_y=False
        )
        
        # Risk bucket bands
        y_bands = [0, edges[0], edges[1], edges[2], edges[3], 100]
        colors = ["rgba(76, 175, 80, 0.1)", "rgba(33, 150, 243, 0.1)", "rgba(255, 193, 7, 0.1)", 
                  "rgba(255, 87, 34, 0.1)", "rgba(244, 67, 54, 0.1)"]
        labels = ["🟢 Very Low", "🔵 Low", "🟡 Moderate", "🟠 High", "🔴 Extreme"]

        for i in range(5):
            fig.add_shape(
                type="rect",
                x0=gpr.index.min(), x1=gpr.index.max(),
                y0=y_bands[i], y1=y_bands[i+1],
                fillcolor=colors[i],
                opacity=0.3,
                line=dict(width=0)
            )
        
        fig.update_layout(
            title="📈 Credit Score Risk Assessment Over Time",
            height=600,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="📅 Date")
        fig.update_yaxes(title_text="🎯 Credit Score (0-100)", secondary_y=False, range=[0, 100])
        fig.update_yaxes(title_text="📊 GPR Level", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "🌍 Multi-Country Comparison":
    # Multi-Country Comparison
    st.title("🌍 Multi-Country GPR Comparison")
    st.markdown("*Comparative analysis of geopolitical risk across selected countries*")
    
    # Display selected countries
    countries_display = ", ".join([get_country_name(code).split(' ', 1)[1] for code in selected_countries])
    st.info(f"**Selected Countries:** {countries_display}")
    
    # Compute scores for all selected countries
    country_data = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, country_code in enumerate(selected_countries):
        main_col_candidate = f"GPRC_{country_code}"
        risk_col_candidate = f"GPRHC_{country_code}"
        
        if main_col_candidate in df.columns:
            gpr = df[main_col_candidate].dropna()
            
            if len(gpr) > 0:
                # Reference window
                ref_mask = gpr.index >= ref_start
                gpr_ref = gpr[ref_mask]
                
                # Prepare overlay parameters
                overlay_series = None
                overlay_params = None
                if use_overlay and risk_col_candidate in df.columns:
                    overlay_series = df[risk_col_candidate]
                    overlay_params = (overlay_thresh, overlay_boost, overlay_decay, ref_start)
                
                # Compute credit score
                results = compute_credit_score(gpr, gpr_ref, chg_horizon, chg_lookback, w_level, ewma_span, overlay_series, overlay_params)
                
                country_data[country_code] = {
                    'name': get_country_name(country_code),
                    'gpr': gpr,
                    'score': results['score'],
                    'color': color_palette[i % len(color_palette)],
                    'latest_score': float(results['score'].iloc[-1]) if len(results['score']) > 0 else float('nan'),
                    'latest_bucket': bucket_from_score(float(results['score'].iloc[-1]), edges=edges) if len(results['score']) > 0 else "N/A"
                }
    
    if not country_data:
        st.error("❌ No valid data found for selected countries")
        st.stop()
    
    # Summary table
    st.markdown("### 📊 Current Risk Levels")
    
    summary_data = []
    for country_code, data in country_data.items():
        summary_data.append({
            'Country': data['name'],
            'Latest Score': f"{data['latest_score']:.1f}" if not pd.isna(data['latest_score']) else "—",
            'Risk Level': data['latest_bucket'],
            'Data Points': len(data['score']) if not data['score'].empty else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add risk level emojis
    bucket_emojis = {"Very Low": "🟢", "Low": "🔵", "Moderate": "🟡", "High": "🟠", "Extreme": "🔴", "N/A": "⚪"}
    summary_df['Risk Level'] = summary_df['Risk Level'].apply(lambda x: f"{bucket_emojis.get(x, '⚪')} {x}")
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Multi-country comparison charts
    st.markdown("---")
    st.markdown("### 📈 Credit Score Comparison")
    
    # Credit Scores Chart
    fig_scores = go.Figure()
    
    for country_code, data in country_data.items():
        if not data['score'].empty:
            fig_scores.add_trace(go.Scatter(
                x=data['score'].index,
                y=data['score'].values,
                name=data['name'],
                mode='lines',
                line=dict(color=data['color'], width=2),
                hovertemplate=f"<b>{data['name']}</b><br>Date: %{{x}}<br>Score: %{{y:.1f}}<extra></extra>"
            ))
    
    # Add risk level bands
    y_bands = [0, edges[0], edges[1], edges[2], edges[3], 100]
    colors = ["rgba(76, 175, 80, 0.05)", "rgba(33, 150, 243, 0.05)", "rgba(255, 193, 7, 0.05)", 
              "rgba(255, 87, 34, 0.05)", "rgba(244, 67, 54, 0.05)"]
    labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]

    if country_data:
        first_country = list(country_data.values())[0]
        if not first_country['score'].empty:
            x_min, x_max = first_country['score'].index.min(), first_country['score'].index.max()
            
            for i in range(5):
                fig_scores.add_shape(
                    type="rect",
                    x0=x_min, x1=x_max,
                    y0=y_bands[i], y1=y_bands[i+1],
                    fillcolor=colors[i],
                    opacity=0.3,
                    line=dict(width=0)
                )
    
    fig_scores.update_layout(
        title="🎯 Credit Scores Comparison Over Time",
        height=600,
        xaxis=dict(title="📅 Date"),
        yaxis=dict(title="🎯 Credit Score (0-100)", range=[0, 100]),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # GPR Levels Comparison
    st.markdown("### 📊 GPR Levels Comparison")
    
    fig_gpr = go.Figure()
    
    for country_code, data in country_data.items():
        if not data['gpr'].empty:
            fig_gpr.add_trace(go.Scatter(
                x=data['gpr'].index,
                y=data['gpr'].values,
                name=data['name'],
                mode='lines',
                line=dict(color=data['color'], width=2),
                hovertemplate=f"<b>{data['name']}</b><br>Date: %{{x}}<br>GPR: %{{y:.1f}}<extra></extra>"
            ))
    
    fig_gpr.update_layout(
        title="📊 GPR Levels Comparison Over Time",
        height=500,
        xaxis=dict(title="📅 Date"),
        yaxis=dict(title="📊 GPR Level"),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_gpr, use_container_width=True)
    
    # Country Statistics Table
    st.markdown("### 📋 Detailed Statistics")
    
    detailed_stats = []
    for country_code, data in country_data.items():
        stats = {
            'Country': data['name'],
            'Avg GPR Level': f"{data['gpr'].mean():.1f}" if not data['gpr'].empty else "—",
            'Max GPR Level': f"{data['gpr'].max():.1f}" if not data['gpr'].empty else "—",
            'Avg Credit Score': f"{data['score'].mean():.1f}" if not data['score'].empty else "—",
            'Max Credit Score': f"{data['score'].max():.1f}" if not data['score'].empty else "—",
            'Current Score': f"{data['latest_score']:.1f}" if not pd.isna(data['latest_score']) else "—",
            'Current Risk': data['latest_bucket']
        }
        detailed_stats.append(stats)
    
    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df['Current Risk'] = detailed_df['Current Risk'].apply(lambda x: f"{bucket_emojis.get(x, '⚪')} {x}")
    
    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare download data - Credit Scores
        download_scores = pd.DataFrame()
        for country_code, data in country_data.items():
            if not data['score'].empty:
                download_scores[f"{country_code}_Score"] = data['score']
        
        if not download_scores.empty:
            csv_scores = download_scores.to_csv()
            st.download_button(
                "📥 Download Credit Scores (CSV)",
                csv_scores,
                f"multi_country_credit_scores_{len(selected_countries)}countries.csv",
                "text/csv"
            )
    
    with col2:
        # Prepare download data - GPR Levels
        download_gpr = pd.DataFrame()
        for country_code, data in country_data.items():
            if not data['gpr'].empty:
                download_gpr[f"{country_code}_GPR"] = data['gpr']
        
        if not download_gpr.empty:
            csv_gpr = download_gpr.to_csv()
            st.download_button(
                "📥 Download GPR Levels (CSV)",
                csv_gpr,
                f"multi_country_gpr_levels_{len(selected_countries)}countries.csv",
                "text/csv"
            )

# --------------------------
# Information and Help Section
# --------------------------
st.markdown("---")
with st.expander("ℹ️ How to Use This Application"):
    st.markdown("""
    ## 🎯 Analysis Modes
    
    ### 📊 GPR Index Dashboard
    - View historical trends and current levels of GPR indices
    - Compare global vs country-specific geopolitical risk
    - Analyze acts and threats components separately
    
    ### 📈 Credit Score Risk Assessment  
    - Convert GPR data into credit risk scores (0-100 scale)
    - Customize scoring parameters and reference periods
    - Apply event overlays for enhanced risk detection
    - Export results for further analysis
    
    ### 🌍 Multi-Country Comparison
    - Compare credit scores across multiple countries
    - Side-by-side analysis of GPR levels and risk trends
    - Summary statistics and ranking tables
    - Batch export of comparative data
    
    ## 🔧 Key Features
    - **📊 Interactive Charts**: Zoom, pan, and hover for detailed information
    - **⚙️ Customizable Parameters**: Adjust scoring weights, time horizons, and risk thresholds
    - **🎨 Risk Visualization**: Color-coded risk levels and intuitive bucket system
    - **💾 Data Export**: Download processed results in CSV format
    - **🚨 Event Analysis**: Overlay geopolitical events for enhanced risk assessment
    
    ## 📋 Parameter Guide
    - **Level Weight**: Balance between current GPR level vs recent changes
    - **Change Horizon**: Time period for measuring GPR momentum
    - **EWMA Smoothing**: Reduce noise while preserving trends
    - **Event Overlay**: Boost scores during significant geopolitical events
    
    ### ⚠️ Important Notes
    - Credit scores are relative to the chosen reference period
    - Multi-country analysis focuses on comparative risk assessment
    - Event overlays provide temporary score boosts that decay over time
    - All data processing happens locally in your browser
    - Results should be interpreted alongside other risk indicators
    """)

# Footer
st.markdown("---")
st.caption(f"🏗️ Built with Streamlit • 📅 Generated on {date.today().isoformat()} • 🔒 Data stays local")
