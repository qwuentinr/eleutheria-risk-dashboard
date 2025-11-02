import pandas as pd
from scipy.stats import rankdata
from datetime import date
import argparse
import os

# --------------------------
# Utilities
# --------------------------
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
        'USA': 'United States',
        'DEU': 'Germany', 
        'FRA': 'France',
        'GBR': 'United Kingdom',
        'JPN': 'Japan',
        'CHN': 'China',
        'BRA': 'Brazil',
        'IND': 'India',
        'RUS': 'Russia',
        'CAN': 'Canada',
        'AUS': 'Australia',
        'ITA': 'Italy',
        'ESP': 'Spain',
        'NLD': 'Netherlands',
        'CHE': 'Switzerland',
        'SWE': 'Sweden',
        'NOR': 'Norway',
        'DNK': 'Denmark',
        'FIN': 'Finland',
        'BEL': 'Belgium',
        'POL': 'Poland',
        'HUN': 'Hungary',
        'PRT': 'Portugal',
        'KOR': 'South Korea',
        'MEX': 'Mexico',
        'ARG': 'Argentina',
        'CHL': 'Chile',
        'PER': 'Peru',
        'COL': 'Colombia',
        'VEN': 'Venezuela',
        'TUR': 'Turkey',
        'ISR': 'Israel',
        'EGY': 'Egypt',
        'SAU': 'Saudi Arabia',
        'THA': 'Thailand',
        'MYS': 'Malaysia',
        'IDN': 'Indonesia',
        'PHL': 'Philippines',
        'VNM': 'Vietnam',
        'HKG': 'Hong Kong',
        'TWN': 'Taiwan',
        'UKR': 'Ukraine',
        'TUN': 'Tunisia',
        'ZAF': 'South Africa',
    }
    return country_mapping.get(country_code, country_code)


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


def compute_credit_score(gpr, gpr_ref, chg_horizon, chg_lookback, w_level, ewma_span, 
                         overlay_series=None, overlay_params=None):
    """
    Compute a 0-100 credit score from GPR data.
    
    Parameters:
    - gpr: main GPR series
    - gpr_ref: reference window for percentile calculations
    - chg_horizon: months for momentum calculation
    - chg_lookback: years of data for change percentiles
    - w_level: weight for level component (0-1)
    - ewma_span: smoothing span
    - overlay_series: optional series for event overlay
    - overlay_params: (threshold, boost, decay, ref_start) for overlay
    """
    # Level percentile
    pct_level = pct_rank(gpr, gpr_ref)
    
    # Change momentum
    chg = gpr.diff(periods=chg_horizon)
    lookback_start = gpr.index.max() - pd.DateOffset(years=chg_lookback)
    chg_ref = chg[chg.index >= lookback_start]
    pct_chg = pct_rank(chg, chg_ref)
    
    # Weighted score
    score_raw = w_level * pct_level + (1 - w_level) * pct_chg
    
    # Event overlay
    overlay = pd.Series(0.0, index=score_raw.index)
    
    if overlay_series is not None and overlay_params is not None:
        overlay_thresh, overlay_boost, overlay_decay, ref_start = overlay_params
        acts_clean = overlay_series.dropna()
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


def process_single_country(df, country_code, ref_start, w_level, chg_horizon, 
                          chg_lookback, ewma_span, use_overlay, overlay_params, edges):
    """Process a single country and return results."""
    main_col = f"GPRC_{country_code}"
    risk_col = f"GPRHC_{country_code}"
    
    if main_col not in df.columns:
        print(f"Warning: Column {main_col} not found, skipping {country_code}")
        return None
    
    gpr = df[main_col].dropna()
    
    if len(gpr) == 0:
        print(f"Warning: No valid data for {country_code}, skipping")
        return None
    
    # Reference window
    ref_mask = gpr.index >= ref_start
    gpr_ref = gpr[ref_mask]
    
    # Prepare overlay parameters
    overlay_series = None
    overlay_params_final = None
    if use_overlay and risk_col in df.columns:
        overlay_series = df[risk_col]
        overlay_params_final = overlay_params
    
    # Compute credit score
    results = compute_credit_score(gpr, gpr_ref, chg_horizon, chg_lookback, 
                                   w_level, ewma_span, overlay_series, overlay_params_final)
    
    # Create output dataframe
    output = pd.DataFrame({
        'date': gpr.index,
        'country_code': country_code,
        'country_name': get_country_name(country_code),
        'gpr_level': gpr.values,
        'credit_score': results['score'].values,
        'level_percentile': results['level_pct'].values,
        'change_percentile': results['change_pct'].values,
        'score_raw': results['score_raw'].values,
    })
    
    # Add risk bucket
    output['risk_bucket'] = output['credit_score'].apply(lambda x: bucket_from_score(x, edges))
    
    return output


def main():
    parser = argparse.ArgumentParser(description='GPR Analytics - CSV Output')
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent.resolve()
    data_dir = project_dir / "data"
    
    parser.add_argument('--input', default=str(data_dir / 'data_gpr_export.xls'), 
                       help='Input Excel file path')
    parser.add_argument('--output', default=str(data_dir / 'gpr_results.csv'), 
                       help='Output CSV file path')
    parser.add_argument('--countries', nargs='+', 
                       help='Country codes to process (e.g., USA DEU FRA). If not specified, all countries will be processed.')
    parser.add_argument('--ref-start', default='1985-01-01', 
                       help='Reference window start date (YYYY-MM-DD)')
    parser.add_argument('--w-level', type=float, default=0.7, 
                       help='Level weight (0-1)')
    parser.add_argument('--chg-horizon', type=int, default=6, 
                       help='Change horizon in months')
    parser.add_argument('--chg-lookback', type=int, default=10, 
                       help='Change lookback in years')
    parser.add_argument('--ewma-span', type=int, default=6, 
                       help='EWMA smoothing span')
    parser.add_argument('--use-overlay', action='store_true', 
                       help='Enable event overlay')
    parser.add_argument('--overlay-thresh', type=float, default=95, 
                       help='Event threshold percentile')
    parser.add_argument('--overlay-boost', type=float, default=5, 
                       help='Event boost points')
    parser.add_argument('--overlay-decay', type=int, default=3, 
                       help='Event persistence in months')
    parser.add_argument('--bucket-edges', nargs=4, type=int, default=[25, 50, 75, 90],
                       help='Risk bucket edges (4 values)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df_raw = load_excel(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Normalize index to month start
    df = df_raw.copy()
    df.index = to_month_start(df.index)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    
    # Detect available countries
    country_cols = [col for col in df.columns if col.startswith('GPRC_')]
    available_countries = sorted(list(set([extract_country_code(col) for col in country_cols if extract_country_code(col)])))
    
    print(f"Available countries: {', '.join(available_countries)}")
    
    # Select countries to process
    if args.countries:
        countries_to_process = [c for c in args.countries if c in available_countries]
        if not countries_to_process:
            print(f"Error: None of the specified countries found in data")
            return
    else:
        countries_to_process = available_countries
    
    print(f"Processing {len(countries_to_process)} countries: {', '.join(countries_to_process)}")
    
    # Set up parameters
    ref_start = pd.Timestamp(args.ref_start)
    edges = tuple(args.bucket_edges)
    overlay_params = (args.overlay_thresh, args.overlay_boost, args.overlay_decay, ref_start)
    
    # Process all countries
    all_results = []
    
    for country_code in countries_to_process:
        print(f"Processing {country_code} ({get_country_name(country_code)})...")
        result = process_single_country(
            df, country_code, ref_start, args.w_level, args.chg_horizon,
            args.chg_lookback, args.ewma_span, args.use_overlay, overlay_params, edges
        )
        
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("Error: No valid results generated")
        return
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")
    print(f"  Total rows: {len(final_df):,}")
    print(f"  Countries: {final_df['country_code'].nunique()}")
    print(f"  Date range: {final_df['date'].min().strftime('%Y-%m')} to {final_df['date'].max().strftime('%Y-%m')}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    summary = final_df.groupby('country_code').agg({
        'credit_score': ['mean', 'std', 'min', 'max'],
        'gpr_level': ['mean', 'std'],
        'date': 'count'
    }).round(2)
    print(summary)
    
    # Print latest scores
    print("\n--- Latest Credit Scores ---")
    latest = final_df.loc[final_df.groupby('country_code')['date'].idxmax()]
    latest_summary = latest[['country_name', 'date', 'credit_score', 'risk_bucket']].sort_values('credit_score', ascending=False)
    print(latest_summary.to_string(index=False))


if __name__ == "__main__":
    main()
