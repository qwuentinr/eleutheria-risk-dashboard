# -*- coding: utf-8 -*-
"""
SOVEREIGN RISK ASSESSMENT FRAMEWORK

This script calculates sovereign risk scores based on five key pillars:
1. Institutional and Social Strength
2. Economic Strength and Prospects
3. External Strength
4. Fiscal Strength
5. Monetary Strength

The methodology follows a structured quantitative framework that normalizes
indicators to a 0-100 risk scale, aggregates them into pillar scores, and
computes an overall risk percentage using weighted aggregation.

Required files in the same directory:
- wgidataset.xlsx (World Governance Indicators)
- WEO_Data.xls (IMF World Economic Outlook)
- data_gpr_export.xls (Geopolitical Risk Index)
- ilc_peps01n$defaultview_spreadsheet.xlsx (Eurostat Poverty Data)
"""

import json
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
import pathlib
import warnings

# Import GPR computation functions
try:
    from gpr_index_csv import (
        load_excel as gpr_load_excel,
        process_single_country,
        to_month_start,
        get_country_name
    )
except ImportError:
    # Fallback if import fails
    gpr_load_excel = None
    process_single_country = None
    to_month_start = None
    get_country_name = None

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration for the sovereign risk assessment."""
    
    # List of target countries to process
    TARGET_COUNTRIES = [
        "France",
        "Germany",
        "Netherlands",
        "Italy",
        "United Kingdom"
    ]
    
    SCRIPT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
    PROJECT_DIR = SCRIPT_DIRECTORY.parent.resolve()  # risk_calculator directory
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "json"
    
    FILE_PATHS = {
        "wgi": DATA_DIR / "wgidataset.xlsx",
        "weo": DATA_DIR / "WEO_Data.xls",
        "gpr": DATA_DIR / "data_gpr_export.xls",
        "poverty": DATA_DIR / "ilc_peps01n$defaultview_spreadsheet.xlsx"
    }
    
    @staticmethod
    def get_output_path(country: str) -> pathlib.Path:
        """Returns the output file path for a given country."""
        json_dir = Config.OUTPUT_DIR
        json_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        return json_dir / f"{country}_sovereign_risk_assessment.json"
    
    ANALYSIS_START_YEAR = 2020
    ANALYSIS_END_YEAR = 2029
    ANALYSIS_YEARS = [str(y) for y in range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1)]
    
    REFERENCE_PANEL = [
        'France', 'Germany', 'Italy', 'Spain', 'Netherlands',
        'Belgium', 'Austria', 'Poland', 'Sweden', 'Denmark',
        'Finland', 'Portugal', 'Greece', 'Ireland', 'Czech Republic'
    ]
    
    PILLAR_WEIGHTS = {
        'P1_Institutional_Social': 0.10,
        'P2_Economic': 0.20,
        'P3_External': 0.25,
        'P4_Fiscal': 0.25,
        'P5_Monetary': 0.20
    }
    
    EUROZONE_MEMBERS = [
        'Austria', 'Belgium', 'Croatia', 'Cyprus', 'Estonia', 'Finland',
        'France', 'Germany', 'Greece', 'Ireland', 'Italy', 'Latvia',
        'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Portugal',
        'Slovakia', 'Slovenia', 'Spain'
    ]
    
    RISK_THRESHOLDS = {
        'debt_high': 100.0,
        'debt_critical': 150.0,
        'inflation_moderate': 3.0,
        'inflation_high': 5.0,
        'unemployment_high': 10.0,
        'deficit_high': 3.0
    }

# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

def safe_numeric_conversion(value: Any) -> float:
    """
    Safely converts any value to float, handling various formats.
    
    Args:
        value: Value to convert
        
    Returns:
        Float value or np.nan if conversion fails
    """
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).strip()
    value_str = value_str.replace(',', '').replace(' ', '')
    
    if value_str in ['', 'n/a', '--', '...']:
        return np.nan
    
    try:
        cleaned = ''
        for i, char in enumerate(value_str):
            if char.isdigit() or char == '.' or (char == '-' and i == 0):
                cleaned += char
            elif char in ['e', 'E'] and i > 0:
                cleaned += char
            else:
                break
        
        if cleaned:
            return float(cleaned)
    except (ValueError, TypeError):
        pass
    
    return np.nan

def normalize_to_risk_score(value: float, min_val: float, max_val: float, 
                            invert: bool = False) -> float:
    """
    Normalizes a value to a 0-100 risk scale.
    
    Args:
        value: Value to normalize
        min_val: Minimum benchmark value
        max_val: Maximum benchmark value
        invert: If True, higher values indicate lower risk
        
    Returns:
        Normalized risk score (0-100)
    """
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
        return 50.0
    
    if max_val == min_val:
        return 50.0
    
    normalized = 100 * (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 100)
    
    if invert:
        normalized = 100 - normalized
    
    return normalized

def normalize_percentile_to_risk(percentile: float, invert: bool = True) -> float:
    """
    Converts a percentile rank (0-100) to a risk score.
    
    Args:
        percentile: Percentile value (0-100)
        invert: If True, higher percentile means lower risk
        
    Returns:
        Risk score (0-100)
    """
    if pd.isna(percentile):
        return 50.0
    
    if invert:
        return 100 - percentile
    
    return percentile

def calculate_volatility(values: List[float], exclude_outliers: bool = True) -> float:
    """
    Calculates the standard deviation (volatility) of a series of values.
    
    Args:
        values: List of numeric values
        exclude_outliers: If True, excludes values beyond 2 standard deviations
        
    Returns:
        Standard deviation
    """
    values = [v for v in values if not pd.isna(v)]
    
    if len(values) < 2:
        return np.nan
    
    if exclude_outliers and len(values) > 3:
        mean = np.mean(values)
        std = np.std(values)
        values = [v for v in values if abs(v - mean) <= 2 * std]
    
    if len(values) < 2:
        return np.nan
    
    return np.std(values)

def get_country_code_from_name(country_name: str) -> str:
    """
    Converts country name to ISO3 code (reverse of get_country_name).
    
    Args:
        country_name: Full country name
        
    Returns:
        ISO3 country code (e.g., 'ITA' for Italy)
    """
    country_mapping = {
        'United States': 'USA',
        'Germany': 'DEU',
        'France': 'FRA',
        'United Kingdom': 'GBR',
        'Japan': 'JPN',
        'China': 'CHN',
        'Brazil': 'BRA',
        'India': 'IND',
        'Russia': 'RUS',
        'Canada': 'CAN',
        'Australia': 'AUS',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Netherlands': 'NLD',
        'Switzerland': 'CHE',
        'Sweden': 'SWE',
        'Norway': 'NOR',
        'Denmark': 'DNK',
        'Finland': 'FIN',
        'Belgium': 'BEL',
        'Poland': 'POL',
        'Hungary': 'HUN',
        'Portugal': 'PRT',
        'South Korea': 'KOR',
        'Mexico': 'MEX',
        'Argentina': 'ARG',
        'Chile': 'CHL',
        'Peru': 'PER',
        'Colombia': 'COL',
        'Venezuela': 'VEN',
        'Turkey': 'TUR',
        'Israel': 'ISR',
        'Egypt': 'EGY',
        'Saudi Arabia': 'SAU',
        'Thailand': 'THA',
        'Malaysia': 'MYS',
        'Indonesia': 'IDN',
        'Philippines': 'PHL',
        'Vietnam': 'VNM',
        'Hong Kong': 'HKG',
        'Taiwan': 'TWN',
        'Ukraine': 'UKR',
        'Tunisia': 'TUN',
        'South Africa': 'ZAF',
    }
    return country_mapping.get(country_name, country_name[:3].upper() if len(country_name) >= 3 else country_name)

def compute_gpr_credit_score(gpr_file_path: pathlib.Path, country_name: str, 
                             country_code: str = None) -> float:
    """
    Computes GPR credit score using the gpr_index_csv script.
    
    Args:
        gpr_file_path: Path to GPR Excel file
        country_name: Full country name
        country_code: Optional ISO3 country code (if not provided, will be derived)
        
    Returns:
        Latest GPR credit score (0-100) or np.nan if computation fails
    """
    if process_single_country is None or gpr_load_excel is None:
        print("Warning: GPR computation functions not available, falling back to simple average")
        return np.nan
    
    try:
        # Get country code if not provided
        if country_code is None:
            country_code = get_country_code_from_name(country_name)
        
        # Load GPR data using the gpr script's load function
        df_raw = gpr_load_excel(str(gpr_file_path))
        df = df_raw.copy()
        
        # Normalize index to month start
        if to_month_start is not None:
            df.index = to_month_start(df.index)
        
        # GPR computation parameters (using defaults from gpr_index_csv)
        ref_start = pd.Timestamp('1985-01-01')
        w_level = 0.7
        chg_horizon = 6  # months
        chg_lookback = 10  # years
        ewma_span = 6
        use_overlay = False
        overlay_params = None
        edges = (25, 50, 75, 90)
        
        # Process the country using the GPR script function
        result = process_single_country(
            df, country_code, ref_start, w_level, chg_horizon,
            chg_lookback, ewma_span, use_overlay, overlay_params, edges
        )
        
        if result is not None and len(result) > 0:
            # Get the latest credit score
            latest_score = result['credit_score'].iloc[-1]
            return float(latest_score)
        else:
            print(f"Warning: No GPR credit score computed for {country_name}")
            return np.nan
            
    except Exception as e:
        print(f"Warning: Error computing GPR credit score for {country_name}: {e}")
        return np.nan

# =============================================================================
# 3. DATA LOADING
# =============================================================================

def load_data_sources() -> Dict[str, pd.DataFrame]:
    """
    Loads all required data sources from files.
    
    Returns:
        Dictionary of DataFrames for each data source
    """
    print("Loading data sources...")
    
    try:
        wgi_df = pd.read_excel(Config.FILE_PATHS["wgi"])
        print(f"✓ WGI data loaded: {len(wgi_df)} rows")
        
        weo_df = pd.read_csv(Config.FILE_PATHS["weo"], sep='\t')
        print(f"✓ WEO data loaded: {len(weo_df)} rows")
        
        gpr_df = pd.read_excel(Config.FILE_PATHS["gpr"], engine="xlrd")
        print(f"✓ GPR data loaded: {len(gpr_df)} rows")
        
        poverty_raw = pd.read_excel(Config.FILE_PATHS["poverty"], header=None)
        poverty_df = clean_poverty_data(poverty_raw)
        print(f"✓ Poverty data loaded: {len(poverty_df)} rows")
        
        return {
            "wgi": wgi_df,
            "weo": weo_df,
            "gpr": gpr_df,
            "poverty": poverty_df
        }
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load data - {e}")
        raise

def clean_poverty_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and structures the Eurostat poverty data.
    
    Args:
        raw_df: Raw poverty DataFrame
        
    Returns:
        Cleaned DataFrame with proper headers and country column
    """
    header = raw_df.iloc[9].tolist()
    data_df = raw_df.iloc[10:].copy().reset_index(drop=True)
    
    unique_header = []
    seen = {}
    
    for col in header:
        if pd.isna(col) or col == '':
            col = f'Unnamed_{len(unique_header)}'
        else:
            col_str = str(col)
            if col_str.replace('.', '').isdigit():
                col = str(int(float(col_str)))
            else:
                col = col_str
        
        if col in seen:
            seen[col] += 1
            unique_header.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_header.append(col)
    
    data_df.columns = unique_header
    
    country_col = None
    for col in data_df.columns:
        if 'GEO' in str(col).upper():
            country_col = col
            break
    
    if country_col is None and len(data_df.columns) > 0:
        country_col = data_df.columns[0]
    
    if country_col:
        data_df = data_df.rename(columns={country_col: 'Country'})
        data_df['Country'] = data_df['Country'].astype(str).str.strip()
    
    return data_df

# =============================================================================
# 4. WEO DATA EXTRACTION
# =============================================================================

def get_weo_indicator(weo_df: pd.DataFrame, country: str, 
                      subject: str, years: List[str]) -> List[float]:
    """
    Extracts WEO indicator values for specified years.
    
    Args:
        weo_df: WEO DataFrame
        country: Country name
        subject: Subject descriptor
        years: List of year strings
        
    Returns:
        List of values for each year
    """
    filtered = weo_df[
        (weo_df['Country'] == country) & 
        (weo_df['Subject Descriptor'] == subject)
    ]
    
    if len(filtered) == 0:
        return [np.nan] * len(years)
    
    values = []
    for year in years:
        if year in filtered.columns:
            val = filtered[year].iloc[0]
            values.append(safe_numeric_conversion(val))
        else:
            values.append(np.nan)
    
    return values

def get_weo_average(weo_df: pd.DataFrame, country: str, 
                   subject: str, years: List[str]) -> float:
    """
    Calculates average of WEO indicator across specified years.
    
    Args:
        weo_df: WEO DataFrame
        country: Country name
        subject: Subject descriptor
        years: List of year strings
        
    Returns:
        Average value
    """
    values = get_weo_indicator(weo_df, country, subject, years)
    valid_values = [v for v in values if not pd.isna(v)]
    
    if not valid_values:
        return np.nan
    
    return np.mean(valid_values)

def get_weo_latest(weo_df: pd.DataFrame, country: str, subject: str) -> float:
    """
    Gets the most recent available value for a WEO indicator.
    
    Args:
        weo_df: WEO DataFrame
        country: Country name
        subject: Subject descriptor
        
    Returns:
        Latest available value
    """
    filtered = weo_df[
        (weo_df['Country'] == country) & 
        (weo_df['Subject Descriptor'] == subject)
    ]
    
    if len(filtered) == 0:
        return np.nan
    
    for year in reversed(Config.ANALYSIS_YEARS):
        if year in filtered.columns:
            val = safe_numeric_conversion(filtered[year].iloc[0])
            if not pd.isna(val):
                return val
    
    return np.nan

# =============================================================================
# 5. INDICATOR CALCULATION
# =============================================================================

def calculate_pillar1_indicators(data_sources: Dict[str, pd.DataFrame], 
                                 country: str, country_code_iso3: str = None) -> Dict[str, float]:
    """
    Pillar 1: Institutional and Social Strength
    
    Indicators:
    - Governance indicators (WGI percentile ranks for all 6 indicators):
      * Control of Corruption (cc)
      * Government Effectiveness (ge)
      * Political Stability and Absence of Violence/Terrorism (pv)
      * Rule of Law (rl)
      * Regulatory Quality (rq)
      * Voice and Accountability (va)
    - Poverty rate
    - Geopolitical risk
    """
    indicators = {}
    
    # Get country code if not provided
    if country_code_iso3 is None:
        country_code_iso3 = get_country_code_from_name(country)
    
    # WGI - Extract all six indicators separately
    wgi_df = data_sources["wgi"]
    
    # WGI indicator codes and their full names
    wgi_indicators = {
        'cc': 'control_of_corruption',
        'ge': 'government_effectiveness',
        'pv': 'political_stability',
        'rl': 'rule_of_law',
        'rq': 'regulatory_quality',
        'va': 'voice_accountability'
    }
    
    # Extract percentile rank for each indicator (average across recent years)
    for indicator_code, indicator_name in wgi_indicators.items():
        country_indicator = wgi_df[
            (wgi_df['code'] == country_code_iso3) &
            (wgi_df['indicator'] == indicator_code) &
            (wgi_df['year'] >= Config.ANALYSIS_START_YEAR)
        ]
        
        if len(country_indicator) > 0:
            # Calculate mean percentile rank, converting '..' to NaN
            pctranks = []
            for val in country_indicator['pctrank']:
                if pd.notna(val) and str(val).strip() not in ['..', '', 'nan']:
                    try:
                        pctranks.append(float(val))
                    except (ValueError, TypeError):
                        pass
            
            if pctranks:
                indicators[f'wgi_{indicator_name}_percentile'] = np.mean(pctranks)
            else:
                indicators[f'wgi_{indicator_name}_percentile'] = np.nan
        else:
            indicators[f'wgi_{indicator_name}_percentile'] = np.nan
    
    # Also keep a combined average for backward compatibility
    all_wgi_percentiles = [
        indicators.get(f'wgi_{name}_percentile', np.nan)
        for name in wgi_indicators.values()
    ]
    valid_percentiles = [p for p in all_wgi_percentiles if not pd.isna(p)]
    if valid_percentiles:
        indicators['governance_percentile'] = np.mean(valid_percentiles)
    else:
        indicators['governance_percentile'] = np.nan
    
    # Poverty Rate - Latest available
    poverty_df = data_sources["poverty"]
    country_poverty = poverty_df[
        poverty_df['Country'].str.lower() == country.lower()
    ]
    
    if len(country_poverty) > 0:
        year_cols = [col for col in country_poverty.columns 
                    if col.isdigit() and int(col) >= 2015]
        if year_cols:
            latest_year = max(year_cols, key=lambda x: int(x))
            indicators['poverty_rate'] = safe_numeric_conversion(
                country_poverty[latest_year].iloc[0]
            )
        else:
            indicators['poverty_rate'] = np.nan
    else:
        indicators['poverty_rate'] = np.nan
    
    # Geopolitical Risk - Compute using gpr_index_csv script
    # This uses the sophisticated GPR credit score computation instead of simple average
    indicators['geopolitical_risk'] = compute_gpr_credit_score(
        Config.FILE_PATHS["gpr"],
        country,
        country_code_iso3
    )
    
    # Fallback to simple average if GPR computation fails
    if pd.isna(indicators['geopolitical_risk']):
        print(f"  Falling back to simple GPR average for {country}")
        gpr_df = data_sources["gpr"]
        # Try to handle different possible column structures
        if 'month' in gpr_df.columns:
            gpr_df['month'] = pd.to_datetime(gpr_df['month'])
            recent_gpr = gpr_df[
                gpr_df['month'].dt.year >= (Config.ANALYSIS_END_YEAR - 3)
            ]
            if len(recent_gpr) > 0 and 'GPR' in recent_gpr.columns:
                indicators['geopolitical_risk'] = recent_gpr['GPR'].mean()
        else:
            # Try to find GPR columns by country code
            country_code = get_country_code_from_name(country)
            gpr_col = f'GPRC_{country_code}'
            if gpr_col in gpr_df.columns:
                gpr_series = gpr_df[gpr_col].dropna()
                if len(gpr_series) > 0:
                    indicators['geopolitical_risk'] = gpr_series.iloc[-12:].mean()  # Last 12 months
    
    return indicators

def calculate_pillar2_indicators(data_sources: Dict[str, pd.DataFrame], 
                                 country: str) -> Dict[str, float]:
    """
    Pillar 2: Economic Strength and Prospects
    
    Indicators:
    - GDP per capita (PPP)
    - Real GDP growth (average and volatility)
    - Unemployment rate
    - Investment as % of GDP
    - Gini coefficient (from poverty data if available)
    """
    indicators = {}
    weo_df = data_sources["weo"]
    years = Config.ANALYSIS_YEARS
    
    # GDP per capita PPP
    indicators['gdp_per_capita_ppp'] = get_weo_average(
        weo_df, country,
        'Gross domestic product per capita, current prices',
        years
    )
    
    # Real GDP Growth
    gdp_growth_values = get_weo_indicator(
        weo_df, country,
        'Gross domestic product, constant prices',
        years
    )
    valid_growth = [v for v in gdp_growth_values if not pd.isna(v)]
    
    if valid_growth:
        indicators['gdp_growth_avg'] = np.mean(valid_growth)
        indicators['gdp_growth_volatility'] = calculate_volatility(
            valid_growth, exclude_outliers=True
        )
    else:
        indicators['gdp_growth_avg'] = np.nan
        indicators['gdp_growth_volatility'] = np.nan
    
    # Unemployment
    indicators['unemployment_rate'] = get_weo_average(
        weo_df, country, 'Unemployment rate', years
    )
    
    # Investment
    indicators['investment_pct_gdp'] = get_weo_average(
        weo_df, country, 'Total investment', years
    )
    
    # Gini - Try to extract from poverty data or use manual input
    poverty_df = data_sources["poverty"]
    indicators['gini_coefficient'] = np.nan  # Would need specific Gini data
    
    return indicators

def calculate_pillar3_indicators(data_sources: Dict[str, pd.DataFrame], 
                                 country: str) -> Dict[str, float]:
    """
    Pillar 3: External Strength
    
    Indicators:
    - Current account balance (% GDP)
    - Change in reserves
    - External debt indicators
    - Eurozone membership (qualitative)
    """
    indicators = {}
    weo_df = data_sources["weo"]
    years = Config.ANALYSIS_YEARS
    
    # Current Account
    indicators['current_account_pct_gdp'] = get_weo_average(
        weo_df, country, 'Current account balance', years
    )
    
    # Exports and Imports volumes
    indicators['exports_volume_growth'] = get_weo_average(
        weo_df, country, 'Volume of exports of goods and services', years
    )
    
    indicators['imports_volume_growth'] = get_weo_average(
        weo_df, country, 'Volume of imports of goods and services', years
    )
    
    # Eurozone membership bonus/penalty
    indicators['is_eurozone_member'] = 1.0 if country in Config.EUROZONE_MEMBERS else 0.0
    
    return indicators

def calculate_pillar4_indicators(data_sources: Dict[str, pd.DataFrame], 
                                 country: str) -> Dict[str, float]:
    """
    Pillar 4: Fiscal Strength
    
    Indicators:
    - Government debt (% GDP)
    - Budget balance (% GDP)
    - Government revenue and expenditure
    - Primary balance
    """
    indicators = {}
    weo_df = data_sources["weo"]
    years = Config.ANALYSIS_YEARS
    
    # Gross Debt
    indicators['gross_debt_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government gross debt', years
    )
    
    # Net Debt
    indicators['net_debt_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government net debt', years
    )
    
    # Budget Balance
    indicators['budget_balance_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government net lending/borrowing', years
    )
    
    # Primary Balance
    indicators['primary_balance_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government primary net lending/borrowing', years
    )
    
    # Revenue
    indicators['revenue_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government revenue', years
    )
    
    # Expenditure
    indicators['expenditure_pct_gdp'] = get_weo_average(
        weo_df, country, 'General government total expenditure', years
    )
    
    # Structural Balance
    indicators['structural_balance_pct_potential_gdp'] = get_weo_average(
        weo_df, country, 'General government structural balance', years
    )
    
    return indicators

def calculate_pillar5_indicators(data_sources: Dict[str, pd.DataFrame], 
                                 country: str) -> Dict[str, float]:
    """
    Pillar 5: Monetary Strength
    
    Indicators:
    - Inflation (average and volatility)
    - PPP conversion rate deviation
    - Exchange rate considerations
    """
    indicators = {}
    weo_df = data_sources["weo"]
    years = Config.ANALYSIS_YEARS
    
    # Inflation
    inflation_values = get_weo_indicator(
        weo_df, country,
        'Inflation, average consumer prices',
        years
    )
    valid_inflation = [v for v in inflation_values if not pd.isna(v)]
    
    if valid_inflation:
        indicators['inflation_avg'] = np.mean(valid_inflation)
        indicators['inflation_volatility'] = calculate_volatility(
            valid_inflation, exclude_outliers=True
        )
    else:
        indicators['inflation_avg'] = np.nan
        indicators['inflation_volatility'] = np.nan
    
    # PPP Rate
    indicators['ppp_conversion_rate'] = get_weo_latest(
        weo_df, country, 'Implied PPP conversion rate'
    )
    
    return indicators

def calculate_all_indicators(data_sources: Dict[str, pd.DataFrame], 
                             country: str, country_code_iso3: str = None) -> Dict[str, Dict[str, float]]:
    """
    Calculates all indicators for all pillars for a given country.
    
    Args:
        data_sources: Dictionary of data source DataFrames
        country: Country name
        country_code_iso3: Optional ISO3 country code
        
    Returns:
        Dictionary of pillar indicators
    """
    print(f"\nCalculating indicators for {country}...")
    
    # Get country code if not provided
    if country_code_iso3 is None:
        country_code_iso3 = get_country_code_from_name(country)
    
    all_indicators = {
        'P1_Institutional_Social': calculate_pillar1_indicators(data_sources, country, country_code_iso3),
        'P2_Economic': calculate_pillar2_indicators(data_sources, country),
        'P3_External': calculate_pillar3_indicators(data_sources, country),
        'P4_Fiscal': calculate_pillar4_indicators(data_sources, country),
        'P5_Monetary': calculate_pillar5_indicators(data_sources, country)
    }
    
    return all_indicators

# =============================================================================
# 6. BENCHMARKING AND NORMALIZATION
# =============================================================================

def calculate_reference_benchmarks(data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculates min/max benchmarks for each indicator across the reference panel.
    
    Returns:
        Dictionary mapping pillar -> indicator -> (min, max)
    """
    print("\nCalculating reference benchmarks...")
    
    benchmarks = {}
    
    for country in Config.REFERENCE_PANEL:
        country_indicators = calculate_all_indicators(data_sources, country)
        
        for pillar, indicators in country_indicators.items():
            if pillar not in benchmarks:
                benchmarks[pillar] = {}
            
            for indicator, value in indicators.items():
                if indicator not in benchmarks[pillar]:
                    benchmarks[pillar][indicator] = []
                
                if not pd.isna(value):
                    benchmarks[pillar][indicator].append(value)
    
    # Convert lists to (min, max) tuples
    min_max_benchmarks = {}
    for pillar, indicators in benchmarks.items():
        min_max_benchmarks[pillar] = {}
        for indicator, values in indicators.items():
            if values:
                min_max_benchmarks[pillar][indicator] = (min(values), max(values))
            else:
                min_max_benchmarks[pillar][indicator] = (0.0, 100.0)
    
    return min_max_benchmarks

def normalize_indicators_to_risk(indicators: Dict[str, Dict[str, float]], 
                                benchmarks: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Normalizes all indicators to 0-100 risk scores using benchmarks.
    
    Higher risk score = Higher risk
    0 = Minimal risk, 100 = Maximum risk
    """
    normalized = {}
    
    # Define which indicators should be inverted (higher value = lower risk)
    invert_indicators = {
        'P1_Institutional_Social': [
            'governance_percentile',
            'wgi_control_of_corruption_percentile',
            'wgi_government_effectiveness_percentile',
            'wgi_political_stability_percentile',
            'wgi_rule_of_law_percentile',
            'wgi_regulatory_quality_percentile',
            'wgi_voice_accountability_percentile'
        ],
        'P2_Economic': ['gdp_per_capita_ppp', 'gdp_growth_avg', 'investment_pct_gdp'],
        'P3_External': ['current_account_pct_gdp', 'exports_volume_growth'],
        'P4_Fiscal': ['revenue_pct_gdp', 'primary_balance_pct_gdp'],
        'P5_Monetary': []
    }
    
    for pillar, pillar_indicators in indicators.items():
        normalized[pillar] = {}
        
        for indicator, value in pillar_indicators.items():
            if indicator in benchmarks.get(pillar, {}):
                min_val, max_val = benchmarks[pillar][indicator]
                should_invert = indicator in invert_indicators.get(pillar, [])
                
                # Special handling for percentiles
                if 'percentile' in indicator.lower():
                    normalized[pillar][indicator] = normalize_percentile_to_risk(
                        value, invert=True
                    )
                else:
                    normalized[pillar][indicator] = normalize_to_risk_score(
                        value, min_val, max_val, invert=should_invert
                    )
            else:
                normalized[pillar][indicator] = 50.0
    
    return normalized

# =============================================================================
# 7. RISK AGGREGATION
# =============================================================================

def calculate_pillar_scores(normalized_indicators: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates normalized indicators into pillar risk scores.
    
    Returns:
        Dictionary mapping pillar name to risk score (0-100)
    """
    pillar_scores = {}
    
    for pillar, indicators in normalized_indicators.items():
        valid_scores = [score for score in indicators.values() if not pd.isna(score)]
        
        if valid_scores:
            pillar_scores[pillar] = np.mean(valid_scores)
        else:
            pillar_scores[pillar] = 50.0
    
    return pillar_scores

def calculate_overall_risk(pillar_scores: Dict[str, float]) -> float:
    """
    Calculates overall risk percentage using weighted pillar scores.
    
    Returns:
        Overall risk score (0-100)
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    for pillar, score in pillar_scores.items():
        if pillar in Config.PILLAR_WEIGHTS and not pd.isna(score):
            weight = Config.PILLAR_WEIGHTS[pillar]
            weighted_sum += score * weight
            total_weight += weight
    
    if total_weight > 0:
        overall_risk = weighted_sum / total_weight
    else:
        overall_risk = 50.0
    
    return np.clip(overall_risk, 0, 100)

def assign_risk_rating(risk_score: float) -> str:
    """
    Assigns a qualitative rating based on risk score.
    
    Risk Scale:
    0-10: AAA (Minimal Risk)
    10-20: AA (Very Low Risk)
    20-30: A (Low Risk)
    30-40: BBB (Moderate Risk)
    40-50: BB (Elevated Risk)
    50-60: B (High Risk)
    60-70: CCC (Very High Risk)
    70-85: CC (Extremely High Risk)
    85-100: C/D (Default Risk)
    """
    if risk_score <= 10:
        return "AAA"
    elif risk_score <= 20:
        return "AA"
    elif risk_score <= 30:
        return "A"
    elif risk_score <= 40:
        return "BBB"
    elif risk_score <= 50:
        return "BB"
    elif risk_score <= 60:
        return "B"
    elif risk_score <= 70:
        return "CCC"
    elif risk_score <= 85:
        return "CC"
    else:
        return "C/D"

# =============================================================================
# 8. REPORTING
# =============================================================================

def generate_risk_report(country: str, raw_indicators: Dict[str, Dict[str, float]], 
                        normalized_indicators: Dict[str, Dict[str, float]], 
                        pillar_scores: Dict[str, float], 
                        overall_risk: float) -> Dict[str, Any]:
    """
    Generates a comprehensive risk assessment report.
    
    Returns:
        Dictionary containing full assessment details
    """
    report = {
        "metadata": {
            "country": country,
            "assessment_date": datetime.now().isoformat(),
            "analysis_period": f"{Config.ANALYSIS_START_YEAR}-{Config.ANALYSIS_END_YEAR}",
            "methodology_version": "1.0"
        },
        "overall_assessment": {
            "risk_score": round(overall_risk, 2),
            "risk_rating": assign_risk_rating(overall_risk),
            "risk_interpretation": get_risk_interpretation(overall_risk)
        },
        "pillar_scores": {
            pillar: round(score, 2) 
            for pillar, score in pillar_scores.items()
        },
        "raw_indicators": convert_dict_to_serializable(raw_indicators),
        "normalized_indicators": convert_dict_to_serializable(normalized_indicators),
        "pillar_weights": Config.PILLAR_WEIGHTS
    }
    
    return report

def get_risk_interpretation(risk_score: float) -> str:
    """
    Provides a textual interpretation of the risk score.
    """
    if risk_score <= 10:
        return "Minimal sovereign risk. Exceptionally strong fundamentals across all pillars."
    elif risk_score <= 20:
        return "Very low sovereign risk. Strong institutional, economic, and fiscal position."
    elif risk_score <= 30:
        return "Low sovereign risk. Sound fundamentals with minor areas of concern."
    elif risk_score <= 40:
        return "Moderate sovereign risk. Generally stable with some vulnerabilities."
    elif risk_score <= 50:
        return "Elevated sovereign risk. Notable vulnerabilities requiring monitoring."
    elif risk_score <= 60:
        return "High sovereign risk. Significant weaknesses across multiple pillars."
    elif risk_score <= 70:
        return "Very high sovereign risk. Severe vulnerabilities and potential stress scenarios."
    elif risk_score <= 85:
        return "Extremely high sovereign risk. Critical vulnerabilities and high default probability."
    else:
        return "Imminent default risk. Severe crisis conditions across all dimensions."

def convert_dict_to_serializable(obj: Any) -> Any:
    """
    Converts numpy types and other non-serializable objects to JSON-compatible types.
    """
    if isinstance(obj, dict):
        return {key: convert_dict_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def print_summary_report(report: Dict[str, Any]) -> None:
    """
    Prints a formatted summary of the risk assessment.
    """
    print("\n" + "="*80)
    print(f"SOVEREIGN RISK ASSESSMENT: {report['metadata']['country'].upper()}")
    print("="*80)
    
    print(f"\nAssessment Date: {report['metadata']['assessment_date']}")
    print(f"Analysis Period: {report['metadata']['analysis_period']}")
    
    print("\n" + "-"*80)
    print("OVERALL RISK ASSESSMENT")
    print("-"*80)
    print(f"Risk Score: {report['overall_assessment']['risk_score']}% (0=Safest, 100=Default)")
    print(f"Risk Rating: {report['overall_assessment']['risk_rating']}")
    print(f"Interpretation: {report['overall_assessment']['risk_interpretation']}")
    
    print("\n" + "-"*80)
    print("PILLAR RISK SCORES")
    print("-"*80)
    for pillar, score in report['pillar_scores'].items():
        weight = Config.PILLAR_WEIGHTS.get(pillar, 0) * 100
        print(f"{pillar}: {score}% (Weight: {weight}%)")
    
    print("\n" + "="*80)

# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================

def process_country_assessment(country: str, data_sources: Dict[str, pd.DataFrame], 
                               benchmarks: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """
    Processes risk assessment for a single country.
    
    Args:
        country: Country name
        data_sources: Dictionary of data source DataFrames
        benchmarks: Reference benchmarks for normalization
    """
    country_code = get_country_code_from_name(country)
    
    # Calculate indicators for target country
    raw_indicators = calculate_all_indicators(
        data_sources, 
        country,
        country_code
    )
    
    # Normalize to risk scores
    normalized_indicators = normalize_indicators_to_risk(
        raw_indicators, 
        benchmarks
    )
    
    # Calculate pillar scores
    pillar_scores = calculate_pillar_scores(normalized_indicators)
    
    # Calculate overall risk
    overall_risk = calculate_overall_risk(pillar_scores)
    
    # Generate report
    report = generate_risk_report(
        country,
        raw_indicators,
        normalized_indicators,
        pillar_scores,
        overall_risk
    )
    
    # Print summary
    print_summary_report(report)
    
    # Save to JSON
    output_path = Config.get_output_path(country)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_path}")

def main():
    """
    Main execution function for sovereign risk assessment.
    Processes all countries in TARGET_COUNTRIES list.
    """
    print("="*80)
    print("SOVEREIGN RISK ASSESSMENT FRAMEWORK")
    print("="*80)
    
    # Load data (shared across all countries)
    data_sources = load_data_sources()
    
    # Calculate benchmarks (shared across all countries)
    benchmarks = calculate_reference_benchmarks(data_sources)
    
    # Process each country
    total_countries = len(Config.TARGET_COUNTRIES)
    print(f"\nProcessing {total_countries} countries...")
    print("="*80)
    
    for idx, country in enumerate(Config.TARGET_COUNTRIES, 1):
        print(f"\n{'='*80}")
        print(f"COUNTRY {idx}/{total_countries}: {country.upper()}")
        print(f"{'='*80}")
        
        try:
            process_country_assessment(country, data_sources, benchmarks)
            print(f"\n✓ Completed assessment for {country}")
        except Exception as e:
            print(f"\n✗ ERROR processing {country}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL ASSESSMENTS COMPLETE")
    print("="*80)
    print(f"\nGenerated {total_countries} JSON files:")
    for country in Config.TARGET_COUNTRIES:
        output_path = Config.get_output_path(country)
        print(f"  - {output_path.name}")
    print()

if __name__ == "__main__":
    main()
