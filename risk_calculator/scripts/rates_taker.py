import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import warnings

# Suppress harmless FRED warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 1. Eurozone members (as given)
# ----------------------------------------------------------------------
EUROZONE_MEMBERS = {
    'Austria', 'Belgium', 'Croatia', 'Cyprus', 'Estonia', 'Finland',
    'France', 'Germany', 'Greece', 'Ireland', 'Italy', 'Latvia',
    'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Portugal',
    'Slovakia', 'Slovenia', 'Spain'
}

# ----------------------------------------------------------------------
# 2. FRED ticker mapping (10-year government bond yields)
#    - Most are monthly: IRLTLT01<ISO>FM156N  (F = France, etc.)
#    - Germany has a daily series: DGS10 (Eurozone benchmark)
# ----------------------------------------------------------------------
TICKER_MAP = {
    'Austria':      'IRLTLT01ATM156N',
    'Belgium':      'IRLTLT01BEM156N',
    'Croatia':      'IRLTLT01HRM156N',   # May be missing
    'Cyprus':       'IRLTLT01CYM156N',   # Often missing
    'Estonia':      'IRLTLT01EEM156N',   # Usually missing
    'Finland':      'IRLTLT01FIM156N',
    'France':       'IRLTLT01FRM156N',
    'Germany':      'IRLTLT01DEM156N',   # Monthly; daily alternative: DGS10 (US)
    'Greece':       'IRLTLT01GRM156N',
    'Ireland':      'IRLTLT01IEM156N',
    'Italy':        'IRLTLT01ITM156N',
    'Latvia':       'IRLTLT01LVM156N',   # May be missing
    'Lithuania':    'IRLTLT01LTM156N',   # May be missing
    'Luxembourg':   'IRLTLT01LUM156N',   # Often missing
    'Malta':        'IRLTLT01MTM156N',   # Often missing
    'Netherlands':  'IRLTLT01NLM156N',
    'Portugal':     'IRLTLT01PTM156N',
    'Slovakia':     'IRLTLT01SKM156N',
    'Slovenia':     'IRLTLT01SIM156N',
    'Spain':        'IRLTLT01ESM156N',
}

# ----------------------------------------------------------------------
# 3. Fetch latest available value for each ticker
# ----------------------------------------------------------------------
def get_latest_yield(ticker: str, start: str = '2020-01-01') -> float | None:
    """Return the most recent non-NaN value for a FRED series."""
    try:
        df = web.DataReader(ticker, 'fred', start)
        series = df[ticker].dropna()
        return series.iloc[-1] if not series.empty else None
    except Exception:
        return None

# ----------------------------------------------------------------------
# 4. Build the result table
# ----------------------------------------------------------------------
results = []

for country in sorted(EUROZONE_MEMBERS):
    ticker = TICKER_MAP.get(country)
    if not ticker:
        results.append({'Country': country, '10Y Yield (%)': None, 'FRED Ticker': None})
        continue

    yield_val = get_latest_yield(ticker)
    results.append({
        'Country': country,
        '10Y Yield (%)': round(yield_val, 2) if yield_val is not None else None,
        'FRED Ticker': ticker
    })

# Convert to DataFrame
df_yields = pd.DataFrame(results)

# ----------------------------------------------------------------------
# 5. Display
# ----------------------------------------------------------------------
print("\nEurozone 10-Year Government Bond Yields (Latest Available)")
print("=" * 65)
print(df_yields.to_string(index=False, na_rep='—'))

# Save to CSV
output_path = Path(__file__).parent.parent / "data" / "eurozone_10y_yields.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_yields.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")