# -*- coding: utf-8 -*-
"""
WEO File Structure Analyzer
Helps understand the structure of WEO Excel/CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()  # risk_calculator directory
DATA_DIR = PROJECT_DIR / "data"
WEO_FILE = DATA_DIR / "WEOApr2025all.xls"


def analyze_weo_structure(file_path: Path):
    """Analyze and display WEO file structure."""
    print("="*80)
    print("WEO FILE STRUCTURE ANALYZER")
    print("="*80)
    print(f"\nFile: {file_path}")
    print(f"File exists: {file_path.exists()}")
    print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB" if file_path.exists() else "")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Check file extension and try to determine format
    file_ext = file_path.suffix.lower()
    print(f"File extension: {file_ext}")
    
    # Try different engines and encodings
    engines = [
        ("openpyxl", "openpyxl"),
        ("xlrd", "xlrd"),
    ]
    
    encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
    separators = ['\t', ',', ';']
    
    weo_data = None
    used_engine = None
    used_method = None
    
    # First, try to peek at the file content
    print(f"\n{'─'*80}")
    print("Checking file encoding and format...")
    print(f"{'─'*80}")
    
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(200)
            print(f"  First 200 bytes (hex): {first_bytes[:100].hex()}")
            print(f"  First 200 bytes (ascii attempt): {first_bytes[:100]}")
            
            # Check for BOM
            is_utf16_le = first_bytes.startswith(b'\xff\xfe')  # UTF-16 Little Endian
            is_utf16_be = first_bytes.startswith(b'\xfe\xff')  # UTF-16 Big Endian
            
            if is_utf16_le:
                print(f"  ✓ Detected: UTF-16 Little Endian (BOM: FF FE)")
            elif is_utf16_be:
                print(f"  ✓ Detected: UTF-16 Big Endian (BOM: FE FF)")
            else:
                print(f"  No UTF-16 BOM detected - may be UTF-8 or other encoding")
                
            # Try to decode first bytes to see what it says
            for enc in ['utf-16-le', 'utf-16-be', 'utf-8', 'latin1']:
                try:
                    decoded = first_bytes[:100].decode(enc, errors='ignore')
                    if 'WEO' in decoded or 'Country' in decoded:
                        print(f"  ✓ Decodes as {enc}: '{decoded[:80]}'")
                        break
                except:
                    pass
                    
    except Exception as e:
        print(f"  Could not read file bytes: {e}")
    
    # Try Excel engines first
    for engine_name, engine_val in engines:
        print(f"\n{'─'*80}")
        print(f"Trying Excel engine: {engine_name}")
        print(f"{'─'*80}")
        
        try:
            # Check if file has multiple sheets
            try:
                xl_file = pd.ExcelFile(file_path, engine=engine_val)
                sheet_names = xl_file.sheet_names
                print(f"  Found {len(sheet_names)} sheet(s): {sheet_names}")
            except:
                sheet_names = [None]
            
            for sheet_name in sheet_names:
                try:
                    # Read without header first to see raw structure
                    if sheet_name:
                        df_raw = pd.read_excel(file_path, engine=engine_val, sheet_name=sheet_name, header=None, nrows=20)
                        print(f"  Reading sheet: {sheet_name}")
                    else:
                        df_raw = pd.read_excel(file_path, engine=engine_val, header=None, nrows=20)
                    
                    print(f"  ✓ Successfully read with {engine_name}")
                    print(f"    Shape: {df_raw.shape} (rows, columns)")
                    
                    # Show first few rows
                    print(f"\n    First 3 rows (raw, no header) - first 10 columns:")
                    print(df_raw.iloc[:3, :10].to_string())
                    
                    # Find header row
                    print(f"\n    Searching for header row...")
                    header_row = None
                    for i in range(min(10, len(df_raw))):
                        row_values = [str(val).strip() if pd.notna(val) else '' for val in df_raw.iloc[i].values[:15]]
                        row_str = ' | '.join([v[:20] for v in row_values])
                        
                        # Check if this looks like a header row
                        row_combined = ' '.join(row_values).lower()
                        if 'country' in row_combined and 'subject' in row_combined and 'descriptor' in row_combined:
                            header_row = i
                            print(f"      Row {i}: ✓ Contains 'Country' and 'Subject Descriptor' - HEADER FOUND")
                            print(f"        First 10 columns: {row_values[:10]}")
                        elif 'country' in row_combined:
                            print(f"      Row {i}: ⚠️  Contains 'Country' but may not be header")
                            print(f"        First columns: {row_values[:5]}")
                        elif i < 5:
                            print(f"      Row {i}: First columns: {row_values[:5]}")
                    
                    if header_row is not None:
                        # Read with proper header
                        if sheet_name:
                            weo_data = pd.read_excel(file_path, engine=engine_val, sheet_name=sheet_name, header=header_row)
                        else:
                            weo_data = pd.read_excel(file_path, engine=engine_val, header=header_row)
                        used_engine = engine_name
                        used_method = f"Excel ({engine_name})" + (f" - Sheet: {sheet_name}" if sheet_name else "")
                        print(f"\n    ✓ Read full file with header row {header_row}")
                        print(f"      Shape: {weo_data.shape} (rows, columns)")
                        break
                except Exception as e:
                    if sheet_name:
                        print(f"  Error reading sheet {sheet_name}: {e}")
                    else:
                        print(f"  Error: {e}")
                    continue
            
            if weo_data is not None:
                break
                        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # If Excel didn't work, try CSV/text format with different encodings and separators
    if weo_data is None:
        print(f"\n{'─'*80}")
        print("Trying text/CSV format with different encodings...")
        print(f"{'─'*80}")
        
        # Add UTF-16 variants
        encodings_extended = encodings + ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-16-le-sig', 'utf-16-be-sig']
        
        for encoding in encodings_extended:
            for sep in separators:
                try:
                    print(f"\n  Trying: encoding='{encoding}', sep='{repr(sep)}'")
                    
                    # Special handling for UTF-16
                    if 'utf-16' in encoding.lower():
                        # Read as binary first, then decode
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                # Remove BOM if present
                                if content.startswith(b'\xff\xfe'):
                                    content = content[2:]
                                elif content.startswith(b'\xfe\xff'):
                                    content = content[2:]
                                # Decode
                                text = content.decode(encoding, errors='ignore')
                                # Split into lines
                                lines = text.split('\n')[:25]
                                # Try to parse as CSV
                                import io
                                df_raw = pd.read_csv(io.StringIO('\n'.join(lines)), sep=sep, header=None, on_bad_lines='skip')
                        except Exception as e:
                            print(f"    ❌ UTF-16 read failed: {str(e)[:100]}")
                            continue
                    else:
                        df_raw = pd.read_csv(file_path, sep=sep, encoding=encoding, header=None, nrows=20, 
                                            on_bad_lines='skip', low_memory=False)
                    
                    print(f"    ✓ Successfully read")
                    print(f"    Shape: {df_raw.shape} (rows, columns)")
                    
                    # Show first few rows
                    print(f"\n    First 3 rows (first 10 columns):")
                    print(df_raw.iloc[:3, :10].to_string())
                    
                    # Find header
                    for i in range(min(10, len(df_raw))):
                        row_values = [str(val).strip() if pd.notna(val) else '' for val in df_raw.iloc[i].values[:15]]
                        row_combined = ' '.join(row_values).lower()
                        if 'country' in row_combined and 'subject' in row_combined and 'descriptor' in row_combined:
                            # Read full file
                            if 'utf-16' in encoding.lower():
                                try:
                                    with open(file_path, 'rb') as f:
                                        content = f.read()
                                        if content.startswith(b'\xff\xfe'):
                                            content = content[2:]
                                        elif content.startswith(b'\xfe\xff'):
                                            content = content[2:]
                                        text = content.decode(encoding, errors='ignore')
                                    weo_data = pd.read_csv(io.StringIO(text), sep=sep, header=i, on_bad_lines='skip', low_memory=False)
                                except Exception as e:
                                    print(f"    ⚠️  Full file read failed: {e}")
                                    continue
                            else:
                                weo_data = pd.read_csv(file_path, sep=sep, encoding=encoding, header=i, 
                                                      on_bad_lines='skip', low_memory=False)
                            
                            used_engine = "CSV"
                            used_method = f"CSV (encoding={encoding}, sep={repr(sep)})"
                            print(f"    ✓ Found header in row {i}")
                            print(f"    ✓ Read full file")
                            print(f"      Shape: {weo_data.shape} (rows, columns)")
                            break
                    
                    if weo_data is not None:
                        break
                        
                except Exception as e:
                    error_msg = str(e)
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    print(f"    ❌ Failed: {error_msg}")
                    continue
            
            if weo_data is not None:
                break
    
    if weo_data is None:
        print(f"\n❌ Could not read file with any method")
        return
    
    print(f"\n{'='*80}")
    print(f"FILE STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\n📊 Basic Information:")
    print(f"  Method used: {used_method}")
    print(f"  Total rows: {len(weo_data):,}")
    print(f"  Total columns: {len(weo_data.columns)}")
    print(f"  Memory usage: {weo_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n📋 Column Information:")
    print(f"  Column count: {len(weo_data.columns)}")
    print(f"\n  All columns:")
    for i, col in enumerate(weo_data.columns):
        col_str = str(col)[:50]
        non_null = weo_data[col].notna().sum()
        null_pct = (weo_data[col].isna().sum() / len(weo_data)) * 100
        print(f"    {i+1:3d}. {col_str:50s} | Non-null: {non_null:6,} ({100-null_pct:.1f}%)")
    
    # Key columns
    print(f"\n🔍 Key Column Detection:")
    key_columns = {
        'Country': None,
        'Subject Descriptor': None,
        'Units': None,
        'ISO': None,
        'WEO Country Code': None
    }
    
    for col in weo_data.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()
        # Exact match for Country (not Country/Series-specific Notes)
        if col_str == 'Country':
            key_columns['Country'] = col
        elif 'country' in col_lower and 'series-specific' not in col_lower and 'code' not in col_lower:
            key_columns['Country'] = col
        elif 'subject' in col_lower and 'descriptor' in col_lower:
            key_columns['Subject Descriptor'] = col
        elif col_str == 'Units':
            key_columns['Units'] = col
        elif 'units' in col_lower:
            key_columns['Units'] = col
        elif col_str == 'ISO':
            key_columns['ISO'] = col
        elif 'iso' in col_lower:
            key_columns['ISO'] = col
        elif 'weo country code' in col_lower:
            key_columns['WEO Country Code'] = col
    
    for key, col_name in key_columns.items():
        if col_name:
            print(f"  ✓ {key}: '{col_name}'")
        else:
            print(f"  ❌ {key}: NOT FOUND")
    
    # Year columns
    print(f"\n📅 Year Columns:")
    year_cols = []
    for col in weo_data.columns:
        col_str = str(col).strip()
        try:
            year = int(col_str)
            if 1970 <= year <= 2050:
                year_cols.append((col, year))
        except:
            pass
    
    if year_cols:
        year_cols.sort(key=lambda x: x[1])
        print(f"  Found {len(year_cols)} year columns:")
        print(f"    First 10: {[y[1] for y in year_cols[:10]]}")
        print(f"    Last 10: {[y[1] for y in year_cols[-10:]]}")
        print(f"    Year range: {year_cols[0][1]} - {year_cols[-1][1]}")
    else:
        print(f"  ❌ No year columns detected")
    
    # Sample data
    print(f"\n📊 Sample Data (first 3 rows):")
    if key_columns['Country']:
        display_cols = []
        for key in ['Country', 'Subject Descriptor', 'Units']:
            if key_columns[key]:
                display_cols.append(key_columns[key])
        # Add a few year columns if available
        if year_cols:
            display_cols.extend([y[0] for y in year_cols[:3]])
        
        if display_cols:
            sample = weo_data[display_cols].head(3)
            print(sample.to_string())
    
    # Country information
    if key_columns['Country']:
        print(f"\n🌍 Country Information:")
        country_col = key_columns['Country']
        unique_countries = weo_data[country_col].unique()
        print(f"  Unique countries: {len(unique_countries)}")
        print(f"\n  First 20 countries:")
        # Filter out NaN and convert to string for sorting
        valid_countries = [str(c) for c in unique_countries if pd.notna(c)]
        for i, country in enumerate(sorted(valid_countries)[:20], 1):
            count = len(weo_data[weo_data[country_col] == country])
            print(f"    {i:2d}. {country:40s} ({count} rows)")
        
        # Check for target countries
        target_countries = ['France', 'Germany', 'Italy', 'Netherlands', 'United Kingdom']
        print(f"\n  Target countries in data:")
        for country in target_countries:
            matches = weo_data[weo_data[country_col].str.contains(country, case=False, na=False)]
            if len(matches) > 0:
                exact_match = weo_data[weo_data[country_col] == country]
                print(f"    ✓ {country}:")
                print(f"        Exact match: {len(exact_match)} rows")
                print(f"        Contains match: {len(matches)} rows")
                if len(exact_match) == 0 and len(matches) > 0:
                    similar = matches[country_col].unique()[:3]
                    print(f"        Similar names: {list(similar)}")
            else:
                print(f"    ❌ {country}: NOT FOUND")
    
    # Subject Descriptor information
    if key_columns['Subject Descriptor']:
        print(f"\n📈 Subject Descriptor Information:")
        subject_col = key_columns['Subject Descriptor']
        unique_subjects = weo_data[subject_col].unique()
        print(f"  Unique indicators: {len(unique_subjects)}")
        print(f"\n  First 30 indicators:")
        for i, subject in enumerate(sorted(unique_subjects)[:30], 1):
            count = len(weo_data[weo_data[subject_col] == subject])
            print(f"    {i:2d}. {subject[:60]:60s} ({count} rows)")
    
    # Units information
    if key_columns['Units']:
        print(f"\n📏 Units Information:")
        units_col = key_columns['Units']
        unique_units = weo_data[units_col].unique()
        print(f"  Unique units: {len(unique_units)}")
        print(f"\n  All units:")
        for unit in sorted(unique_units):
            count = len(weo_data[weo_data[units_col] == unit])
            print(f"    • {unit:50s} ({count} rows)")
    
    # Data quality check
    print(f"\n🔍 Data Quality Checks:")
    
    # Check for common indicators we need
    if key_columns['Subject Descriptor']:
        subject_col = key_columns['Subject Descriptor']
        needed_indicators = [
            'General government gross debt',
            'Gross debt',
            'General government net lending/borrowing',
            'Current account balance',
            'Gross domestic product per capita',
            'Unemployment rate',
            'Inflation, average consumer prices',
            'Gross domestic product, constant prices'
        ]
        
        print(f"\n  Checking for required indicators:")
        for indicator in needed_indicators:
            matches = weo_data[weo_data[subject_col].str.contains(indicator, case=False, na=False)]
            if len(matches) > 0:
                print(f"    ✓ '{indicator}': {len(matches)} rows found")
                if key_columns['Units']:
                    units_found = matches[key_columns['Units']].unique()
                    print(f"         Units: {list(units_found)[:3]}")
            else:
                print(f"    ❌ '{indicator}': NOT FOUND")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"✓ File successfully read using: {used_engine}")
    print(f"✓ Total data rows: {len(weo_data):,}")
    print(f"✓ Total columns: {len(weo_data.columns)}")
    if key_columns['Country']:
        print(f"✓ Countries found: {len(weo_data[key_columns['Country']].unique())}")
    if key_columns['Subject Descriptor']:
        print(f"✓ Indicators found: {len(weo_data[key_columns['Subject Descriptor']].unique())}")
    if year_cols:
        print(f"✓ Year range: {year_cols[0][1]} - {year_cols[-1][1]}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_weo_structure(WEO_FILE)

