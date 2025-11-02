# Project Structure

This project has been reorganized for better organization and maintainability.

## Folder Structure

```
risk_calculator/
├── scripts/          # All Python scripts
│   ├── risk_calculator_2.py      # Main risk calculation model
│   ├── risk_dashboard.py          # Streamlit dashboard
│   ├── eurozone_statistics.py    # Statistics analyzer
│   ├── rates_taker.py            # Bond yield data fetcher
│   └── ...                       # Other utility scripts
│
├── data/            # All data source files
│   ├── wgidataset.xlsx           # World Governance Indicators
│   ├── WEOApr2025all.xls         # IMF World Economic Outlook
│   ├── data_gpr_export.xls       # Geopolitical Risk Index
│   ├── eurozone_10y_yields.csv   # Bond yield data
│   └── ...                       # Other data files
│
├── json/            # Output JSON files (country assessments)
│   ├── Austria_default_probability_assessment.json
│   ├── Austria_time_series_analysis.json
│   └── ...                       # Other country assessments
│
└── statistics_output/  # Generated statistics reports
    ├── eurozone_statistics_report.json
    ├── eurozone_summary_table.csv
    ├── correlation_matrix.csv
    └── feature_contributions.csv
```

## Running Scripts

All scripts in the `scripts/` folder are configured to:
- Read data from `../data/` (relative to scripts folder)
- Write outputs to `../json/` or `../statistics_output/` (relative to scripts folder)

### Main Scripts:

1. **risk_calculator_2.py** - Main model execution
   ```bash
   python scripts/risk_calculator_2.py
   ```

2. **risk_dashboard.py** - Streamlit dashboard
   ```bash
   streamlit run scripts/risk_dashboard.py
   ```

3. **eurozone_statistics.py** - Generate statistics
   ```bash
   python scripts/eurozone_statistics.py
   ```

4. **rates_taker.py** - Fetch bond yields
   ```bash
   python scripts/rates_taker.py
   ```

## Path Configuration

All scripts use the following path structure:
- `SCRIPT_DIR = Path(__file__).parent.resolve()` - Scripts directory
- `PROJECT_DIR = SCRIPT_DIR.parent.resolve()` - Main project directory (risk_calculator)
- `DATA_DIR = PROJECT_DIR / "data"` - Data files directory
- `OUTPUT_DIR = PROJECT_DIR / "json"` - JSON output directory

This ensures scripts work correctly regardless of where they are executed from.

