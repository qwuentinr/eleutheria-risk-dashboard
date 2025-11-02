# file_explorer.py

import pandas as pd
import pathlib

# Use the same robust path logic
SCRIPT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIRECTORY.parent.resolve()  # risk_calculator directory
DATA_DIR = PROJECT_DIR / "data"

FILENAMES = [
    "wgidataset.xlsx",
    "WEO_Data.xls",
    "data_gpr_export.xls",
    "ilc_peps01n$defaultview_spreadsheet.xlsx"
]

def explore_file(filepath: pathlib.Path):
    """
    Tries to read a file as Excel and as a text file (CSV/TSV)
    to determine its true format and structure.
    """
    print(f"\n{'='*20} EXPLORING: {filepath.name} {'='*20}")

    if not filepath.exists():
        print("--- STATUS: FILE NOT FOUND ---")
        return

    # --- Step 1: Read the first few lines as raw text ---
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print("\n--- First 3 lines (as raw text): ---")
            for i in range(3):
                print(f.readline().strip())
    except Exception as e:
        print(f"\n--- Could not read as text. This is likely a true binary Excel file. ---")
        print(f"   (Error: {e})")

    # --- Step 2: Try reading with pandas.read_excel ---
    try:
        print("\n--- Attempting to read with pd.read_excel()... ---")
        df_excel = pd.read_excel(filepath, engine='openpyxl' if '.xlsx' in filepath.name else 'xlrd')
        print("   SUCCESS: File is a valid Excel format.")
        print("   Columns:", df_excel.columns.tolist()[:5]) # Show first 5 columns
        print("   DataFrame Head:")
        print(df_excel.head(2))
        return # Stop if successful
    except Exception as e:
        print(f"   FAILED: {e}")

    # --- Step 3: If Excel fails, try reading as a text file (CSV/TSV) ---
    # Try with tab separator (most likely based on the error)
    try:
        print("\n--- Attempting to read with pd.read_csv(sep='\\t')... ---")
        df_tsv = pd.read_csv(filepath, sep='\t')
        print("   SUCCESS: File is a Tab-Separated Values (TSV) text file.")
        print("   Columns:", df_tsv.columns.tolist()[:5])
        print("   DataFrame Head:")
        print(df_tsv.head(2))
        return
    except Exception as e:
        print(f"   FAILED: {e}")
        
    # Try with comma separator as a fallback
    try:
        print("\n--- Attempting to read with pd.read_csv(sep=',')... ---")
        df_csv = pd.read_csv(filepath, sep=',')
        print("   SUCCESS: File is a Comma-Separated Values (CSV) text file.")
        print("   Columns:", df_csv.columns.tolist()[:5])
        print("   DataFrame Head:")
        print(df_csv.head(2))
        return
    except Exception as e:
        print(f"   FAILED: {e}")


def main():
    """Runs the exploration for all configured files."""
    print("Starting file format investigation...")
    for filename in FILENAMES:
        explore_file(DATA_DIR / filename)
    print(f"\n{'='*20} EXPLORATION COMPLETE {'='*20}")

if __name__ == "__main__":
    main()
