# Eleutheria - Sovereign Risk Dashboard

Streamlit dashboard for visualizing sovereign default probability assessments for Eurozone countries.

## Overview

This dashboard provides comprehensive visualization and analysis of sovereign risk assessments, including:
- Country-specific risk analysis
- Comparative analysis across Eurozone members
- Time series analysis
- Interactive charts and visualizations

## Project Structure

```
risk_calculator/
├── scripts/
│   └── risk_dashboard.py          # Main Streamlit dashboard application
├── json/                           # Country assessment JSON files
├── images/                         # Logo and banner images
├── data/                           # Source data files
└── statistics_output/              # Generated statistics reports
```

## Deployment

This project is deployed on Streamlit Cloud at: [https://eleutheria-rc.com](https://eleutheria-rc.com)

## Requirements

See `requirements.txt` for all Python dependencies.

Main dependencies:
- Streamlit >= 1.28.0
- Plotly >= 5.17.0
- Pandas >= 2.0.0
- Pillow >= 9.0.0

## Running Locally

To run the dashboard locally:

```bash
streamlit run risk_calculator/scripts/risk_dashboard.py
```

## License

© Eleutheria - All rights reserved

