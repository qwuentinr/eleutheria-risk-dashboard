# -*- coding: utf-8 -*-
"""
EUROZONE RISK STATISTICS ANALYZER
Computes comprehensive statistics and analysis of model outputs for all Eurozone members
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()  # risk_calculator directory
JSON_DIR = PROJECT_DIR / "json"
OUTPUT_DIR = PROJECT_DIR / "statistics_output"

EUROZONE_MEMBERS = [
    "Austria", "Belgium", "Croatia", "Cyprus", "Estonia", "Finland",
    "France", "Germany", "Greece", "Ireland", "Italy", "Latvia",
    "Lithuania", "Luxembourg", "Malta", "Netherlands", "Portugal",
    "Slovakia", "Slovenia", "Spain"
]

# Rating order for sorting
RATING_ORDER = [
    'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
    'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D'
]


def load_assessment(country: str) -> Optional[Dict[str, Any]]:
    """Load assessment JSON for a country."""
    json_path = JSON_DIR / f"{country}_default_probability_assessment.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {country}: {e}")
        return None


def load_all_assessments() -> Dict[str, Dict[str, Any]]:
    """Load all assessments."""
    assessments = {}
    for country in EUROZONE_MEMBERS:
        assessment = load_assessment(country)
        if assessment:
            assessments[country] = assessment
        else:
            print(f"Warning: No assessment found for {country}")
    
    return assessments


def compute_basic_statistics(assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute basic statistics across all countries."""
    if not assessments:
        return {}
    
    # Extract key metrics
    default_probs = []
    ratings = []
    countries_list = []
    
    for country, assessment in assessments.items():
        overall = assessment.get('overall_assessment', {})
        default_probs.append(pd.to_numeric(overall.get('default_probability_5yr'), errors='coerce'))
        ratings.append(overall.get('rating', 'N/A'))
        countries_list.append(country)
    
    # Convert to numpy arrays for easier computation
    default_probs = np.array([p for p in default_probs if not pd.isna(p)])
    
    stats = {
        'total_countries': len(assessments),
        'countries_with_data': len(default_probs),
        'default_probability': {
            'mean': float(np.mean(default_probs)) if len(default_probs) > 0 else np.nan,
            'median': float(np.median(default_probs)) if len(default_probs) > 0 else np.nan,
            'std': float(np.std(default_probs)) if len(default_probs) > 0 else np.nan,
            'min': float(np.min(default_probs)) if len(default_probs) > 0 else np.nan,
            'max': float(np.max(default_probs)) if len(default_probs) > 0 else np.nan,
            'q25': float(np.percentile(default_probs, 25)) if len(default_probs) > 0 else np.nan,
            'q75': float(np.percentile(default_probs, 75)) if len(default_probs) > 0 else np.nan,
            'coefficient_of_variation': float(np.std(default_probs) / np.mean(default_probs)) if len(default_probs) > 0 and np.mean(default_probs) > 0 else np.nan
        },
        'rating_distribution': {}
    }
    
    # Rating distribution
    for rating in RATING_ORDER:
        stats['rating_distribution'][rating] = ratings.count(rating)
    
    return stats


def analyze_feature_contributions(assessments: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Analyze feature contributions across countries."""
    feature_data = []
    
    for country, assessment in assessments.items():
        key_drivers = assessment.get('key_drivers', {})
        risk_factors = key_drivers.get('top_risk_factors', [])
        
        for factor in risk_factors:
            feature_data.append({
                'country': country,
                'feature': factor.get('feature', ''),
                'contribution': factor.get('contribution', 0),
                'direction': factor.get('direction', 'neutral')
            })
    
    if not feature_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(feature_data)
    
    # Aggregate statistics by feature
    feature_stats = df.groupby('feature')['contribution'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    feature_stats.columns = ['Feature', 'Mean Contribution', 'Std Contribution',
                            'Min Contribution', 'Max Contribution', 'Country Count']
    
    return feature_stats.sort_values('Mean Contribution', key=lambda x: x.abs(), ascending=False)


def create_summary_table(assessments: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create comprehensive summary table."""
    summary_data = []
    
    for country, assessment in assessments.items():
        overall = assessment.get('overall_assessment', {})
        uncertainty = assessment.get('uncertainty', {})
        scenarios = assessment.get('scenario_analysis', {})
        thresholds = assessment.get('threshold_analysis', {})
        
        # Extract feature values
        raw_features = assessment.get('raw_features', {})
        
        summary_data.append({
            'Country': country,
            'Default Probability (%)': overall.get('default_probability_5yr', np.nan),
            'Rating': overall.get('rating', 'N/A'),
            'Rating Outlook': overall.get('rating_outlook', 'N/A'),
            'Confidence Interval Lower': uncertainty.get('confidence_interval_90', {}).get('lower', np.nan),
            'Confidence Interval Upper': uncertainty.get('confidence_interval_90', {}).get('upper', np.nan),
            'Volatility': uncertainty.get('volatility', np.nan),
            'Base Case PD': scenarios.get('base_case', {}).get('probability', np.nan),
            'Upside Case PD': scenarios.get('upside_case', {}).get('probability', np.nan),
            'Downside Case PD': scenarios.get('downside_case', {}).get('probability', np.nan),
            'Expected PD': scenarios.get('expected_probability', np.nan),
            'Debt/GDP': raw_features.get('debt_gdp', np.nan),
            'Deficit/GDP': raw_features.get('deficit_gdp', np.nan),
            'Current Account/GDP': raw_features.get('current_account_gdp', np.nan),
            'Inflation (5yr)': raw_features.get('inflation_5yr', np.nan),
            'GDP per Capita': raw_features.get('gdp_per_capita', np.nan),
            'Unemployment': raw_features.get('unemployment', np.nan),
            'Institutional Quality': raw_features.get('institutional_quality', np.nan),
        })
    
    return pd.DataFrame(summary_data)


def analyze_threshold_status(assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze threshold status across countries."""
    threshold_summary = {
        'debt_gdp': {'safe': 0, 'warning': 0, 'critical': 0, 'total': 0},
        'deficit_gdp': {'safe': 0, 'warning': 0, 'critical': 0, 'total': 0},
        'current_account_gdp': {'safe': 0, 'warning': 0, 'critical': 0, 'total': 0},
        'inflation_5yr': {'safe': 0, 'warning': 0, 'critical': 0, 'total': 0},
        'reserves_months': {'safe': 0, 'warning': 0, 'critical': 0, 'total': 0}
    }
    
    threshold_values = {
        'debt_gdp': [],
        'deficit_gdp': [],
        'current_account_gdp': [],
        'inflation_5yr': [],
        'reserves_months': []
    }
    
    for country, assessment in assessments.items():
        thresholds = assessment.get('threshold_analysis', {})
        
        for threshold_name in threshold_summary.keys():
            if threshold_name in thresholds:
                status = thresholds[threshold_name].get('status', 'unknown')
                value = pd.to_numeric(thresholds[threshold_name].get('value'), errors='coerce')
                
                if status in ['safe', 'warning', 'critical']:
                    threshold_summary[threshold_name][status] += 1
                    threshold_summary[threshold_name]['total'] += 1
                
                if not pd.isna(value):
                    threshold_values[threshold_name].append(value)
    
    # Compute statistics for threshold values
    for threshold_name in threshold_values:
        values = threshold_values[threshold_name]
        if values:
            threshold_summary[threshold_name]['mean_value'] = float(np.mean(values))
            threshold_summary[threshold_name]['median_value'] = float(np.median(values))
            threshold_summary[threshold_name]['min_value'] = float(np.min(values))
            threshold_summary[threshold_name]['max_value'] = float(np.max(values))
    
    return threshold_summary


def identify_outliers(assessments: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Identify outlier countries."""
    outliers = {
        'high_pd': [],
        'low_pd': [],
        'extreme_ratings': []
    }
    
    default_probs = {}
    ratings = {}
    
    for country, assessment in assessments.items():
        overall = assessment.get('overall_assessment', {})
        pd_val = pd.to_numeric(overall.get('default_probability_5yr'), errors='coerce')
        rating_val = overall.get('rating', 'N/A')
        
        if not pd.isna(pd_val):
            default_probs[country] = pd_val
        if rating_val != 'N/A':
            ratings[country] = rating_val
    
    if default_probs:
        pd_values = list(default_probs.values())
        pd_mean = np.mean(pd_values)
        pd_std = np.std(pd_values)
        
        for country, pd_val in default_probs.items():
            z_score = (pd_val - pd_mean) / pd_std if pd_std > 0 else 0
            if z_score > 2:
                outliers['high_pd'].append(country)
            elif z_score < -2:
                outliers['low_pd'].append(country)
    
    # Extreme ratings (worst and best)
    if ratings:
        rating_nums = {}
        for country, rating in ratings.items():
            if rating in RATING_ORDER:
                rating_nums[country] = RATING_ORDER.index(rating)
        
        if rating_nums:
            rating_values = list(rating_nums.values())
            worst_idx = max(rating_values)
            best_idx = min(rating_values)
            
            for country, idx in rating_nums.items():
                if idx == worst_idx:
                    outliers['extreme_ratings'].append(f"{country} (worst: {ratings[country]})")
                elif idx == best_idx:
                    outliers['extreme_ratings'].append(f"{country} (best: {ratings[country]})")
    
    return outliers


def create_correlation_matrix(assessments: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create correlation matrix between features and outcomes."""
    data = []
    
    for country, assessment in assessments.items():
        overall = assessment.get('overall_assessment', {})
        raw_features = assessment.get('raw_features', {})
        
        row = {
            'Default Probability': overall.get('default_probability_5yr', np.nan),
            'Debt/GDP': raw_features.get('debt_gdp', np.nan),
            'Deficit/GDP': raw_features.get('deficit_gdp', np.nan),
            'Current Account/GDP': raw_features.get('current_account_gdp', np.nan),
            'Inflation (5yr)': raw_features.get('inflation_5yr', np.nan),
            'GDP per Capita': raw_features.get('gdp_per_capita', np.nan),
            'Unemployment': raw_features.get('unemployment', np.nan),
            'External Debt/GDP': raw_features.get('external_debt_gdp', np.nan),
            'Institutional Quality': raw_features.get('institutional_quality', np.nan),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Compute correlation matrix (only numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    return correlation_matrix


def generate_report(assessments: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate comprehensive statistics report."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("EUROZONE RISK STATISTICS REPORT")
    print("="*80)
    
    # 1. Basic Statistics
    print("\n[1] BASIC STATISTICS")
    print("-" * 80)
    basic_stats = compute_basic_statistics(assessments)
    
    print(f"\nTotal Countries Analyzed: {basic_stats['total_countries']}")
    print(f"Countries with Data: {basic_stats['countries_with_data']}")
    
    print("\nDefault Probability Statistics (%):")
    pd_stats = basic_stats['default_probability']
    print(f"  Mean:   {pd_stats['mean']:.2f}")
    print(f"  Median: {pd_stats['median']:.2f}")
    print(f"  Std:    {pd_stats['std']:.2f}")
    print(f"  Min:    {pd_stats['min']:.2f} ({min([c for c, a in assessments.items() if a.get('overall_assessment', {}).get('default_probability_5yr', np.nan) == pd_stats['min']])})")
    print(f"  Max:    {pd_stats['max']:.2f} ({min([c for c, a in assessments.items() if a.get('overall_assessment', {}).get('default_probability_5yr', np.nan) == pd_stats['max']])})")
    print(f"  Q25:    {pd_stats['q25']:.2f}")
    print(f"  Q75:    {pd_stats['q75']:.2f}")
    print(f"  CV:     {pd_stats['coefficient_of_variation']:.2f}")
    
    print("\nRating Distribution:")
    for rating in RATING_ORDER:
        count = basic_stats['rating_distribution'].get(rating, 0)
        if count > 0:
            print(f"  {rating}: {count} countries")
    
    # 2. Summary Table
    print("\n[2] COUNTRY SUMMARY TABLE")
    print("-" * 80)
    summary_df = create_summary_table(assessments)
    summary_df_sorted = summary_df.sort_values('Default Probability (%)', ascending=False)
    print("\nCountries ranked by Default Probability:")
    print(summary_df_sorted[['Country', 'Default Probability (%)', 'Rating']].to_string(index=False))
    
    # Save to CSV
    summary_csv = output_dir / "eurozone_summary_table.csv"
    summary_df_sorted.to_csv(summary_csv, index=False)
    print(f"\n  Saved to: {summary_csv}")
    
    # 3. Feature Contributions
    print("\n[3] FEATURE CONTRIBUTIONS ANALYSIS")
    print("-" * 80)
    feature_stats = analyze_feature_contributions(assessments)
    if not feature_stats.empty:
        print("\nTop Risk Drivers (by mean absolute contribution):")
        print(feature_stats.head(10).to_string(index=False))
        
        feature_csv = output_dir / "feature_contributions.csv"
        feature_stats.to_csv(feature_csv, index=False)
        print(f"\n  Saved to: {feature_csv}")
    
    # 4. Threshold Analysis
    print("\n[4] THRESHOLD STATUS ANALYSIS")
    print("-" * 80)
    threshold_analysis = analyze_threshold_status(assessments)
    
    for threshold_name, stats in threshold_analysis.items():
        if 'total' in stats and stats['total'] > 0:
            print(f"\n{threshold_name.upper()}:")
            print(f"  Safe:      {stats['safe']} ({stats['safe']/stats['total']*100:.1f}%)")
            print(f"  Warning:   {stats['warning']} ({stats['warning']/stats['total']*100:.1f}%)")
            print(f"  Critical:  {stats['critical']} ({stats['critical']/stats['total']*100:.1f}%)")
            if 'mean_value' in stats:
                print(f"  Mean Value: {stats['mean_value']:.2f}")
    
    # 5. Outliers
    print("\n[5] OUTLIER ANALYSIS")
    print("-" * 80)
    outliers = identify_outliers(assessments)
    
    if outliers['high_pd']:
        print(f"\nHigh Default Probability Outliers (>2 std): {', '.join(outliers['high_pd'])}")
    if outliers['low_pd']:
        print(f"Low Default Probability Outliers (<2 std): {', '.join(outliers['low_pd'])}")
    if outliers['high_spread']:
        print(f"\nHigh Spread Outliers (>2 std): {', '.join(outliers['high_spread'])}")
    if outliers['low_spread']:
        print(f"Low Spread Outliers (<2 std): {', '.join(outliers['low_spread'])}")
    if outliers['extreme_ratings']:
        print(f"\nExtreme Ratings: {', '.join(outliers['extreme_ratings'])}")
    
    # 6. Correlation Analysis
    print("\n[6] CORRELATION ANALYSIS")
    print("-" * 80)
    corr_matrix = create_correlation_matrix(assessments)
    
    print("\nCorrelation with Default Probability:")
    if 'Default Probability' in corr_matrix.index:
        pd_corr = corr_matrix['Default Probability'].sort_values(key=lambda x: x.abs(), ascending=False)
        for feature, corr_val in pd_corr.items():
            if feature != 'Default Probability' and not pd.isna(corr_val):
                print(f"  {feature}: {corr_val:.3f}")
    
    # Save correlation matrix
    corr_csv = output_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv)
    print(f"\n  Correlation matrix saved to: {corr_csv}")
    
    # 7. Model Behavior Insights
    print("\n[7] MODEL BEHAVIOR INSIGHTS")
    print("-" * 80)
    
    # Analyze relationship between debt and PD
    debt_pd_data = []
    for country, assessment in assessments.items():
        raw_features = assessment.get('raw_features', {})
        overall = assessment.get('overall_assessment', {})
        debt = pd.to_numeric(raw_features.get('debt_gdp'), errors='coerce')
        pd_val = pd.to_numeric(overall.get('default_probability_5yr'), errors='coerce')
        if not pd.isna(debt) and not pd.isna(pd_val):
            debt_pd_data.append({'debt': debt, 'pd': pd_val, 'country': country})
    
    if len(debt_pd_data) >= 3:
        debt_pd_df = pd.DataFrame(debt_pd_data)
        debt_pd_corr = debt_pd_df['debt'].corr(debt_pd_df['pd'])
        print(f"\nDebt/GDP vs Default Probability correlation: {debt_pd_corr:.3f}")
        
        if debt_pd_corr > 0.5:
            print("  -> Strong positive relationship: Higher debt associated with higher default risk")
        elif debt_pd_corr < -0.5:
            print("  -> Strong negative relationship: Higher debt associated with lower default risk (unexpected!)")
        else:
            print("  -> Weak relationship: Debt alone doesn't strongly predict default probability")
    
    # Analyze institutional quality impact
    inst_quality_data = []
    for country, assessment in assessments.items():
        raw_features = assessment.get('raw_features', {})
        overall = assessment.get('overall_assessment', {})
        inst_qual = pd.to_numeric(raw_features.get('institutional_quality'), errors='coerce')
        pd_val = pd.to_numeric(overall.get('default_probability_5yr'), errors='coerce')
        if not pd.isna(inst_qual) and not pd.isna(pd_val):
            inst_quality_data.append({'inst_qual': inst_qual, 'pd': pd_val, 'country': country})
    
    if len(inst_quality_data) >= 3:
        inst_df = pd.DataFrame(inst_quality_data)
        inst_corr = inst_df['inst_qual'].corr(inst_df['pd'])
        print(f"\nInstitutional Quality vs Default Probability correlation: {inst_corr:.3f}")
        
        if inst_corr < -0.5:
            print("  -> Strong negative relationship: Better institutions associated with lower default risk")
        elif inst_corr > 0.5:
            print("  -> Strong positive relationship: Better institutions associated with higher default risk (unexpected!)")
        else:
            print("  -> Weak relationship: Institutional quality alone doesn't strongly predict default probability")
    
    # 8. Generate JSON report
    print("\n[8] GENERATING JSON REPORT")
    print("-" * 80)
    
    report = {
        'metadata': {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_countries': basic_stats['total_countries'],
            'countries_analyzed': basic_stats['countries_with_data']
        },
        'basic_statistics': basic_stats,
        'threshold_analysis': threshold_analysis,
        'outliers': outliers,
        'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
        'country_rankings': {
            'by_pd': summary_df_sorted[['Country', 'Default Probability (%)', 'Rating']].to_dict('records')
        }
    }
    
    report_json = output_dir / "eurozone_statistics_report.json"
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  Comprehensive report saved to: {report_json}")
    
    print("\n" + "="*80)
    print("STATISTICS ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


def main():
    """Main execution function."""
    print("="*80)
    print("EUROZONE RISK STATISTICS ANALYZER")
    print("="*80)
    
    # Load all assessments
    print("\nLoading assessments...")
    assessments = load_all_assessments()
    
    if not assessments:
        print("ERROR: No assessments found. Please run risk_calculator_2.py first.")
        return
    
    print(f"Loaded {len(assessments)} country assessments")
    
    # Generate report
    generate_report(assessments, OUTPUT_DIR)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

