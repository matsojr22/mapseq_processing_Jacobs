#!/usr/bin/env python3
"""
Generate conclusions markdown from extracted stability data.

Reads extracted_stability_data.json and generates a comprehensive
markdown document with findings about temporal stability.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def format_p_value(pval):
    """Format p-value for display."""
    if pval is None:
        return "N/A"
    if pval < 0.001:
        return f"{pval:.2e}***"
    elif pval < 0.01:
        return f"{pval:.4f}**"
    elif pval < 0.05:
        return f"{pval:.4f}*"
    else:
        return f"{pval:.4f}"

def generate_kruskal_wallis_section(data, model_type):
    """Generate section for Kruskal-Wallis results."""
    kw_data = data.get('kruskal_wallis', {})
    results = kw_data.get('results', [])
    summary = kw_data.get('summary', {})
    
    if not results:
        return f"### Kruskal-Wallis Test Results ({model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Kruskal-Wallis Test Results ({model_type} model)\n\n"
    
    # Summary statistics
    if summary:
        md += "#### Summary Statistics\n\n"
        md += f"- **Total motifs tested**: {summary.get('total_motifs', 0)}\n"
        md += f"- **Significant motifs (p < 0.05)**: {summary.get('significant_count', 0)} ({summary.get('significant_count', 0) / summary.get('total_motifs', 1) * 100:.1f}%)\n"
        md += f"- **Non-significant motifs (p ≥ 0.05)**: {summary.get('non_significant_count', 0)} ({summary.get('non_significant_count', 0) / summary.get('total_motifs', 1) * 100:.1f}%)\n"
        md += f"- **Mean p-value**: {summary.get('mean_p_value', 0):.4f}\n"
        md += f"- **Median p-value**: {summary.get('median_p_value', 0):.4f}\n"
        md += f"- **Min p-value**: {format_p_value(summary.get('min_p_value'))}\n"
        md += f"- **Max p-value**: {format_p_value(summary.get('max_p_value'))}\n\n"
    
    # Top significant motifs
    significant = [r for r in results if r.get('significant', False)]
    if significant:
        significant.sort(key=lambda x: x.get('p_value', 1.0))
        md += "#### Most Significant Motifs (Top 20)\n\n"
        md += "| Motif | H-Statistic | P-Value | Significant |\n"
        md += "|-------|-------------|---------|-------------|\n"
        for r in significant[:20]:
            motif = r.get('motif', 'N/A')
            hstat = r.get('h_statistic', 'N/A')
            pval = format_p_value(r.get('p_value'))
            sig = "Yes" if r.get('significant') else "No"
            md += f"| {motif} | {hstat} | {pval} | {sig} |\n"
        md += "\n"
    
    # Interpretation
    if summary:
        sig_pct = (summary.get('significant_count', 0) / summary.get('total_motifs', 1)) * 100
        if sig_pct > 50:
            interpretation = "**UNSTABLE**: More than 50% of motifs show significant changes over time."
        elif sig_pct > 25:
            interpretation = "**MODERATELY UNSTABLE**: 25-50% of motifs show significant changes over time."
        elif sig_pct > 10:
            interpretation = "**MODERATELY STABLE**: 10-25% of motifs show significant changes over time."
        else:
            interpretation = "**STABLE**: Less than 10% of motifs show significant changes over time."
        
        md += f"#### Interpretation\n\n{interpretation}\n\n"
    
    return md

def generate_transition_significance_section(data, model_type):
    """Generate section for transition significance results."""
    trans_data = data.get('transition_significance', {})
    transitions = trans_data.get('transitions', [])
    summary = trans_data.get('summary', {})
    
    if not transitions:
        return f"### Transition Significance (Script 07, {model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Transition Significance (Script 07, {model_type} model)\n\n"
    
    # Summary by transition
    if summary:
        md += "#### Summary by Transition\n\n"
        md += "| Transition | Total Motifs | Significant | Non-Significant | % Significant |\n"
        md += "|-----------|--------------|------------|-----------------|---------------|\n"
        for trans_name, trans_summary in summary.items():
            total = trans_summary.get('total_motifs', 0)
            sig = trans_summary.get('significant_count', 0)
            non_sig = trans_summary.get('non_significant_count', 0)
            pct = trans_summary.get('significant_percentage', 0)
            md += f"| {trans_name} | {total} | {sig} | {non_sig} | {pct:.1f}% |\n"
        md += "\n"
    
    # Most significant transitions
    significant_trans = [t for t in transitions if t.get('significant', False)]
    if significant_trans:
        significant_trans.sort(key=lambda x: x.get('p_value', 1.0))
        md += "#### Most Significant Transitions (Top 20)\n\n"
        md += "| Motif | Transition | P-Value | Significant |\n"
        md += "|-------|------------|---------|-------------|\n"
        for t in significant_trans[:20]:
            motif = t.get('motif', 'N/A')
            transition = t.get('transition', 'N/A')
            pval = format_p_value(t.get('p_value'))
            sig = "Yes" if t.get('significant') else "No"
            md += f"| {motif} | {transition} | {pval} | {sig} |\n"
        md += "\n"
    
    return md

def generate_effect_sizes_section(data, model_type):
    """Generate section for effect size trajectories."""
    es_data = data.get('effect_sizes', {})
    trajectories = es_data.get('trajectories', [])
    summary = es_data.get('summary', {})
    
    if not trajectories:
        return f"### Effect Size Trajectories ({model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Effect Size Trajectories ({model_type} model)\n\n"
    
    # Summary statistics
    if summary:
        md += "#### Summary Statistics\n\n"
        md += f"- **Total motifs**: {summary.get('total_motifs', 0)}\n"
        md += f"- **Mean effect size range**: {summary.get('mean_range', 0):.3f}\n"
        md += f"- **Maximum effect size range**: {summary.get('max_range', 0):.3f}\n"
        md += f"- **Mean absolute effect size**: {summary.get('mean_abs_effect_size', 0):.3f}\n"
        md += f"- **Motifs with large changes (range > 1.0)**: {summary.get('motifs_with_large_changes', 0)}\n\n"
    
    # Motifs with largest changes
    trajectories_with_range = [t for t in trajectories if 'range_effect_size' in t]
    if trajectories_with_range:
        trajectories_with_range.sort(key=lambda x: x.get('range_effect_size', 0), reverse=True)
        md += "#### Motifs with Largest Effect Size Changes (Top 20)\n\n"
        md += "| Motif | Min ES | Max ES | Range | Mean ES | Trend |\n"
        md += "|-------|--------|--------|-------|---------|-------|\n"
        for t in trajectories_with_range[:20]:
            motif = t.get('motif', 'N/A')
            min_es = t.get('min_effect_size', 0)
            max_es = t.get('max_effect_size', 0)
            range_es = t.get('range_effect_size', 0)
            mean_es = t.get('mean_effect_size', 0)
            trend = t.get('trend', 'N/A')
            md += f"| {motif} | {min_es:.3f} | {max_es:.3f} | {range_es:.3f} | {mean_es:.3f} | {trend} |\n"
        md += "\n"
    
    return md

def generate_motif_percentages_section(data, model_type):
    """Generate section for motif percentage matrix."""
    mp_data = data.get('motif_percentages', {})
    matrix = mp_data.get('matrix')
    summary = mp_data.get('summary', {})
    
    if not matrix:
        return f"### Motif Percentages by Age ({model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Motif Percentages by Age ({model_type} model)\n\n"
    
    if summary:
        md += "#### Summary Statistics\n\n"
        ages = summary.get('ages', [])
        md += "| Age | Mean Percentage | Std Deviation |\n"
        md += "|-----|----------------|---------------|\n"
        for age in ages:
            mean_pct = summary.get('mean_percentages_by_age', {}).get(age, 0)
            std_pct = summary.get('std_percentages_by_age', {}).get(age, 0)
            md += f"| {age.upper()} | {mean_pct:.2f}% | {std_pct:.2f}% |\n"
        md += "\n"
    
    return md

def generate_transition_summary_section(data, model_type):
    """Generate section for transition summary from script 05."""
    ts_data = data.get('transition_summary', {})
    transitions = ts_data.get('transitions', [])
    summary = ts_data.get('summary', {})
    
    if not transitions:
        return f"### Transition Summary (Script 05, {model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Transition Summary (Script 05, {model_type} model)\n\n"
    
    if summary:
        md += "#### Summary by Transition\n\n"
        md += "| Transition | Total Motifs | Significant | Mean JSD | Max JSD | % Significant |\n"
        md += "|-----------|--------------|------------|----------|---------|---------------|\n"
        for trans_name, trans_summary in summary.items():
            total = trans_summary.get('total_motifs', 0)
            sig = trans_summary.get('significant_count', 0)
            mean_jsd = trans_summary.get('mean_jsd', 0)
            max_jsd = trans_summary.get('max_jsd', 0)
            pct = trans_summary.get('significant_percentage', 0)
            md += f"| {trans_name} | {total} | {sig} | {mean_jsd:.4f} | {max_jsd:.4f} | {pct:.1f}% |\n"
        md += "\n"
    
    # Motifs with highest JSD
    transitions_with_jsd = [t for t in transitions if t.get('js_divergence') is not None]
    if transitions_with_jsd:
        transitions_with_jsd.sort(key=lambda x: x.get('js_divergence', 0), reverse=True)
        md += "#### Motifs with Highest JSD (Top 20)\n\n"
        md += "| Motif | Transition | JSD | Significant |\n"
        md += "|-------|------------|-----|-------------|\n"
        for t in transitions_with_jsd[:20]:
            motif = t.get('motif', 'N/A')
            transition = t.get('transition', 'N/A')
            jsd = t.get('js_divergence', 0)
            sig = "Yes" if t.get('significant') else "No"
            md += f"| {motif} | {transition} | {jsd:.4f} | {sig} |\n"
        md += "\n"
    
    return md

def generate_upsetplot_section(data, model_type):
    """Generate section for upsetplot data by age."""
    up_data = data.get('upsetplot_data', {})
    ages_data = up_data.get('ages', {})
    
    if not ages_data:
        return f"### Upsetplot Data by Age ({model_type} model)\n\n*No data available.*\n\n"
    
    md = f"### Upsetplot Data by Age ({model_type} model)\n\n"
    
    for age, age_info in ages_data.items():
        summary = age_info.get('summary', {})
        if summary:
            md += f"#### {age.upper()}\n\n"
            md += f"- **Total motifs**: {summary.get('total_motifs', 0)}\n"
            md += f"- **Significant motifs**: {summary.get('significant_count', 0)}\n"
            md += f"- **Mean effect size**: {summary.get('mean_effect_size', 0):.3f}\n"
            md += f"- **Mean absolute effect size**: {summary.get('mean_abs_effect_size', 0):.3f}\n"
            if summary.get('mean_p_value') is not None:
                md += f"- **Mean p-value**: {format_p_value(summary.get('mean_p_value'))}\n"
            md += "\n"
    
    return md

def generate_pie_chart_section(data):
    """Generate section for pie chart data (number of targets per cell)."""
    pie_data = data.get('pie_chart_data', {})
    ages_data = pie_data.get('ages', {})
    summary = pie_data.get('summary', {})
    
    if not ages_data:
        return "### Number of Targets Per Cell (Pie Chart Data)\n\n*No data available.*\n\n"
    
    md = "### Number of Targets Per Cell (Pie Chart Data)\n\n"
    md += "This analysis shows the distribution of cells by the number of target regions they project to, expressed as percentages of total cells at each age. This provides direct evidence of stability in projection complexity across development.\n\n"
    
    # Table showing percentages for each age
    md += "#### Distribution by Age (Percentages)\n\n"
    md += "*Note: Percentages sum to 100% for each age, representing the proportion of total cells at that developmental stage.*\n\n"
    
    # Get all target types
    all_target_types = summary.get('target_types', [])
    if not all_target_types:
        # Fallback: get from first age
        if ages_data:
            first_age = list(ages_data.keys())[0]
            all_target_types = sorted(ages_data[first_age].get('percentages', {}).keys(), 
                                     key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 0)
    
    # Create header
    md += "| Target Count | "
    for age in sorted(ages_data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        md += f"{age.upper()} | "
    md += "Mean | Std Dev |\n"
    
    # Create separator
    md += "|" + "|".join(["---"] * (len(ages_data) + 3)) + "|\n"
    
    # Add rows for each target type
    mean_percentages = summary.get('mean_percentages', {})
    std_percentages = summary.get('std_percentages', {})
    
    for target_type in all_target_types:
        md += f"| {target_type} | "
        for age in sorted(ages_data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
            pct = ages_data[age].get('percentages', {}).get(target_type, 0.0)
            md += f"{pct:.2f}% | "
        mean_pct = mean_percentages.get(target_type, 0.0)
        std_pct = std_percentages.get(target_type, 0.0)
        md += f"{mean_pct:.2f}% | {std_pct:.2f}% |\n"
    
    md += "\n"
    
    # Summary statistics
    if summary:
        md += "#### Summary Statistics\n\n"
        md += "- **Mean percentages across ages** (showing stability of distribution):\n"
        for target_type, mean_pct in sorted(mean_percentages.items(), 
                                            key=lambda x: int(x[0].split()[0]) if x[0].split()[0].isdigit() else 0):
            std_pct = std_percentages.get(target_type, 0.0)
            md += f"  - {target_type}: {mean_pct:.2f}% ± {std_pct:.2f}%\n"
        md += "\n"
    
    # Interpretation
    md += "#### Interpretation\n\n"
    md += "The pie chart data shows the percentage of cells projecting to 1, 2, 3, 4, or more target regions at each developmental stage. "
    md += "Low standard deviations across ages for each target count category indicate stability in the distribution of projection complexity. "
    md += "This provides direct, model-independent evidence that the overall pattern of projection complexity (number of targets per cell) remains stable across development.\n\n"
    
    return md

def generate_overall_summary(data):
    """Generate overall summary section."""
    md = "## Overall Summary\n\n"
    
    for model_type, model_data in data.get('models', {}).items():
        md += f"### {model_type.upper()} Model\n\n"
        
        # Kruskal-Wallis summary
        kw_summary = model_data.get('kruskal_wallis', {}).get('summary', {})
        if kw_summary:
            total = kw_summary.get('total_motifs', 0)
            sig = kw_summary.get('significant_count', 0)
            sig_pct = (sig / total * 100) if total > 0 else 0
            md += f"- **Kruskal-Wallis**: {sig}/{total} motifs ({sig_pct:.1f}%) show significant changes over time\n"
        
        # Transition significance summary
        trans_summary = model_data.get('transition_significance', {}).get('summary', {})
        if trans_summary:
            total_trans = sum(s.get('total_motifs', 0) for s in trans_summary.values())
            sig_trans = sum(s.get('significant_count', 0) for s in trans_summary.values())
            if total_trans > 0:
                md += f"- **Transition Significance**: {sig_trans}/{total_trans} transitions ({sig_trans/total_trans*100:.1f}%) are significant\n"
        
        # Effect size summary
        es_summary = model_data.get('effect_sizes', {}).get('summary', {})
        if es_summary:
            large_changes = es_summary.get('motifs_with_large_changes', 0)
            total = es_summary.get('total_motifs', 0)
            if total > 0:
                md += f"- **Effect Size Changes**: {large_changes}/{total} motifs ({large_changes/total*100:.1f}%) show large changes (range > 1.0)\n"
        
        md += "\n"
    
    # Model comparison
    if len(data.get('models', {})) == 2:
        md += "### Model Comparison\n\n"
        md += "Compare uniform and region-specific models to assess robustness of findings.\n\n"
    
    return md

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate conclusions markdown from extracted data")
    parser.add_argument('--input', type=str, default='extracted_stability_data.json',
                       help='Input JSON file (default: extracted_stability_data.json)')
    parser.add_argument('--output', type=str, default='../stability_analysis_conclusions.md',
                       help='Output markdown file (default: ../stability_analysis_conclusions.md)')
    args = parser.parse_args()
    
    # Load extracted data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Generate markdown
    md = f"""# Stability Analysis Conclusions

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Parameterization**: {data.get('parameterization', 'N/A')}
**Base Directory**: {data.get('base_dir', 'N/A')}

## Introduction

This document presents findings from the stability analysis of MAPseq projection data across developmental timepoints (P3, P12, P20, P60). The central question addressed is: **Do results change significantly over time?**

The analysis uses two statistical models:
- **Uniform Model**: Single edge probability (pₑ) for all regions
- **Region-Specific Model**: Region-specific projection probabilities

---

"""
    
    # Overall summary
    md += generate_overall_summary(data)
    md += "\n---\n\n"
    
    # Detailed results for each model
    for model_type, model_data in data.get('models', {}).items():
        md += f"## Detailed Results: {model_type.upper()} Model\n\n"
        
        md += generate_kruskal_wallis_section(model_data, model_type)
        md += "\n"
        
        md += generate_transition_significance_section(model_data, model_type)
        md += "\n"
        
        md += generate_effect_sizes_section(model_data, model_type)
        md += "\n"
        
        md += generate_motif_percentages_section(model_data, model_type)
        md += "\n"
        
        md += generate_transition_summary_section(model_data, model_type)
        md += "\n"
        
        md += generate_upsetplot_section(model_data, model_type)
        md += "\n"
        
        md += "---\n\n"
    
    # Pie chart data (model-independent, shared across models)
    md += "## Model-Independent Analysis: Number of Targets Per Cell\n\n"
    md += generate_pie_chart_section(data)
    md += "\n---\n\n"
    
    # Interpretation section
    md += """## Interpretation Guidelines

### Stability Assessment

- **STABLE**: < 10% of motifs show significant changes (Kruskal-Wallis p < 0.05)
- **MODERATELY STABLE**: 10-25% of motifs show significant changes
- **MODERATELY UNSTABLE**: 25-50% of motifs show significant changes
- **UNSTABLE**: > 50% of motifs show significant changes

### Effect Size Interpretation

- **Small changes**: |Effect Size| < 0.5
- **Moderate changes**: |Effect Size| 0.5-1.0
- **Large changes**: |Effect Size| > 1.0

### JSD Interpretation

- **Very stable**: JSD < 0.05
- **Moderately stable**: JSD 0.05-0.1
- **Moderately unstable**: JSD 0.1-0.2
- **Unstable**: JSD > 0.2

### Transition Significance

- **Significant transition**: Fisher's exact test p < 0.05
- Indicates when changes occur (P3→P12, P12→P20, P20→P60)

---

## Key Findings

### Primary Evidence

1. **Kruskal-Wallis Test Results**: Overall test of temporal stability
2. **Transition Significance**: Identifies which stage transitions show changes
3. **Effect Size Trajectories**: Quantifies magnitude of changes

### Secondary Evidence

1. **Motif Percentage Changes**: Quantitative percentage changes across ages
2. **JSD Values**: Distribution similarity metrics
3. **Upsetplot Data**: Per-age motif representation

---

## Notes

- P-values marked with: * (p < 0.05), ** (p < 0.01), *** (p < 0.001)
- Both uniform and region-specific models should be compared for robustness
- Significant results indicate temporal instability; non-significant results indicate stability

"""
    
    # Save markdown
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"✅ Conclusions document generated!")
    print(f"📁 Saved to: {output_path}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
