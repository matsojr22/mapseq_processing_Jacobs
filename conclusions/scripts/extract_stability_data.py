#!/usr/bin/env python3
"""
Extract stability analysis data from pipeline outputs.

This script searches for and extracts:
1. Kruskal-Wallis results from script 01
2. Transition significance from script 07
3. Motif frequency matrices from script 05
4. Divergence metrics from script 06
5. Aggregate upsetplot data from each age group

Outputs structured data for conclusions document generation.
"""

import os
import sys
import json
import glob
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def find_files_by_pattern(base_dir, pattern, recursive=True):
    """Find files matching pattern in base directory."""
    if recursive:
        pattern = str(Path(base_dir) / "**" / pattern)
    else:
        pattern = str(Path(base_dir) / pattern)
    return glob.glob(pattern, recursive=recursive)

def extract_kruskal_wallis_results(helper_dir, parameterization, model_type):
    """Extract Kruskal-Wallis test results from script 01 outputs."""
    results = {
        'model': model_type,
        'results': [],
        'summary': {}
    }
    
    # Look for Kruskal-Wallis results files
    script01_dir = Path(helper_dir) / f"{parameterization}_helpers" / "01_motif_analysis_per_animal" / model_type
    
    print(f"    Searching in: {script01_dir}")
    print(f"    Directory exists: {script01_dir.exists()}")
    
    # Try to find CSV or text files with Kruskal-Wallis results (recursive search)
    patterns = [
        "*kruskal*.csv",
        "*kruskal*.txt",
        "*KW*.csv",
        "*KW*.txt",
    ]
    
    kw_files = []
    for pattern in patterns:
        files = find_files_by_pattern(script01_dir, pattern, recursive=True)
        kw_files.extend(files)
        if kw_files:
            print(f"    Found {len(files)} files matching pattern: {pattern}")
            break
    
    # Also check for results in any CSV files (including cross_age subdirectory)
    if not kw_files:
        print(f"    No files found with kruskal patterns, searching all CSVs...")
        all_csvs = find_files_by_pattern(script01_dir, "*.csv", recursive=True)
        print(f"    Found {len(all_csvs)} total CSV files")
        for csv_file in all_csvs:
            try:
                df = pd.read_csv(csv_file)
                # Check if it looks like Kruskal-Wallis results
                if any(col.lower() in ['p_value', 'p-value', 'pvalue', 'h_statistic', 'h-statistic'] for col in df.columns):
                    kw_files.append(csv_file)
                    print(f"    Identified Kruskal-Wallis file: {csv_file}")
            except Exception as e:
                print(f"    Warning: Could not read {csv_file}: {e}")
    
    if not kw_files:
        print(f"    ERROR: No Kruskal-Wallis files found in {script01_dir}")
        return results
    
    # Parse results - prefer global normalization file
    preferred_file = None
    for f in kw_files:
        if 'global' in str(f).lower():
            preferred_file = f
            break
    
    if not preferred_file:
        preferred_file = kw_files[0]
    
    print(f"    Using file: {preferred_file}")
    
    try:
        df = pd.read_csv(preferred_file)
        print(f"    Loaded {len(df)} rows from CSV")
        
        # Look for motif and p-value columns
        motif_col = None
        pval_col = None
        hstat_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'motif' in col_lower:
                motif_col = col
            if 'p_value' in col_lower or 'p-value' in col_lower or 'pvalue' in col_lower:
                pval_col = col
            if 'h_statistic' in col_lower or 'h-statistic' in col_lower or 'hstat' in col_lower:
                hstat_col = col
        
        print(f"    Found columns: motif={motif_col}, pval={pval_col}, hstat={hstat_col}")
        
        if motif_col and pval_col:
            for _, row in df.iterrows():
                motif = str(row[motif_col]).strip()
                # Skip empty/NaN motifs
                if not motif or motif.lower() in ['nan', 'none', '']:
                    continue
                
                pval = float(row[pval_col]) if pd.notna(row[pval_col]) else None
                hstat = float(row[hstat_col]) if hstat_col and pd.notna(row[hstat_col]) else None
                
                results['results'].append({
                    'motif': motif,
                    'p_value': pval,
                    'h_statistic': hstat,
                    'significant': pval < 0.05 if pval is not None else None
                })
            
            print(f"    Extracted {len(results['results'])} valid motifs")
        else:
            print(f"    ERROR: Missing required columns. Found: {list(df.columns)}")
    except Exception as e:
        print(f"    ERROR: Could not parse {preferred_file}: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate summary statistics
    if results['results']:
        pvals = [r['p_value'] for r in results['results'] if r['p_value'] is not None]
        if pvals:
            results['summary'] = {
                'total_motifs': len(results['results']),
                'significant_count': sum(1 for r in results['results'] if r.get('significant', False)),
                'non_significant_count': sum(1 for r in results['results'] if not r.get('significant', True)),
                'min_p_value': float(min(pvals)),
                'max_p_value': float(max(pvals)),
                'mean_p_value': float(np.mean(pvals)),
                'median_p_value': float(np.median(pvals))
            }
            print(f"    Summary: {results['summary']['total_motifs']} motifs, {results['summary']['significant_count']} significant")
    
    return results

def extract_transition_significance(helper_dir, parameterization, model_type):
    """Extract transition significance from script 07 outputs."""
    results = {
        'model': model_type,
        'transitions': [],
        'summary': {}
    }
    
    script07_dir = Path(helper_dir) / f"{parameterization}_helpers" / "07_motif_significange_trajectories" / model_type
    
    print(f"    Searching in: {script07_dir}")
    print(f"    Directory exists: {script07_dir.exists()}")
    
    # Look for transition_significance.csv
    trans_file = script07_dir / "transition_significance.csv"
    
    if not trans_file.exists():
        print(f"    ERROR: File not found: {trans_file}")
        return results
    
    print(f"    Found file: {trans_file}")
    
    try:
        df = pd.read_csv(trans_file)
        print(f"    Loaded {len(df)} rows from CSV")
        print(f"    Columns: {list(df.columns)}")
        
        # Handle case where df might be a Series (single row)
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        
        # Find p-value column (case-insensitive)
        pval_col = None
        for col in df.columns:
            if col.lower() in ['p-value', 'p_value', 'pvalue']:
                pval_col = col
                break
        
        sig_col = None
        for col in df.columns:
            if col.lower() == 'significant':
                sig_col = col
                break
        
        motif_col = None
        for col in df.columns:
            if col.lower() == 'motif':
                motif_col = col
                break
        
        trans_col = None
        for col in df.columns:
            if col.lower() == 'transition':
                trans_col = col
                break
        
        print(f"    Found columns: motif={motif_col}, transition={trans_col}, pval={pval_col}, sig={sig_col}")
        
        if not motif_col or not trans_col:
            print(f"    ERROR: Missing required columns (Motif or Transition)")
            return results
        
        for _, row in df.iterrows():
            motif = str(row.get(motif_col, '')).strip()
            # Skip empty/NaN motifs
            if not motif or motif.lower() in ['nan', 'none', '']:
                continue
            
            transition = str(row.get(trans_col, ''))
            pval = float(row[pval_col]) if pval_col and pd.notna(row.get(pval_col)) else None
            significant = bool(row[sig_col]) if sig_col and pd.notna(row.get(sig_col)) else (pval < 0.05 if pval is not None else False)
            
            results['transitions'].append({
                'motif': motif,
                'transition': transition,
                'p_value': pval,
                'significant': significant
            })
        
        print(f"    Extracted {len(results['transitions'])} valid transitions")
    except Exception as e:
        print(f"    ERROR: Could not parse {trans_file}: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate summary by transition
    if results['transitions']:
        transitions = defaultdict(list)
        for trans in results['transitions']:
            transitions[trans['transition']].append(trans)
        
        results['summary'] = {}
        for trans_name, trans_list in transitions.items():
            significant_count = sum(1 for t in trans_list if t['significant'])
            results['summary'][trans_name] = {
                'total_motifs': len(trans_list),
                'significant_count': significant_count,
                'non_significant_count': len(trans_list) - significant_count,
                'significant_percentage': (significant_count / len(trans_list) * 100) if trans_list else 0
            }
        print(f"    Summary: {len(results['summary'])} transitions, {sum(r['significant_count'] for r in results['summary'].values())} significant")
    
    return results

def extract_effect_sizes(helper_dir, parameterization, model_type):
    """Extract effect size trajectories from script 07 outputs."""
    results = {
        'model': model_type,
        'trajectories': [],
        'summary': {}
    }
    
    script07_dir = Path(helper_dir) / f"{parameterization}_helpers" / "07_motif_significange_trajectories" / model_type
    
    print(f"    Searching in: {script07_dir}")
    print(f"    Directory exists: {script07_dir.exists()}")
    
    # Look for combined_effect_sizes CSV
    effect_file = script07_dir / f"combined_effect_sizes_{model_type}.csv"
    
    if not effect_file.exists():
        effect_file = script07_dir / "combined_effect_sizes.csv"
    
    if not effect_file.exists():
        print(f"    ERROR: File not found: {effect_file}")
        return results
    
    print(f"    Found file: {effect_file}")
    
    try:
        df = pd.read_csv(effect_file)
        print(f"    Loaded {len(df)} rows from CSV")
        
        # Debug: Check what we got
        if not isinstance(df, pd.DataFrame):
            print(f"    ERROR: {effect_file} did not return a DataFrame, got {type(df)}")
            return results
        
        # Check if we have a valid DataFrame with data
        if len(df) == 0 or df.empty:
            print(f"    ERROR: {effect_file} is empty")
            return results
        
        print(f"    Columns: {list(df.columns)}")
        
        # Find motif label column (case-insensitive)
        motif_col = None
        for col in df.columns:
            if col.lower() in ['motif_label', 'motif', 'motif-label']:
                motif_col = col
                break
        
        if not motif_col:
            print(f"    ERROR: Motif_Label column not found, columns: {list(df.columns)}")
            return results
        
        # Filter out NaN/empty motifs
        df = df[df[motif_col].notna() & (df[motif_col] != '')]
        if len(df) == 0:
            print(f"    ERROR: No valid motifs after filtering NaN/empty")
            return results
            
        motifs = df[motif_col].unique()
        print(f"    Found {len(motifs)} unique motifs")
        
        # Find effect size and stage columns
        es_col = None
        for col in df.columns:
            if col.lower() in ['effect size', 'effect_size', 'effect-size']:
                es_col = col
                break
        
        stage_col = None
        for col in df.columns:
            if col.lower() == 'stage':
                stage_col = col
                break
        
        obs_col = None
        for col in df.columns:
            if col.lower() == 'observed':
                obs_col = col
                break
        
        sig_col = None
        for col in df.columns:
            if col.lower() == 'significant':
                sig_col = col
                break
        
        if not es_col:
            print(f"    ERROR: Effect Size column not found, columns: {list(df.columns)}")
            return results
        
        print(f"    Using columns: motif={motif_col}, effect_size={es_col}, stage={stage_col}, observed={obs_col}, significant={sig_col}")
        
        for motif in motifs:
            motif_str = str(motif).strip()
            if not motif_str or motif_str.lower() in ['nan', 'none', '']:
                continue
            
            motif_data = df[df[motif_col] == motif]
            if stage_col and stage_col in motif_data.columns:
                motif_data = motif_data.sort_values(stage_col)
            
            trajectory = {
                'motif': motif_str,
                'stages': [],
                'effect_sizes': [],
                'observed': [],
                'significant': []
            }
            
            for _, row in motif_data.iterrows():
                if stage_col:
                    trajectory['stages'].append(str(row.get(stage_col, '')))
                trajectory['effect_sizes'].append(float(row.get(es_col, 0)) if pd.notna(row.get(es_col)) else 0)
                if obs_col:
                    trajectory['observed'].append(int(row.get(obs_col, 0)) if pd.notna(row.get(obs_col)) else 0)
                if sig_col:
                    trajectory['significant'].append(bool(row.get(sig_col, False)) if pd.notna(row.get(sig_col)) else False)
            
            # Calculate trajectory metrics
            if len(trajectory['effect_sizes']) > 1:
                es_array = np.array(trajectory['effect_sizes'])
                trajectory['mean_effect_size'] = float(np.mean(es_array))
                trajectory['std_effect_size'] = float(np.std(es_array))
                trajectory['min_effect_size'] = float(np.min(es_array))
                trajectory['max_effect_size'] = float(np.max(es_array))
                trajectory['range_effect_size'] = float(np.max(es_array) - np.min(es_array))
                trajectory['trend'] = 'increasing' if es_array[-1] > es_array[0] else 'decreasing' if es_array[-1] < es_array[0] else 'stable'
            
            results['trajectories'].append(trajectory)
        
        print(f"    Extracted {len(results['trajectories'])} valid trajectories")
    except Exception as e:
        print(f"    ERROR: Could not parse {effect_file}: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate summary statistics
    if results['trajectories']:
        all_ranges = [t.get('range_effect_size', 0) for t in results['trajectories'] if 'range_effect_size' in t]
        all_means = [abs(t.get('mean_effect_size', 0)) for t in results['trajectories'] if 'mean_effect_size' in t]
        
        results['summary'] = {
            'total_motifs': len(results['trajectories']),
            'mean_range': float(np.mean(all_ranges)) if all_ranges else 0,
            'max_range': float(np.max(all_ranges)) if all_ranges else 0,
            'mean_abs_effect_size': float(np.mean(all_means)) if all_means else 0,
            'motifs_with_large_changes': sum(1 for r in all_ranges if r > 1.0) if all_ranges else 0
        }
        print(f"    Summary: {results['summary']['total_motifs']} motifs, {results['summary']['motifs_with_large_changes']} with large changes")
    
    return results

def extract_motif_percentages(helper_dir, parameterization, model_type):
    """Extract motif percentage matrix from script 05 outputs."""
    results = {
        'model': model_type,
        'matrix': None,
        'summary': {}
    }
    
    script05_dir = Path(helper_dir) / f"{parameterization}_helpers" / "05_motif_analysis"
    
    print(f"    Searching in: {script05_dir}")
    print(f"    Directory exists: {script05_dir.exists()}")
    
    # Look for motif_percent_matrix_by_age CSV
    matrix_file = script05_dir / f"motif_percent_matrix_by_age_{model_type}.csv"
    
    if not matrix_file.exists():
        print(f"    ERROR: File not found: {matrix_file}")
        return results
    
    print(f"    Found file: {matrix_file}")
    
    try:
        df = pd.read_csv(matrix_file, index_col=0)
        print(f"    Loaded matrix with {len(df)} motifs and {len(df.columns)} ages")
        print(f"    Ages: {list(df.columns)}")
        
        results['matrix'] = df.to_dict()
        
        # Calculate summary statistics
        if not df.empty:
            results['summary'] = {
                'total_motifs': len(df),
                'ages': list(df.columns),
                'mean_percentages_by_age': {age: float(df[age].mean()) for age in df.columns},
                'std_percentages_by_age': {age: float(df[age].std()) for age in df.columns}
            }
            print(f"    Summary: {results['summary']['total_motifs']} motifs across {len(results['summary']['ages'])} ages")
    except Exception as e:
        print(f"    ERROR: Could not parse {matrix_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def extract_transition_summary(helper_dir, parameterization, model_type):
    """Extract transition significance summary from script 05 outputs."""
    results = {
        'model': model_type,
        'transitions': [],
        'summary': {}
    }
    
    script05_dir = Path(helper_dir) / f"{parameterization}_helpers" / "05_motif_analysis"
    
    # Look for motif_transition_significance_summary text file
    summary_file = script05_dir / f"motif_transition_significance_summary_{model_type}.txt"
    
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()
            
            current_transition = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for transition header (e.g., "P12 vs P20")
                trans_match = re.match(r'(P\d+)\s+vs\s+(P\d+)', line)
                if trans_match:
                    current_transition = f"{trans_match.group(1)}_vs_{trans_match.group(2)}"
                    continue
                
                # Parse motif line (e.g., "[motif]: P12 = 10.5%, P20 = 15.2% -> JS Divergence = 0.123, Significant = True")
                if current_transition and 'JS Divergence' in line:
                    # Extract motif
                    motif_match = re.search(r'\[(.*?)\]:', line)
                    if motif_match:
                        motif = motif_match.group(1)
                        
                        # Extract percentages
                        pct_match = re.search(r'P\d+\s*=\s*([\d.]+)%', line)
                        pct1 = float(pct_match.group(1)) if pct_match else None
                        pct_match2 = re.search(r'P\d+\s*=\s*([\d.]+)%', line[pct_match.end():] if pct_match else line)
                        pct2 = float(pct_match2.group(1)) if pct_match2 else None
                        
                        # Extract JSD
                        jsd_match = re.search(r'JS Divergence\s*=\s*([\d.]+)', line)
                        jsd = float(jsd_match.group(1)) if jsd_match else None
                        
                        # Extract significance
                        sig_match = re.search(r'Significant\s*=\s*(True|False)', line)
                        significant = sig_match.group(1) == 'True' if sig_match else None
                        
                        results['transitions'].append({
                            'motif': motif,
                            'transition': current_transition,
                            'percentage_1': pct1,
                            'percentage_2': pct2,
                            'js_divergence': jsd,
                            'significant': significant
                        })
        except Exception as e:
            print(f"Warning: Could not parse {summary_file}: {e}")
    
    # Calculate summary by transition
    if results['transitions']:
        transitions = defaultdict(list)
        for trans in results['transitions']:
            transitions[trans['transition']].append(trans)
        
        results['summary'] = {}
        for trans_name, trans_list in transitions.items():
            jsd_values = [t['js_divergence'] for t in trans_list if t['js_divergence'] is not None]
            significant_count = sum(1 for t in trans_list if t.get('significant', False))
            
            results['summary'][trans_name] = {
                'total_motifs': len(trans_list),
                'significant_count': significant_count,
                'mean_jsd': float(np.mean(jsd_values)) if jsd_values else None,
                'max_jsd': float(np.max(jsd_values)) if jsd_values else None,
                'significant_percentage': (significant_count / len(trans_list) * 100) if trans_list else 0
            }
    
    return results

def extract_upsetplot_data(base_dir, parameterization, ages, model_type):
    """Extract aggregate upsetplot data from each age group."""
    results = {
        'model': model_type,
        'ages': {},
        'summary': {}
    }
    
    for age in ages:
        age_data = {
            'motifs': [],
            'summary': {}
        }
        
        # Look for aggregate upsetplot file
        upsetplot_dir = Path(base_dir) / age / parameterization / "analysis" / model_type
        
        print(f"    [{age}] Searching in: {upsetplot_dir}")
        print(f"    [{age}] Directory exists: {upsetplot_dir.exists()}")
        
        pattern = f"*_ALL_*_filters_upsetplot_{model_type}.csv"
        
        upsetplot_files = find_files_by_pattern(upsetplot_dir, pattern, recursive=False)
        
        if not upsetplot_files:
            # Try case variations
            pattern = f"*_alL_*_filters_upsetplot_{model_type}.csv"
            upsetplot_files = find_files_by_pattern(upsetplot_dir, pattern, recursive=False)
        
        if not upsetplot_files:
            # Try uppercase age prefix
            pattern = f"{age.upper()}_ALL_*_filters_upsetplot_{model_type}.csv"
            upsetplot_files = find_files_by_pattern(upsetplot_dir, pattern, recursive=False)
        
        if not upsetplot_files:
            # Try lowercase age prefix
            pattern = f"{age.lower()}_ALL_*_filters_upsetplot_{model_type}.csv"
            upsetplot_files = find_files_by_pattern(upsetplot_dir, pattern, recursive=False)
        
        if upsetplot_files:
            print(f"    [{age}] Found file: {upsetplot_files[0]}")
            try:
                df = pd.read_csv(upsetplot_files[0])
                print(f"    [{age}] Loaded {len(df)} rows from CSV")
                print(f"    [{age}] Columns: {list(df.columns)}")
                
                # Find column names (case-insensitive)
                motif_col = None
                for col in df.columns:
                    if col.lower() == 'motifs':
                        motif_col = col
                        break
                
                obs_col = None
                for col in df.columns:
                    if col.lower() == 'observed':
                        obs_col = col
                        break
                
                exp_col = None
                for col in df.columns:
                    if col.lower() == 'expected':
                        exp_col = col
                        break
                
                es_col = None
                for col in df.columns:
                    if col.lower() in ['effect size', 'effect_size', 'effect-size']:
                        es_col = col
                        break
                
                pval_col = None
                for col in df.columns:
                    if col.lower() in ['p-value', 'p_value', 'pvalue']:
                        pval_col = col
                        break
                
                if not motif_col:
                    print(f"    [{age}] ERROR: Motifs column not found")
                else:
                    for _, row in df.iterrows():
                        motif = str(row.get(motif_col, '')).strip()
                        # Skip empty/NaN motifs
                        if not motif or motif.lower() in ['nan', 'none', ''] or motif == "['']":
                            continue
                        
                        observed = int(row.get(obs_col, 0)) if obs_col and pd.notna(row.get(obs_col)) else 0
                        expected = float(row.get(exp_col, 0)) if exp_col and pd.notna(row.get(exp_col)) else 0
                        effect_size = float(row.get(es_col, 0)) if es_col and pd.notna(row.get(es_col)) else 0
                        pval = float(row.get(pval_col, np.nan)) if pval_col and pd.notna(row.get(pval_col)) else None
                        significant = pval < 0.05 if pval is not None else False
                        
                        age_data['motifs'].append({
                            'motif': motif,
                            'observed': observed,
                            'expected': expected,
                            'effect_size': effect_size,
                            'p_value': pval,
                            'significant': significant
                        })
                    
                    print(f"    [{age}] Extracted {len(age_data['motifs'])} valid motifs")
                    
                    # Calculate summary
                    if age_data['motifs']:
                        effect_sizes = [m['effect_size'] for m in age_data['motifs']]
                        pvals = [m['p_value'] for m in age_data['motifs'] if m['p_value'] is not None]
                        
                        age_data['summary'] = {
                            'total_motifs': len(age_data['motifs']),
                            'significant_count': sum(1 for m in age_data['motifs'] if m['significant']),
                            'mean_effect_size': float(np.mean(effect_sizes)) if effect_sizes else 0,
                            'mean_abs_effect_size': float(np.mean([abs(es) for es in effect_sizes])) if effect_sizes else 0,
                            'mean_p_value': float(np.mean(pvals)) if pvals else None
                        }
                        print(f"    [{age}] Summary: {age_data['summary']['total_motifs']} motifs, {age_data['summary']['significant_count']} significant")
            except Exception as e:
                print(f"    [{age}] ERROR: Could not parse upsetplot: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    [{age}] ERROR: No upsetplot files found")
        
        results['ages'][age] = age_data
    
    return results

def extract_pie_chart_data(base_dir, parameterization, ages):
    """Extract pie chart data (number of targets per cell) from aggregate files."""
    results = {
        'ages': {},
        'summary': {}
    }
    
    for age in ages:
        age_data = {
            'target_counts': {},
            'percentages': {},
            'total_cells': 0
        }
        
        # Look for aggregate pie chart file (check both model subdirectories and main analysis dir)
        age_dir = Path(base_dir) / age / parameterization / "analysis"
        
        print(f"    [{age}] Searching for pie chart data in: {age_dir}")
        print(f"    [{age}] Directory exists: {age_dir.exists()}")
        
        pie_file = None
        
        # Try model subdirectories first (pie chart data is model-independent, but may be in either)
        for model_subdir in ['uniform', 'region_specific']:
            model_dir = age_dir / model_subdir
            if model_dir.exists():
                patterns = [
                    f"*ALL*pie_chart_data.csv",
                    f"*alL*pie_chart_data.csv",
                    f"{age.upper()}_ALL*pie_chart_data.csv",
                    f"{age.lower()}_ALL*pie_chart_data.csv",
                ]
                
                for pattern in patterns:
                    files = find_files_by_pattern(model_dir, pattern, recursive=False)
                    if files:
                        pie_file = files[0]
                        print(f"    [{age}] Found pie chart file in {model_subdir}: {pie_file}")
                        break
                if pie_file:
                    break
        
        # Fallback to main analysis directory
        if not pie_file:
            patterns = [
                f"*ALL*pie_chart_data.csv",
                f"*alL*pie_chart_data.csv",
                f"{age.upper()}_ALL*pie_chart_data.csv",
                f"{age.lower()}_ALL*pie_chart_data.csv",
            ]
            
            for pattern in patterns:
                files = find_files_by_pattern(age_dir, pattern, recursive=False)
                if files:
                    pie_file = files[0]
                    print(f"    [{age}] Found pie chart file in main directory: {pie_file}")
                    break
        
        if pie_file and Path(pie_file).exists():
            try:
                df = pd.read_csv(pie_file, index_col=0)
                print(f"    [{age}] Loaded pie chart data with {len(df)} target types")
                
                # Extract counts - handle different column names
                if '# Cells' in df.columns:
                    counts_col = '# Cells'
                elif 'Cells' in df.columns:
                    counts_col = 'Cells'
                else:
                    counts_col = df.columns[0]
                
                total_cells = 0
                target_counts = {}
                
                for idx, row in df.iterrows():
                    target_type = str(idx).strip()
                    count = int(row[counts_col]) if pd.notna(row[counts_col]) else 0
                    if count > 0:
                        target_counts[target_type] = count
                        total_cells += count
                
                # Calculate percentages
                percentages = {}
                for target_type, count in target_counts.items():
                    if total_cells > 0:
                        percentages[target_type] = (count / total_cells) * 100
                    else:
                        percentages[target_type] = 0.0
                
                age_data['target_counts'] = target_counts
                age_data['percentages'] = percentages
                age_data['total_cells'] = total_cells
                
                print(f"    [{age}] Extracted data: {len(target_counts)} target types, {total_cells} total cells")
                
            except Exception as e:
                print(f"    [{age}] ERROR: Could not parse pie chart: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    [{age}] ERROR: No pie chart file found")
        
        results['ages'][age] = age_data
    
    # Calculate summary statistics across ages
    if results['ages']:
        # Get all unique target types across all ages
        all_target_types = set()
        for age_data in results['ages'].values():
            all_target_types.update(age_data['percentages'].keys())
        all_target_types = sorted(all_target_types, key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 0)
        
        # Calculate mean percentages across ages for each target type
        mean_percentages = {}
        std_percentages = {}
        for target_type in all_target_types:
            percentages = []
            for age_data in results['ages'].values():
                pct = age_data['percentages'].get(target_type, 0.0)
                percentages.append(pct)
            if percentages:
                mean_percentages[target_type] = float(np.mean(percentages))
                std_percentages[target_type] = float(np.std(percentages))
        
        results['summary'] = {
            'mean_percentages': mean_percentages,
            'std_percentages': std_percentages,
            'target_types': all_target_types
        }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract stability analysis data from pipeline outputs")
    parser.add_argument('--base_dir', type=str, default='02_output',
                       help='Base output directory (default: 02_output)')
    parser.add_argument('--parameterization', type=str, default='05.HAN_filter_parameters_i300_r10_t10_u5',
                       help='Parameterization name (default: 05.HAN_filter_parameters_i300_r10_t10_u5)')
    parser.add_argument('--output', type=str, default='extracted_stability_data.json',
                       help='Output JSON file (default: extracted_stability_data.json)')
    args = parser.parse_args()
    
    # Resolve base_dir relative to repo root if it's a relative path
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        # Resolve relative to repo root (two levels up from script: conclusions/scripts -> conclusions -> repo root)
        script_dir = Path(__file__).parent.parent.parent
        base_dir = (script_dir / base_dir).resolve()
    
    print(f"Resolved base directory: {base_dir}")
    print(f"Base directory exists: {base_dir.exists()}")
    
    parameterization = args.parameterization
    ages = ['p3', 'p12', 'p20', 'p60']
    models = ['uniform', 'region_specific']
    
    all_data = {
        'parameterization': parameterization,
        'base_dir': str(base_dir),
        'models': {}
    }
    
    print("Extracting stability analysis data...")
    print(f"Base directory: {base_dir}")
    print(f"Parameterization: {parameterization}")
    print("=" * 80)
    
    for model_type in models:
        print(f"\nProcessing {model_type} model...")
        model_data = {}
        
        # Extract from each source
        print("  - Extracting Kruskal-Wallis results (script 01)...")
        model_data['kruskal_wallis'] = extract_kruskal_wallis_results(base_dir, parameterization, model_type)
        
        print("  - Extracting transition significance (script 07)...")
        model_data['transition_significance'] = extract_transition_significance(base_dir, parameterization, model_type)
        
        print("  - Extracting effect size trajectories (script 07)...")
        model_data['effect_sizes'] = extract_effect_sizes(base_dir, parameterization, model_type)
        
        print("  - Extracting motif percentages (script 05)...")
        model_data['motif_percentages'] = extract_motif_percentages(base_dir, parameterization, model_type)
        
        print("  - Extracting transition summary (script 05)...")
        model_data['transition_summary'] = extract_transition_summary(base_dir, parameterization, model_type)
        
        print("  - Extracting upsetplot data (per age)...")
        model_data['upsetplot_data'] = extract_upsetplot_data(base_dir, parameterization, ages, model_type)
        
        all_data['models'][model_type] = model_data
    
    # Extract pie chart data (model-independent, shared across models)
    print("\nExtracting pie chart data (number of targets per cell)...")
    all_data['pie_chart_data'] = extract_pie_chart_data(base_dir, parameterization, ages)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 Data saved to: {output_path}")
    print(f"📊 Total motifs analyzed: {sum(len(m['kruskal_wallis']['results']) for m in all_data['models'].values())}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
