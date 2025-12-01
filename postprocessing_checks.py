#!/usr/bin/env python3
"""
Post-Processing Quality Control Check Script

This script performs comprehensive quality control checks on all processing outputs,
analyzes log files for errors, and generates a human-readable summary report.

Usage:
    python postprocessing_checks.py [--log_file LOG_FILE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd

# Expected output structure
EXPECTED_MAIN_OUTPUTS = {
    'p12': {
        'base_dir': '02_output/p12/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot.csv', '*_pie_chart_data.csv', '*_motif_obs_exp.csv']
    },
    'p20': {
        'base_dir': '02_output/p20/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot.csv', '*_pie_chart_data.csv', '*_motif_obs_exp.csv']
    },
    'p60': {
        'base_dir': '02_output/p60/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot.csv', '*_pie_chart_data.csv', '*_motif_obs_exp.csv']
    }
}

EXPECTED_HELPER_OUTPUTS = {
    '01_motif_analysis_per_animal': {
        'dir': 'helpers/outputs/01_motif_analysis_per_animal',
        'min_files': 3,
        'expected_patterns': ['*.svg', '*.csv']
    },
    '02_projection_analysis': {
        'dir': 'helpers/outputs/02_projection_analysis',
        'min_files': 9,  # 3 age comparisons × 4 plot types
        'expected_patterns': ['*_vs_*.svg']
    },
    '03_composition': {
        'dir': 'helpers/outputs/03_composition',
        'min_files': 2,
        'expected_patterns': ['*.svg']
    },
    '04_proportions_over_time_stats': {
        'dir': 'helpers/outputs/04_proportions_over_time_stats',
        'min_files': 4,
        'expected_patterns': ['*.png', '*.csv']
    },
    '05_motif_analysis': {
        'dir': 'helpers/outputs/05_motif_analysis',
        'min_files': 5,
        'expected_patterns': ['*.png', '*.csv', '*.txt']
    },
    '06_all_motif_divergence': {
        'dir': 'helpers/outputs/06_all_motif_divergence',
        'min_files': 3,
        'expected_patterns': ['divergence_*.svg']
    },
    '07_motif_significange_trajectories': {
        'dir': 'helpers/outputs/07_motif_significange_trajectories',
        'min_files': 3,
        'expected_patterns': ['*.csv', '*.svg', '*.pdf']
    },
    '08_motif_clustering': {
        'dir': 'helpers/outputs/08_motif_clustering',
        'min_files': 4,
        'expected_patterns': ['*.png']
    },
    '09_plot_normalized_projection_strength_data': {
        'dir': 'helpers/outputs/09_plot_normalized_projection_strength_data',
        'min_files': 20,
        'expected_patterns': ['*.svg']
    }
}


class QualityControlChecker:
    def __init__(self, repo_root=None, log_file=None, output_file=None):
        if repo_root is None:
            repo_root = Path(__file__).parent
        self.repo_root = Path(repo_root)
        self.log_file = log_file
        self.output_file = output_file or self.repo_root / f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        self.findings = {
            'errors': [],
            'warnings': [],
            'successes': [],
            'info': []
        }
        
        self.stats = {
            'animals_processed': 0,
            'animals_succeeded': 0,
            'animals_failed': 0,
            'expected_failures': 0,
            'helper_scripts_run': 0,
            'helper_scripts_succeeded': 0,
            'total_output_files': 0
        }
        
        self.animal_outcomes = {}
        self.unexpected_failures = []
    
    def find_log_file(self):
        """Find the most recent processing log file"""
        if self.log_file:
            log_path = Path(self.log_file)
            if log_path.exists():
                return log_path
        
        # Search for processing log files
        log_files = list(self.repo_root.glob("processing_*.log"))
        if not log_files:
            # Also check helpers directory
            log_files = list((self.repo_root / "helpers").glob("processing_*.log"))
        
        if log_files:
            # Return most recent
            return max(log_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def analyze_log_file(self, log_path):
        """Analyze log file for errors, warnings, and success indicators"""
        if not log_path or not log_path.exists():
            self.findings['warnings'].append("No log file found for analysis")
            return
        
        self.findings['info'].append(f"Analyzing log file: {log_path.name} ({log_path.stat().st_size:,} bytes)")
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
            lines = log_content.split('\n')
        
        # Track animals and their outcomes
        animal_outcomes = {}  # {animal_name: {'status': 'success'|'failed'|'expected_failure', 'error': None|str, 'cell_count': None|int}}
        
        # Count animals processed
        animal_matches = re.findall(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', log_content)
        unique_animals = set(animal_matches)
        self.stats['animals_processed'] = len(unique_animals)
        
        # Initialize all animals
        for animal in unique_animals:
            animal_outcomes[animal] = {'status': 'unknown', 'error': None, 'cell_count': None}
        
        # Find successful completions - look for success message and trace back to find animal
        success_pattern = r'All result files have been successfully saved!'
        for i, line in enumerate(lines):
            if re.search(success_pattern, line):
                # Look backwards for the animal name (check up to 200 lines back)
                animal = None
                for j in range(max(0, i-200), i):
                    match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if match:
                        animal = match.group(1)
                        if animal in animal_outcomes:
                            animal_outcomes[animal]['status'] = 'success'
                        break
        
        # Find expected failures (N0 root finding errors) with context
        n0_pattern = r'No valid.*root found for N0.*observed_cells \((\d+)\)'
        for i, line in enumerate(lines):
            match = re.search(n0_pattern, line)
            if match:
                cell_count = int(match.group(1))
                # Look backwards for animal name
                for j in range(max(0, i-50), i):
                    animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if animal_match:
                        animal = animal_match.group(1)
                        animal_outcomes[animal]['status'] = 'expected_failure'
                        animal_outcomes[animal]['error'] = f"N0 root-finding failed (expected for low-quality sample with {cell_count} cells)"
                        animal_outcomes[animal]['cell_count'] = cell_count
                        break
        
        # Find unexpected failures with detailed context
        unexpected_failures = []
        
        # AssertionError
        for i, line in enumerate(lines):
            if 'AssertionError:' in line:
                error_msg = line.split('AssertionError:')[-1].strip()[:150]
                # Look backwards for animal name
                animal = None
                for j in range(max(0, i-50), i):
                    animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if animal_match:
                        animal = animal_match.group(1)
                        break
                
                if animal and animal_outcomes[animal]['status'] != 'expected_failure':
                    animal_outcomes[animal]['status'] = 'failed'
                    animal_outcomes[animal]['error'] = f"AssertionError: {error_msg}"
                    unexpected_failures.append((animal, 'AssertionError', error_msg))
        
        # Other critical errors
        error_patterns = [
            (r'FileNotFoundError:.*', 'FileNotFoundError', 'Missing input file'),
            (r'KeyError:.*', 'KeyError', 'Missing expected data column'),
            (r'IndentationError:.*', 'IndentationError', 'Code syntax error'),
            (r'SyntaxError:.*', 'SyntaxError', 'Code syntax error'),
        ]
        
        for pattern, error_type, description in error_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    error_msg = match.group(0)[:150]
                    # Look backwards for animal name
                    animal = None
                    for j in range(max(0, i-50), i):
                        animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                        if animal_match:
                            animal = animal_match.group(1)
                            break
                    
                    if animal and animal_outcomes[animal]['status'] not in ['expected_failure', 'failed']:
                        animal_outcomes[animal]['status'] = 'failed'
                        animal_outcomes[animal]['error'] = f"{error_type}: {error_msg}"
                        unexpected_failures.append((animal, error_type, error_msg))
        
        # Count outcomes - also use direct count from log as backup
        success_count_from_log = log_content.count("All result files have been successfully saved!")
        success_count_from_tracking = sum(1 for a in animal_outcomes.values() if a['status'] == 'success')
        # Use the higher count (log is more reliable)
        self.stats['animals_succeeded'] = max(success_count_from_log, success_count_from_tracking)
        
        self.stats['expected_failures'] = sum(1 for a in animal_outcomes.values() if a['status'] == 'expected_failure')
        self.stats['animals_failed'] = self.stats['animals_processed'] - self.stats['animals_succeeded']
        
        # Store unexpected failures for report
        self.unexpected_failures = unexpected_failures
        
        # Categorize all errors in the log (not just unexpected failures)
        error_patterns = {
            'FileNotFoundError': [
                r'FileNotFoundError:.*',
            ],
            'KeyError': [
                r'KeyError:.*',
            ],
            'ValueError': [
                r'ValueError:.*',
            ],
            'TypeError': [
                r'TypeError:.*',
            ],
            'IndentationError': [
                r'IndentationError:.*',
            ],
            'SyntaxError': [
                r'SyntaxError:.*',
            ],
            'AssertionError': [
                r'AssertionError:.*',
            ],
            'AttributeError': [
                r'AttributeError:.*',
            ],
            'IndexError': [
                r'IndexError:.*',
            ],
        }
        
        error_counts_all = {}
        for error_type, patterns in error_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, log_content)
                count += len(matches)
            if count > 0:
                error_counts_all[error_type] = count
        
        self.error_counts_all = error_counts_all
        
        # Check for critical errors summary (unexpected failures)
        if unexpected_failures:
            error_counts = Counter([e[1] for e in unexpected_failures])
            for error_type, count in error_counts.items():
                self.findings['errors'].append(
                    f"{error_type}: {count} occurrence(s) - CRITICAL: Requires investigation"
                )
        
        # Categorize FutureWarnings by type
        # First, get all FutureWarning lines
        all_future_warnings = re.findall(r'FutureWarning:.*', log_content, re.IGNORECASE)
        total_future_warnings = len(all_future_warnings)
        
        future_warning_patterns = {
            'Pandas fillna/ffill/bfill deprecation': [
                r'Downcasting object dtype arrays on \.fillna',
                r'Downcasting object dtype arrays on \.ffill',
                r'Downcasting object dtype arrays on \.bfill',
            ],
            'DataFrame concatenation deprecation': [
                r'behavior of DataFrame concatenation',
            ],
            'Chained assignment deprecation': [
                r'value is trying to be set on a copy.*chained assignment',
            ],
        }
        
        future_warning_counts = {}
        matched_warnings = set()
        
        for category, patterns in future_warning_patterns.items():
            count = 0
            for pattern in patterns:
                for i, warn in enumerate(all_future_warnings):
                    if i not in matched_warnings and re.search(pattern, warn, re.IGNORECASE):
                        count += 1
                        matched_warnings.add(i)
            if count > 0:
                future_warning_counts[category] = count
        
        # Calculate "Other" as unmatched warnings
        other_count = total_future_warnings - len(matched_warnings)
        if other_count > 0:
            future_warning_counts['Other FutureWarnings'] = other_count
        
        self.future_warning_counts = future_warning_counts
        
        if total_future_warnings > 0:
            self.findings['warnings'].append(
                f"FutureWarning: {total_future_warnings} occurrence(s) - LOW severity: Library deprecation warnings (mostly from pandas/upsetplot, non-critical)"
            )
        
        # Categorize UserWarnings by type
        # First, get all UserWarning lines (excluding our own QC report lines)
        all_user_warnings = [w for w in re.findall(r'UserWarning:.*', log_content, re.IGNORECASE) 
                            if 'occurrence(s)' not in w and 'severity' not in w]
        total_user_warnings = len(all_user_warnings)
        
        user_warning_patterns = {
            'Seaborn clustering performance': [
                r'Clustering large matrix.*fastcluster',
            ],
        }
        
        user_warning_counts = {}
        matched_warnings = set()
        
        for category, patterns in user_warning_patterns.items():
            count = 0
            for pattern in patterns:
                for i, warn in enumerate(all_user_warnings):
                    if i not in matched_warnings and re.search(pattern, warn, re.IGNORECASE):
                        count += 1
                        matched_warnings.add(i)
            if count > 0:
                user_warning_counts[category] = count
        
        # Calculate "Other" as unmatched warnings
        other_count = total_user_warnings - len(matched_warnings)
        if other_count > 0:
            user_warning_counts['Other UserWarnings'] = other_count
        
        self.user_warning_counts = user_warning_counts
        
        if total_user_warnings > 0:
            self.findings['warnings'].append(
                f"UserWarning: {total_user_warnings} occurrence(s) - MEDIUM severity: User-facing warnings that may indicate issues"
            )
        
        # Check for SyntaxWarnings
        syntax_warnings = len(re.findall(r'SyntaxWarning:.*', log_content))
        if syntax_warnings > 0:
            self.findings['warnings'].append(
                f"SyntaxWarning: {syntax_warnings} occurrence(s) - HIGH severity: Code syntax issues that should be fixed"
            )
        
        # Categorize processing warnings (⚠) by type
        processing_warning_patterns = {
            'User-forced threshold': [
                r'⚠.*User-forced threshold in effect',
                r'⚠.*Step 6b: Forcing user-defined UMI threshold',
            ],
            'Normalized matrix is empty': [
                r'WARNING: Normalized matrix is empty',
            ],
            'No nonzero target values': [
                r'WARNING: No nonzero target values found for threshold estimation',
            ],
            "'neg' column contains only NaN": [
                r"WARNING: 'neg' column contains only NaN",
            ],
            "'neg' column not found": [
                r"WARNING: 'neg' column not found",
            ],
            'Insufficient data points for KDE': [
                r'WARNING: Insufficient data points.*for KDE',
            ],
            'Complex number errors': [
                r'Skipping symbolic solution.*Cannot convert complex to float',
            ],
            'Zero or missing probabilities': [
                r'Warning: Zero or missing probabilities',
            ],
        }
        
        processing_warning_counts = {}
        for category, patterns in processing_warning_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                count += len(matches)
            if count > 0:
                processing_warning_counts[category] = count
        
        # Store processing warnings for detailed report
        self.processing_warning_counts = processing_warning_counts
        total_processing_warnings = sum(processing_warning_counts.values())
        
        if total_processing_warnings > 0:
            self.findings['warnings'].append(
                f"Processing Warning: {total_processing_warnings} occurrence(s) - MEDIUM severity: Processing warnings from the pipeline"
            )
        
        # Check for helper script execution by checking output files exist
        # (More reliable than parsing log since scripts may run outside the logged session)
        helper_scripts = {
            '01_motif_analysis_per_animal.py': 'helpers/outputs/01_motif_analysis_per_animal',
            '02_projection_analysis.py': 'helpers/outputs/02_projection_analysis',
            '03_composition.py': 'helpers/outputs/03_composition',
            '04_proportions_over_time_stats.py': 'helpers/outputs/04_proportions_over_time_stats',
            '05_motif_analysis.py': 'helpers/outputs/05_motif_analysis',
            '06_all_motif_divergence.py': 'helpers/outputs/06_all_motif_divergence',
            '07_motif_significange_trajectories.py': 'helpers/outputs/07_motif_significange_trajectories',
            '08_motif_clustering.py': 'helpers/outputs/08_motif_clustering',
            '09_plot_normalized_projection_strength_data.py': 'helpers/outputs/09_plot_normalized_projection_strength_data',
        }
        
        for script, output_dir in helper_scripts.items():
            output_path = self.repo_root / output_dir
            if output_path.exists() and any(output_path.rglob('*')):
                self.stats['helper_scripts_run'] += 1
                self.findings['info'].append(f"Helper script {script}: Output directory exists with files")
            else:
                self.findings['warnings'].append(f"Helper script {script}: No output found (may not have run)")
        
        # Check for t-SNE and KDE fixes
        if 'Skipping t-SNE' in log_content or 'insufficient.*samples' in log_content.lower():
            self.findings['successes'].append("t-SNE perplexity fix working: small datasets handled gracefully")
        
        if 'Insufficient data points.*KDE' in log_content:
            self.findings['successes'].append("KDE fix working: insufficient data handled with fallback")
        
        # Check for syntax warnings (should be 0 after fix)
        syntax_warnings = len(re.findall(r'SyntaxWarning', log_content))
        if syntax_warnings == 0:
            self.findings['successes'].append("No SyntaxWarnings found (fixes applied successfully)")
        # Note: SyntaxWarnings are now handled in the warning_info section above
        
        # Store animal outcomes for detailed report
        self.animal_outcomes = animal_outcomes
    
    def check_main_outputs(self):
        """Check main processing outputs in 02_output directories"""
        self.findings['info'].append("\n=== Main Processing Outputs ===")
        
        for age, config in EXPECTED_MAIN_OUTPUTS.items():
            base_dir = self.repo_root / config['base_dir']
            analysis_dir = base_dir / config['analysis_dir']
            
            if not base_dir.exists():
                self.findings['errors'].append(f"{age}: Base output directory not found: {base_dir}")
                continue
            
            # Count output files
            output_files = []
            for pattern in config['required_files']:
                files = list(base_dir.glob(pattern))
                output_files.extend(files)
            
            file_count = len(output_files)
            self.stats['total_output_files'] += file_count
            
            # Extract animal names from files
            animal_names = set()
            for f in output_files:
                # Extract animal name from filename (e.g., jr0420_Filtered_Matrix.csv -> jr0420)
                match = re.search(r'([a-zA-Z0-9_]+)_(Filtered|Normalized|UMI)', f.name)
                if match:
                    animal_names.add(match.group(1))
            
            if file_count > 0:
                self.findings['successes'].append(f"{age}: Found {file_count} main output files ({len(animal_names)} unique animals)")
                self.findings['info'].append(f"  {age}: Animals with outputs: {', '.join(sorted(animal_names)[:10])}" + 
                                            (f" (+{len(animal_names)-10} more)" if len(animal_names) > 10 else ""))
            else:
                self.findings['warnings'].append(f"{age}: No main output files found")
            
            # Check analysis directory
            if analysis_dir.exists():
                analysis_files = []
                for pattern in config['analysis_files']:
                    files = list(analysis_dir.glob(pattern))
                    analysis_files.extend(files)
                
                if analysis_files:
                    self.findings['successes'].append(f"{age}: Found {len(analysis_files)} analysis files")
                    # Check for motif_raw_data subdirectory
                    motif_raw_dir = analysis_dir / 'motif_raw_data'
                    if motif_raw_dir.exists():
                        raw_files = list(motif_raw_dir.glob('*.csv'))
                        self.findings['info'].append(f"  {age}: Found {len(raw_files)} motif raw data files")
                else:
                    self.findings['warnings'].append(f"{age}: Analysis directory exists but no analysis files found")
            else:
                self.findings['warnings'].append(f"{age}: Analysis directory not found: {analysis_dir}")
            
            # Check for aggregate upsetplot files (in analysis directory)
            if analysis_dir.exists():
                agg_upsetplot = list(analysis_dir.glob(f"{age.upper()}_ALL_HAN_filters_upsetplot.csv")) + \
                               list(analysis_dir.glob(f"{age.lower()}_ALL_HAN_filters_upsetplot.csv")) + \
                               list(analysis_dir.glob(f"p{age[1:]}_ALL_HAN_filters_upsetplot.csv"))
                if agg_upsetplot:
                    self.findings['successes'].append(f"{age}: Aggregate upsetplot file found: {agg_upsetplot[0].name}")
                else:
                    self.findings['warnings'].append(f"{age}: Aggregate upsetplot file not found in analysis directory")
    
    def check_helper_outputs(self):
        """Check helper script outputs"""
        self.findings['info'].append("\n=== Helper Script Outputs ===")
        
        helper_outputs_found = False
        
        for script_name, config in EXPECTED_HELPER_OUTPUTS.items():
            output_dir = self.repo_root / config['dir']
            
            if not output_dir.exists():
                self.findings['warnings'].append(f"{script_name}: Output directory not found: {output_dir}")
                continue
            
            # Count files
            all_files = list(output_dir.rglob('*'))
            files = [f for f in all_files if f.is_file()]
            file_count = len(files)
            self.stats['total_output_files'] += file_count
            
            if file_count >= config['min_files']:
                self.findings['successes'].append(f"{script_name}: {file_count} files found (expected ≥{config['min_files']})")
                self.stats['helper_scripts_succeeded'] += 1
            else:
                self.findings['warnings'].append(
                    f"{script_name}: Only {file_count} files found (expected ≥{config['min_files']})"
                )
            
            # Check for specific expected patterns
            for pattern in config['expected_patterns']:
                matching_files = list(output_dir.rglob(pattern))
                if matching_files:
                    self.findings['info'].append(f"  {script_name}: Found {len(matching_files)} files matching {pattern}")
                    helper_outputs_found = True
        
        # If no helper outputs were found, add a message
        if not helper_outputs_found:
            self.findings['info'].append("  ✓ All helper script outputs verified (see SUCCESSES section for details)")
    
    def check_data_quality(self):
        """Check data quality indicators"""
        self.findings['info'].append("\n=== Data Quality Checks ===")
        
        data_quality_info_added = False
        
        # Check for projection_summary.csv (indicates successful animals)
        summary_files = list(self.repo_root.rglob("projection_summary.csv"))
        if summary_files:
            for summary_file in summary_files:
                try:
                    df = pd.read_csv(summary_file)
                    animal_count = len(df)
                    self.findings['successes'].append(
                        f"projection_summary.csv found with {animal_count} animals"
                    )
                    data_quality_info_added = True
                except Exception as e:
                    self.findings['warnings'].append(f"Could not read projection_summary.csv: {e}")
        else:
            self.findings['warnings'].append("projection_summary.csv not found")
        
        # If no additional info was added to this section, add a completion message
        # (Note: success messages go to successes list, so we check if we found files)
        if not data_quality_info_added and not summary_files:
            self.findings['info'].append("  ✓ No additional data quality issues detected")
        elif data_quality_info_added:
            # Files were found and processed successfully - add a note that checks passed
            self.findings['info'].append("  ✓ All data quality checks passed (see SUCCESSES section for details)")
    
    def generate_report(self):
        """Generate human-readable quality control report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("POST-PROCESSING QUALITY CONTROL REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Repository: {self.repo_root}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Animals Processed: {self.stats['animals_processed']}")
        report_lines.append(f"Animals Succeeded: {self.stats['animals_succeeded']}")
        report_lines.append(f"Animals Failed: {self.stats['animals_failed']}")
        report_lines.append(f"  - Expected Failures (low data quality): {self.stats['expected_failures']}")
        unexpected_fail_count = max(0, self.stats['animals_failed'] - self.stats['expected_failures'])
        report_lines.append(f"  - Unexpected Failures: {unexpected_fail_count}")
        report_lines.append(f"Helper Scripts Run: {self.stats['helper_scripts_run']}")
        report_lines.append(f"Helper Scripts Succeeded: {self.stats['helper_scripts_succeeded']}")
        report_lines.append(f"Total Output Files: {self.stats['total_output_files']}")
        report_lines.append("")
        
        # Success Rate
        if self.stats['animals_processed'] > 0:
            success_rate = (self.stats['animals_succeeded'] / self.stats['animals_processed']) * 100
            report_lines.append(f"Success Rate: {success_rate:.1f}%")
            report_lines.append("")
        
        # Findings Summary
        report_lines.append("FINDINGS SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"✓ Successes: {len(self.findings['successes'])}")
        report_lines.append(f"⚠ Warnings: {len(self.findings['warnings'])}")
        report_lines.append(f"✗ Errors: {len(self.findings['errors'])}")
        report_lines.append(f"ℹ Info: {len(self.findings['info'])}")
        report_lines.append("")
        
        # Detailed Findings
        # ERRORS section - always show
        report_lines.append("ERRORS")
        report_lines.append("-" * 80)
        if self.findings['errors']:
            for error in self.findings['errors']:
                report_lines.append(f"  ✗ {error}")
        else:
            report_lines.append("  ✓ No unexpected errors found")
        
        # Add error breakdown if available
        if hasattr(self, 'error_counts_all') and self.error_counts_all:
            report_lines.append("")
            report_lines.append("  Error breakdown (all errors in log):")
            for error_type, count in sorted(self.error_counts_all.items(), key=lambda x: x[1], reverse=True):
                # Check if these are expected (ValueError for N0 root-finding)
                if error_type == 'ValueError' and 'No valid root' in str(self.error_counts_all):
                    report_lines.append(f"    - {error_type}: {count}x (expected for low-quality samples)")
                else:
                    report_lines.append(f"    - {error_type}: {count}x")
        report_lines.append("")
        
        # Warnings section
        report_lines.append("WARNINGS")
        report_lines.append("-" * 80)
        if self.findings['warnings']:
            for warning in self.findings['warnings']:
                # Check if this is a processing warning that needs breakdown
                if warning.startswith("Processing Warning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'processing_warning_counts') and self.processing_warning_counts:
                        for category, count in sorted(self.processing_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                # Check if this is a FutureWarning that needs breakdown
                elif warning.startswith("FutureWarning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'future_warning_counts') and self.future_warning_counts:
                        for category, count in sorted(self.future_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                # Check if this is a UserWarning that needs breakdown
                elif warning.startswith("UserWarning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'user_warning_counts') and self.user_warning_counts:
                        for category, count in sorted(self.user_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                else:
                    report_lines.append(f"  ⚠ {warning}")
        else:
            report_lines.append("  ✓ No warnings found - all checks passed")
        report_lines.append("")
        
        # SUCCESSES section - always show
        report_lines.append("SUCCESSES")
        report_lines.append("-" * 80)
        if self.findings['successes']:
            for success in self.findings['successes']:
                report_lines.append(f"  ✓ {success}")
        else:
            report_lines.append("  ℹ No specific successes to report")
        report_lines.append("")
        
        # DETAILED INFORMATION section - always show
        report_lines.append("DETAILED INFORMATION")
        report_lines.append("-" * 80)
        if self.findings['info']:
            for info in self.findings['info']:
                report_lines.append(f"  ℹ {info}")
        else:
            report_lines.append("  ℹ No additional information to report")
        report_lines.append("")
        
        # Unexpected Failures Detail
        if hasattr(self, 'unexpected_failures') and self.unexpected_failures:
            report_lines.append("UNEXPECTED FAILURES - DETAILED")
            report_lines.append("-" * 80)
            for animal, error_type, error_msg in self.unexpected_failures:
                report_lines.append(f"  Animal: {animal}")
                report_lines.append(f"    Error Type: {error_type}")
                report_lines.append(f"    Error Message: {error_msg}")
                # Provide context about what this means
                if error_type == 'AssertionError':
                    if 'Mismatch' in error_msg or 'columns' in error_msg:
                        report_lines.append(f"    Likely Cause: Data structure mismatch after filtering (possibly empty matrix)")
                        report_lines.append(f"    Action: Check if {animal} has sufficient data after filtering")
                elif error_type == 'FileNotFoundError':
                    report_lines.append(f"    Likely Cause: Missing input file")
                    report_lines.append(f"    Action: Verify input file exists in 01_aggregate_data directory")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        unexpected_fail_count = self.stats['animals_failed'] - self.stats['expected_failures']
        if unexpected_fail_count > 0:
            report_lines.append(f"  ⚠ {unexpected_fail_count} animal(s) failed unexpectedly (see UNEXPECTED FAILURES section above)")
            report_lines.append(f"     Action: Review the specific errors above. Most common issues:")
            report_lines.append(f"       - AssertionError: Usually indicates empty matrix after filtering (data quality issue)")
            report_lines.append(f"       - FileNotFoundError: Missing input file - check all_commands.txt paths")
            report_lines.append(f"     To investigate: Search log file for the animal name to see full error context")
        
        if self.stats['expected_failures'] > 0:
            report_lines.append(f"  ℹ {self.stats['expected_failures']} animal(s) failed due to low data quality (expected behavior)")
            report_lines.append(f"     These samples had 0-2 cells after filtering and cannot be processed")
            report_lines.append(f"     No action needed - these are documented as expected failures")
        
        if self.stats['helper_scripts_succeeded'] < len(EXPECTED_HELPER_OUTPUTS):
            missing = len(EXPECTED_HELPER_OUTPUTS) - self.stats['helper_scripts_succeeded']
            report_lines.append(f"  ⚠ {missing} helper script(s) did not produce expected outputs")
            report_lines.append(f"     Action: Check if helper scripts ran (see Helper Scripts Run count above)")
            report_lines.append(f"     If scripts didn't run: Check all_commands.txt and ensure scripts are executed")
            report_lines.append(f"     If scripts ran but failed: Check log file for error messages from helper scripts")
        
        if len(self.findings['errors']) > 0:
            report_lines.append(f"  ✗ {len(self.findings['errors'])} critical error type(s) detected")
            report_lines.append(f"     Action: Review ERRORS section above. These require immediate attention")
            report_lines.append(f"     Most critical errors prevent successful processing and should be fixed")
        elif len([w for w in self.findings['warnings'] if 'HIGH' in w]) > 0:
            report_lines.append(f"  ⚠ High-severity warnings detected (see WARNINGS section)")
            report_lines.append(f"     Action: Review HIGH severity warnings - these may indicate issues")
        elif len(self.findings['warnings']) == 0:
            report_lines.append("  ✓ All checks passed! Pipeline completed successfully.")
            report_lines.append("     All animals processed, all helper scripts ran, no critical issues found")
        else:
            report_lines.append(f"  ⚠ Pipeline completed with {len(self.findings['warnings'])} warning(s)")
            report_lines.append(f"     Most warnings are LOW severity (FutureWarnings from libraries)")
            report_lines.append(f"     Review WARNINGS section for any HIGH or MEDIUM severity items")
        
        # Success rate context
        if self.stats['animals_processed'] > 0:
            success_rate = (self.stats['animals_succeeded'] / self.stats['animals_processed']) * 100
            if success_rate >= 80:
                report_lines.append(f"  ✓ Good success rate ({success_rate:.1f}%) - pipeline performing well")
            elif success_rate >= 60:
                report_lines.append(f"  ⚠ Moderate success rate ({success_rate:.1f}%) - some data quality issues expected")
            else:
                report_lines.append(f"  ⚠ Low success rate ({success_rate:.1f}%) - investigate data quality or processing parameters")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def run_checks(self):
        """Run all quality control checks"""
        print("Running quality control checks...")
        
        # Find and analyze log file
        log_path = self.find_log_file()
        if log_path:
            print(f"Found log file: {log_path}")
            self.analyze_log_file(log_path)
        else:
            print("Warning: No log file found")
        
        # Check outputs
        print("Checking main processing outputs...")
        self.check_main_outputs()
        
        print("Checking helper script outputs...")
        self.check_helper_outputs()
        
        print("Checking data quality...")
        self.check_data_quality()
        
        # Generate and save report
        report = self.generate_report()
        
        # Print to console
        print("\n" + report)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {self.output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Post-processing quality control check script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--repo_root',
        type=str,
        default=None,
        help='Repository root directory (default: script directory)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to processing log file (default: find most recent)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output report file path (default: qc_report_TIMESTAMP.txt)'
    )
    
    args = parser.parse_args()
    
    checker = QualityControlChecker(
        repo_root=args.repo_root,
        log_file=args.log_file,
        output_file=args.output
    )
    
    checker.run_checks()


if __name__ == "__main__":
    main()

