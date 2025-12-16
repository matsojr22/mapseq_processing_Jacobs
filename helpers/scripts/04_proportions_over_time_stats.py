import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid Qt/Wayland errors

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import glob
from pathlib import Path

# --- Dynamically load pie chart data from 02_output directory ---
# Get repository root (assuming helpers/ is a subdirectory)
import argparse
parser = argparse.ArgumentParser(description="Analyze proportions over time")
parser.add_argument('--base_output_dir', type=str, default=None,
                   help='Base output directory for processing results (default: REPO_ROOT/02_output)')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/04_proportions_over_time_stats)')
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
if args.base_output_dir:
    OUTPUT_DIR = Path(args.base_output_dir)
else:
    OUTPUT_DIR = REPO_ROOT / "02_output"

# Extract parameterization name and filter type from helper_output_dir if provided
parameterization_filter = None
filter_type = None
if args.helper_output_dir:
    helper_path = Path(args.helper_output_dir)
    # Look for parameterization name in path (e.g., .../01.minimal_filter_parameters_..._helpers/...)
    for part in helper_path.parts:
        if part.startswith(('01.', '02.', '03.', '04.', '05.')):
            # Extract parameterization name (before _helpers if present)
            if '_helpers' in part:
                parameterization_filter = part.split('_helpers')[0]
            else:
                parameterization_filter = part
            # Extract filter type from parameterization name
            if 'minimal' in part:
                filter_type = 'minimal'
            elif 'medium' in part:
                filter_type = 'medium'
            elif 'strict' in part:
                filter_type = 'strict'
            elif 'extreme' in part:
                filter_type = 'extreme'
            elif 'HAN' in part:
                filter_type = 'HAN'
            print(f"Filtering by parameterization: {parameterization_filter}, filter type: {filter_type}")
            break

# Define age groups (p3 was removed from manuscript, so we skip it)
age_groups = ['p12', 'p20', 'p60']
# p3 data was removed in manuscript before submission - commented out
# age_groups = ['p3', 'p12', 'p20', 'p60']

# Target types: 1-7 targets (based on 7 regions: rsp, pm, am, a, rl, al, lm)
target_types = [f'{i} target{"s" if i > 1 else ""}' for i in range(1, 8)]

# Dictionary to store data for each age
age_data = {}

for age in age_groups:
    # Look for aggregate pie chart files
    # Filter by parameterization and filter type if specified
    if parameterization_filter and filter_type:
        # Search only in the specific parameterization directory
        param_path = OUTPUT_DIR / age / parameterization_filter
        if param_path.exists():
            patterns = [
                str(param_path / "**" / f"*ALL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"*alL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"{age.upper()}_ALL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"{age.lower()}_ALL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"{age.upper()}_alL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"{age.lower()}_alL_{filter_type}_filters_pie_chart_data.csv"),
                # Handle p60 uppercase variation
                str(param_path / "**" / f"P60_ALL_{filter_type}_filters_pie_chart_data.csv"),
                str(param_path / "**" / f"P60_alL_{filter_type}_filters_pie_chart_data.csv"),
            ]
        else:
            patterns = []
    else:
        # Original behavior: search for HAN filters (backward compatibility)
        patterns = [
            str(OUTPUT_DIR / age / "**" / f"*ALL_HAN_filters_pie_chart_data.csv"),
            str(OUTPUT_DIR / age / "**" / f"*alL_HAN_filters_pie_chart_data.csv"),
            str(OUTPUT_DIR / age / "**" / f"{age.upper()}_ALL_HAN_filters_pie_chart_data.csv"),
            str(OUTPUT_DIR / age / "**" / f"{age.lower()}_ALL_HAN_filters_pie_chart_data.csv"),
            str(OUTPUT_DIR / age / "**" / f"{age.upper()}_alL_HAN_filters_pie_chart_data.csv"),
            str(OUTPUT_DIR / age / "**" / f"{age.lower()}_alL_HAN_filters_pie_chart_data.csv"),
        ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = list(set(files))  # Remove duplicates
    
    if not files:
        print(f"Warning: No aggregate pie chart file found for {age}")
        # Create empty data with zeros for all target types
        age_data[age] = {target_type: 0 for target_type in target_types}
        continue
    
    # Use the first matching file (should only be one aggregate file per age)
    file_path = files[0]
    print(f"Loading {age} data from: {file_path}")
    
    # Read the CSV file
    df_temp = pd.read_csv(file_path, index_col=0)
    
    # Extract counts - the first column should be the target type, second is # Cells
    # Handle different possible column names
    if '# Cells' in df_temp.columns:
        counts_col = '# Cells'
    elif 'Cells' in df_temp.columns:
        counts_col = 'Cells'
    else:
        counts_col = df_temp.columns[0]
    
    # Create a dictionary mapping target type to count
    # Normalize target type strings to match our expected format
    target_counts = {}
    for idx, row in df_temp.iterrows():
        target_type = str(idx).strip()
        # Normalize target type format (handle variations like "1 target" vs "1 targets")
        # The CSV might have "1 target" but we expect "1 target" or "1 targets"
        count = int(row[counts_col])
        if count > 0:  # Only store non-zero counts
            target_counts[target_type] = count
    
    # Store counts for this age
    age_data[age] = target_counts

# Build the data dictionary with all target types (1-7)
# Fill missing target types with 0
data_dict = {'type': target_types}

for age in age_groups:
    counts = []
    total_cells = sum(age_data[age].values())
    
    for target_type in target_types:
        # Try exact match first
        count = age_data[age].get(target_type, 0)
        
        # If no exact match, try case-insensitive and flexible matching
        if count == 0:
            # Try matching without case sensitivity
            for key, value in age_data[age].items():
                if key.lower().strip() == target_type.lower().strip():
                    count = value
                    break
            # If still no match, try matching just the number (e.g., "1 target" matches "1 targets")
            if count == 0:
                target_num = target_type.split()[0]  # Extract number
                for key, value in age_data[age].items():
                    if key.strip().startswith(target_num):
                        count = value
                        break
        
        # Calculate percentage
        if total_cells > 0:
            percentage = (count / total_cells) * 100
        else:
            percentage = 0.0
        counts.append(percentage)
    
    data_dict[age] = counts

# Create DataFrame
df = pd.DataFrame(data_dict)
df.set_index('type', inplace=True)

print("\nLoaded data:")
print(df)
print(f"\nTotal cells per age:")
for age in age_groups:
    total = sum(age_data[age].values())
    print(f"  {age}: {total} cells")

# --- Step 1: Chi-square test of independence ---
df_counts = (df / 100 * 1000).round().astype(int)

# Handle chi-square test with zero values
# Add small pseudocount to avoid zero expected frequencies
df_counts_pseudo = df_counts + 1
try:
    chi2, pval, dof, expected = chi2_contingency(df_counts_pseudo)
    print(f"\nChi-square test (with pseudocount): χ² = {chi2:.4f}, p = {pval:.4e}, df = {dof}")
except ValueError as e:
    print(f"\nWarning: Chi-square test failed due to zero expected frequencies: {e}")
    print("Using pseudocount-adjusted values for test")
    # Try with larger pseudocount
    df_counts_pseudo = df_counts + 10
    chi2, pval, dof, expected = chi2_contingency(df_counts_pseudo)
    print(f"Chi-square test (with larger pseudocount): χ² = {chi2:.4f}, p = {pval:.4e}, df = {dof}")

# For standardized residuals, use original counts
observed = df_counts.values

# Get script directory for saving outputs
SCRIPT_DIR = Path(__file__).parent
if args.helper_output_dir:
    OUTPUT_DIR = Path(args.helper_output_dir)
else:
    OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "04_proportions_over_time_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_df = pd.DataFrame({
    'Chi2_statistic': [chi2],
    'p_value': [pval],
    'degrees_of_freedom': [dof]
})
summary_df.to_csv(OUTPUT_DIR / "chi_square_summary.csv", index=False)
df.to_csv(OUTPUT_DIR / "compositional_proportions.csv")

# --- Step 2: Visualizations ---

# Stacked bar plot
df.T.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.ylabel('Proportion (%)')
plt.title('Distribution of Target Types Across Ages')
plt.xticks(rotation=0)
plt.legend(title='Target Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "proportion_plot.png", dpi=300)
plt.close()

# Line plot
df.T.plot(figsize=(8, 5), marker='o')
plt.title("Proportion of Each Target Type Across Development")
plt.ylabel("Proportion (%)")
plt.xlabel("Age")
plt.grid(True)
plt.legend(title="Target Type", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "proportion_line_plot.png", dpi=300)
plt.close()

# --- Step 3: CLR transformation ---
def clr_transform(x):
    """Centered log-ratio transformation (expects 1D array or pandas Series)."""
    x = np.asarray(x)
    # Replace zeros with a small value to avoid log(0)
    # Use a value smaller than the smallest non-zero value
    min_nonzero = np.min(x[x > 0]) if np.any(x > 0) else 1.0
    epsilon = min_nonzero * 1e-6
    x = np.where(x == 0, epsilon, x)
    geometric_mean = np.exp(np.mean(np.log(x)))
    return np.log(x / geometric_mean)

# Only perform CLR transformation if we have non-zero data
if not df.empty and df.sum().sum() > 0:
    clr_df = df.apply(clr_transform, axis=0).T  # rows = ages, cols = clr(target types)
    clr_df['age'] = clr_df.index
    clr_df.to_csv(OUTPUT_DIR / "clr_transformed_data.csv", index=False)
else:
    print("Warning: Cannot perform CLR transformation - no data or all zeros")
    clr_df = pd.DataFrame()

# --- Step 4: PCA of CLR data ---
if not clr_df.empty and len(clr_df) > 1:
    clr_only = clr_df.drop(columns='age')
    ages = clr_df['age'].astype(str)  # Ensure age is string for PCA hue
    
    # Check if we have enough data points for PCA (need at least 2)
    if len(clr_only) >= 2 and clr_only.shape[1] > 0:
        pca = PCA(n_components=min(2, len(clr_only), clr_only.shape[1]))
        components = pca.fit_transform(clr_only)
        expl_var = pca.explained_variance_ratio_ * 100
        
        pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
        pca_df['age'] = ages.values
        pca_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(7, 5))
        if len(pca_df.columns) >= 3:  # PC1, PC2, age
            ax = sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='age', s=100)
            
            # Fix legend display
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                plt.legend(title='Age', bbox_to_anchor=(1.05, 1))
            else:
                print("No legend labels found — check 'age' assignment in PCA dataframe.")
            
            pc1_var = expl_var[0] if len(expl_var) > 0 else 0
            pc2_var = expl_var[1] if len(expl_var) > 1 else 0
            plt.title(f"CLR PCA of Target Composition (PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}%)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "clr_pca_plot.png", dpi=300)
            plt.close()
        else:
            print("Warning: Cannot create PCA plot - insufficient components")
        
        # Save PCA scores and loadings
        pca_df.to_csv(OUTPUT_DIR / "clr_pca_scores.csv", index=False)
        if len(pca.components_) > 0:
            loadings = pd.DataFrame(pca.components_.T, 
                                    index=clr_only.columns,
                                    columns=[f'PC{i+1}_loading' for i in range(pca.components_.shape[0])])
            loadings.to_csv(OUTPUT_DIR / "clr_pca_loadings.csv")
    else:
        print("Warning: Cannot perform PCA - insufficient data points or features")
else:
    print("Warning: Cannot perform PCA - CLR transformation failed or insufficient data")

# --- Step 5: Standardized residuals from Chi-square ---
# Use the expected values from the chi-square test above
# Since we used pseudocount, we need to adjust expected back proportionally
# to match the original observed totals
observed_totals = observed.sum(axis=0)
expected_totals = expected.sum(axis=0)

# Avoid division by zero
scaling_factors = np.where(expected_totals > 0, observed_totals / expected_totals, 1.0)
expected_scaled = expected * scaling_factors

# Calculate standardized residuals
std_residuals = (observed - expected_scaled) / np.sqrt(np.maximum(expected_scaled, 1))
resid_df = pd.DataFrame(std_residuals, index=df_counts.index, columns=df_counts.columns)
resid_df.to_csv(OUTPUT_DIR / "chi_square_standardized_residuals.csv")

# Only create heatmap if we have valid data
if not resid_df.empty and not resid_df.isna().all().all():
    plt.figure(figsize=(8, 5))
    sns.heatmap(resid_df, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
                cbar_kws={'label': 'Standardized Residual'})
    plt.title("Standardized Residuals from Chi-square Test")
    plt.ylabel("Target Type")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "chi_square_residuals_heatmap.png", dpi=300)
    plt.close()
else:
    print("Warning: Cannot create residuals heatmap - insufficient data")
