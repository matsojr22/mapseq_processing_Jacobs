# MAPseq Processing Pipeline: Script Analysis and Stability Framework

## Overview

This pipeline processes MAPseq projection data across developmental timepoints (P3, P12, P20, P60) to identify projection motifs and analyze their changes over time. The central question is: **Do results change significantly over time (age groups)?**

The pipeline consists of:
- **Main script**: `process-nbcm-tsv.py` - Per-sample processing and statistical testing
- **Helper scripts** (01-13): Cross-age analyses, temporal comparisons, and visualization

---

## Main Script: `process-nbcm-tsv.py`

### Purpose
Per-sample processing of NBCM (neural barcode count matrix) data with dual-model statistical analysis.

### Key Functions
1. **Data Filtering**: Filters cells based on injection UMI thresholds, target counts, and body-to-target ratios
2. **Normalization**: Normalizes projection matrices per cell
3. **Dual-Model Probability Calculation**:
   - **Uniform Model**: Single edge probability (pₑ) for all regions
   - **Region-Specific Model**: Region-specific projection probabilities
4. **Statistical Testing**: Binomial tests for motif over/under-representation
5. **Effect Size Calculation**: Computes effect sizes for each motif
6. **Visualization**: Generates plots for effect significance, upsetplots, per-cell projection strength, and 2-region motif graphs

### Statistical Tests
- **Binomial Test** (two-tailed): Tests if observed motif counts differ from expected (using N0 estimated population)
- **Bonferroni Correction**: Multiple testing correction for motif significance

### Key Outputs for Stability Analysis

**Location**: `02_output/{age}/{parameterization}/analysis/{model}/`

1. **`*_upsetplot_uniform.csv` / `*_upsetplot_region_specific.csv`**
   - Columns: `Motifs`, `Observed`, `Expected`, `Expected SD`, `Effect Size`, `P-value`, `Degree`, `Group`
   - **Purpose**: Core data for each motif's representation at each age
   - **Use for**: Comparing motif frequencies across ages, identifying significant motifs

2. **`*_effect_significance_uniform.png` / `*_effect_significance_region_specific.png`**
   - Volcano plots showing effect size vs. p-value
   - **Purpose**: Visual identification of significantly over/under-represented motifs
   - **Use for**: Quick visual assessment of significant changes

3. **`*_per_cell_proj_strength_uniform.png` / `*_per_cell_proj_strength_region_specific.png`**
   - Individual cell projection patterns
   - **Purpose**: Visualize heterogeneity in projection patterns
   - **Use for**: Understanding cell-level variation

4. **`*_panel_g_broadcasting_from_canonical_uniform.svg` / `*_panel_g_broadcasting_from_canonical_region_specific.svg`**
   - Network graphs showing 2-region broadcasting motifs
   - **Purpose**: Visual representation of pairwise motif relationships with significance coloring
   - **Features**: Nodes represent regions (RSP, PM, AM, AL, LM), edges represent 2-region motifs, colored by significance (red=overrepresented, blue=underrepresented, black=not significant)
   - **Use for**: Understanding relationships between pairs of target regions and identifying significant 2-region multiplexing patterns

5. **`projection_summary.csv`**
   - Comprehensive summary metrics for each sample
   - **Location**: `analysis/{model}/projection_summary.csv` (separate files for uniform and region-specific models)
   - **Columns**: Sample, Model, injection min, target UMI min, filtering parameters, p_e, N0, consensus_k, motif counts, and other summary statistics
   - **Purpose**: Aggregated metrics for cross-sample and cross-age comparisons
   - **Use for**: High-level summary statistics and parameter tracking

### Questions Answered
- Which motifs are significantly over/under-represented at each age?
- What is the statistical significance of motif representation?
- How do uniform and region-specific models compare?

---

## Helper Scripts

### Helper Script 01: `01_motif_analysis_per_animal.py`

**Purpose**: High-level statistical analysis of motif frequency changes across timepoints, accounting for individual animal variation.

**Statistical Tests**:
- **Kruskal-Wallis Test**: Non-parametric test comparing motif frequencies across 4 timepoints (P3, P12, P20, P60)
  - Null hypothesis: No difference in motif frequency distributions across ages
  - Interpretation: p < 0.05 indicates significant change over time
  - Advantages: Non-parametric, handles non-normal data, multiple groups
- **Jensen-Shannon Divergence (JSD)**: Measures distribution similarity between ages
  - Range: 0 (identical) to 1 (maximally different)
  - Interpretation: JSD > 0.05 suggests meaningful difference
  - Advantages: Symmetric, bounded, interpretable
- **Global vs Domain-wise Normalization**: 
  - Global: Compares entire frequency distributions across all motifs
  - Domain-wise: Compares within each motif complexity level (degree)

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/01_motif_analysis_per_animal/{model}/`
- Bar plots with SEM across animals per timepoint
- Kruskal-Wallis results (CSV/text)
- JSD comparison matrices
- Cross-age comparison plots (global and domain-wise normalization)

**Questions Answered**:
- Do motif frequencies change significantly across ages? (Kruskal-Wallis)
- Which motifs show the most dramatic changes? (Effect sizes, JSD)
- Are changes consistent across animals? (SEM, individual animal data)
- Are changes uniform across all complexity levels? (Domain-wise analysis)

**Most Important for Stability Analysis**: ✅ **PRIMARY SCRIPT** - Directly tests temporal stability

---

### Helper Script 07: `07_motif_significange_trajectories.py`

**Purpose**: Motif-level analysis of effect size trajectories across developmental stages.

**Statistical Tests**:
- **Fisher's Exact Test**: Tests significance of transitions between consecutive stages
  - Null hypothesis: No change in motif representation between stages
  - Interpretation: p < 0.05 indicates significant transition
  - Advantages: Exact test, good for small sample sizes
- **Effect Size Tracking**: Tracks effect sizes across P3→P12→P20→P60

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/07_motif_significange_trajectories/{model}/`
- `combined_effect_sizes_uniform.csv` / `combined_effect_sizes_region_specific.csv`
  - Columns: `Motif_Label`, `Effect Size`, `Stage`, `Significant`, `Observed`
- `transition_significance.csv`
  - Columns: `Motif`, `Transition`, `P-value`, `Significant`
- Trajectory plots (PDF): Effect size over time for each motif

**Questions Answered**:
- Which motifs show significant transitions between stages? (Fisher's exact test)
- Do effect sizes increase or decrease over time? (Trajectory plots)
- Are transitions gradual or abrupt? (Trajectory shape)
- Which specific stage transitions are most significant? (Transition significance CSV)

**Most Important for Stability Analysis**: ✅ **PRIMARY SCRIPT** - Identifies when changes occur

---

### Helper Script 05: `05_motif_analysis.py`

**Purpose**: Cross-age motif percentage analysis with distribution comparisons.

**Statistical Tests**:
- **Welch's t-test**: Compares motif percentages between two ages (handles unequal variances)
- **Kolmogorov-Smirnov Test**: Compares entire distributions between ages
  - Null hypothesis: Distributions are identical
  - Advantages: Non-parametric, sensitive to shape differences
- **Jensen-Shannon Divergence**: Distribution similarity metric
- **Hierarchical Clustering**: Groups motifs by temporal patterns
- **PCA + KMeans**: Dimensionality reduction and clustering of temporal patterns

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/05_motif_analysis/`
- `motif_percent_matrix_by_age_uniform.csv` / `motif_percent_matrix_by_age_region_specific.csv`
  - Rows: Motifs, Columns: Ages (P12, P20, P60)
  - Values: Percentage of observed motifs
- `motif_transition_significance_summary_uniform.txt` / `motif_transition_significance_summary_region_specific.txt`
  - Per-motif transition comparisons with JSD and significance flags
- `histogram_similarity_summary_uniform.txt` / `histogram_similarity_summary_region_specific.txt`
  - Degree-wise comparisons with Welch's t-test, KS test, and JSD
- PDF plots: Motif percentage barplots by age (sorted and by rank)
- Clustering heatmaps and PCA plots

**Questions Answered**:
- How do motif percentages change across ages? (Percentage matrix)
- Which transitions show significant distribution changes? (Transition significance)
- Are changes consistent across motif complexity levels? (Degree-wise analysis)

**Most Important for Stability Analysis**: ✅ **SECONDARY SCRIPT** - Provides detailed transition analysis

---

### Helper Script 06: `06_all_motif_divergence.py`

**Purpose**: Visualize divergence patterns across age transitions.

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/06_all_motif_divergence/{model}/`
- `divergence_{transition}_{model}.svg`: Bar plots of JS divergence per transition
  - Shows significant (red) vs non-significant (blue) divergences
  - Transitions: P12vsP20, P20vsP60, P12vsP60

**Questions Answered**:
- Which motifs show high divergence between specific transitions?
- Are divergences consistent with statistical significance?

**Most Important for Stability Analysis**: ⚠️ **VISUALIZATION ONLY** - Complements script 05

---

### Helper Script 04: `04_proportions_over_time_stats.py`

**Purpose**: Analyze proportions of target types (1-7 targets) across ages.

**Statistical Tests**:
- **Chi-square Test of Independence**: Tests if target type proportions are independent of age
  - Null hypothesis: Proportions are independent of age
  - Advantages: Good for categorical/count data
- **CLR (Centered Log-Ratio) Transformation**: Compositional data transformation
- **PCA on CLR Data**: Dimensionality reduction of compositional changes

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/04_proportions_over_time_stats/`
- `chi_square_summary.csv`: Chi-square test results
- `compositional_proportions.csv`: Proportions by age and target type
- `clr_transformed_data.csv`: CLR-transformed data
- Stacked bar plots and line plots of proportions
- CLR PCA plot
- Chi-square residuals heatmap

**Questions Answered**:
- Do target type proportions change significantly across ages? (Chi-square)
- Which target types show the largest changes? (Residuals heatmap)
- How do compositional patterns evolve? (CLR PCA)

**Most Important for Stability Analysis**: ⚠️ **DESCRIPTIVE** - Analyzes target count distributions, not motif-specific

---

### Helper Script 02: `02_projection_analysis.py`

**Purpose**: PCA and clustering analysis of projection patterns (not directly temporal).

**Statistical Tests**: None (descriptive analysis)

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/02_projection_analysis/`
- PCA plots comparing age groups
- Hierarchical clustering dendrograms
- Heatmaps of projection patterns

**Questions Answered**:
- How do projection patterns differ between age groups?
- What are the main sources of variation? (PCA)

**Most Important for Stability Analysis**: ❌ **NOT TEMPORAL** - Cross-age comparisons only

---

### Helper Script 03: `03_composition.py`

**Purpose**: Composition analysis by age (UMI and projection counts).

**Statistical Tests**: None (descriptive)

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/03_composition/`
- UMI composition plots by age
- Projection count composition plots by age

**Questions Answered**:
- How do UMI and projection counts vary by age?
- What is the composition of projections across regions?

**Most Important for Stability Analysis**: ⚠️ **DESCRIPTIVE** - Provides context but not statistical tests

---

### Helper Script 08: `08_motif_clustering.py`

**Purpose**: Clustering motifs based on effect size trajectories from script 07.

**Statistical Tests**: None (clustering analysis)

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/08_motif_clustering/{model}/`
- Dendrograms
- Clustering heatmaps
- Motif ordering analysis
- Evaluation metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)

**Questions Answered**:
- Which motifs have similar temporal trajectories?
- How can motifs be grouped by developmental patterns?

**Most Important for Stability Analysis**: ⚠️ **EXPLORATORY** - Groups motifs but doesn't test stability

---

### Helper Script 09: `09_plot_normalized_projection_strength_data.py`

**Purpose**: Visualize normalized projection strength data.

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/09_plot_normalized_projection_strength_data/`
- Projection strength plots (individual and aggregate)

**Questions Answered**:
- How do projection strengths vary across ages?

**Most Important for Stability Analysis**: ⚠️ **VISUALIZATION ONLY**

---

### Helper Script 13: `13_aggregate_projection_summaries.py`

**Purpose**: Aggregate projection summary data across all parameterizations.

**Key Outputs**:
- **Location**: `02_output/{parameterization}_helpers/13_aggregate_projection_summaries/`
- Combined projection summary CSV

**Questions Answered**:
- What are the overall projection statistics across all samples?

**Most Important for Stability Analysis**: ❌ **AGGREGATION ONLY** - Not for temporal analysis

---

## Stability Over Time: Key Data Sources

### From Each Age Group (P3, P12, P20, P60)

**Location**: `02_output/{age}/{parameterization}/analysis/{model}/`

1. **Aggregate upsetplot files** (`*_ALL_*_filters_upsetplot_*.csv`)
   - **Columns**: `Motifs`, `Observed`, `Expected`, `Effect Size`, `P-value`, `Degree`, `Group`
   - **Purpose**: Core motif data for each age
   - **Key metrics**: Effect Size (magnitude of change), P-value (significance), Observed vs Expected (representation)

2. **Effect significance plots** (`*_effect_significance_*.png`)
   - **Purpose**: Visual identification of significant motifs
   - **Use**: Quick assessment of which motifs are significant at each age

3. **Per-cell projection strength** (`*_per_cell_proj_strength_*.png`)
   - **Purpose**: Individual cell patterns
   - **Use**: Understanding heterogeneity and cell-level variation

4. **2-region motif graphs** (`*_panel_g_broadcasting_from_canonical_*.svg`)
   - Network graphs showing pairwise relationships between target regions
   - **Purpose**: Visual representation of 2-region broadcasting motifs with significance coloring
   - **Use**: Understanding relationships between pairs of target regions and identifying significant 2-region multiplexing patterns

5. **Projection summary files** (`projection_summary.csv`)
   - Comprehensive summary metrics for each sample
   - **Purpose**: Aggregated metrics for cross-sample and cross-age comparisons
   - **Use**: High-level summary statistics and parameter tracking

### Cross-Age Analysis Files

1. **Script 01 Outputs** (PRIMARY)
   - **Location**: `02_output/{parameterization}_helpers/01_motif_analysis_per_animal/{model}/`
   - Kruskal-Wallis test results (CSV/text)
   - JSD matrices
   - Bar plots with SEM across animals
   - **Key file**: Kruskal-Wallis results table

2. **Script 07 Outputs** (PRIMARY)
   - **Location**: `02_output/{parameterization}_helpers/07_motif_significange_trajectories/{model}/`
   - `combined_effect_sizes_*.csv`: Effect sizes per motif per stage
   - `transition_significance.csv`: P-values for each transition
   - Trajectory plots (PDF)
   - **Key files**: Both CSVs are critical

3. **Script 05 Outputs** (SECONDARY)
   - **Location**: `02_output/{parameterization}_helpers/05_motif_analysis/`
   - **Note**: Files use model-specific naming (`*_uniform.csv`, `*_region_specific.csv`) but are stored in the main directory
   - `motif_percent_matrix_by_age_*.csv`: Motif percentages by age
   - `motif_transition_significance_summary_*.txt`: Detailed transition analysis
   - **Key file**: Percentage matrix for quantitative comparisons

4. **Script 06 Outputs** (VISUALIZATION)
   - **Location**: `02_output/{parameterization}_helpers/06_all_motif_divergence/{model}/`
   - Divergence plots per transition
   - **Key file**: SVG plots showing divergence patterns

---

## Recommended Statistical Tests for Stability Analysis

### 1. Kruskal-Wallis Test (Script 01) ⭐ PRIMARY

**Use case**: Compare motif frequencies across 4 timepoints (P3, P12, P20, P60)

**Null hypothesis**: No difference in motif frequency distributions across ages

**Interpretation**: 
- p < 0.05: Significant change over time
- p ≥ 0.05: No significant change (stable)

**Advantages**: 
- Non-parametric (no normality assumption)
- Handles non-normal data
- Multiple groups (4 ages)
- Accounts for individual animal variation

**When to use**: Primary test for overall stability question

---

### 2. Fisher's Exact Test (Script 07) ⭐ PRIMARY

**Use case**: Test significance of transitions between consecutive stages (P3→P12, P12→P20, P20→P60)

**Null hypothesis**: No change in motif representation between stages

**Interpretation**: 
- p < 0.05: Significant transition
- p ≥ 0.05: No significant change between these stages

**Advantages**: 
- Exact test (no approximation)
- Good for small sample sizes
- Tests specific transitions (when changes occur)

**When to use**: To identify which specific transitions show changes

---

### 3. Jensen-Shannon Divergence (Scripts 01, 05, 06) ⭐ SECONDARY

**Use case**: Measure distribution similarity between ages

**Range**: 0 (identical) to 1 (maximally different)

**Interpretation**: 
- JSD < 0.05: Very similar distributions (stable)
- JSD 0.05-0.2: Moderate difference
- JSD > 0.2: Large difference (unstable)

**Advantages**: 
- Symmetric (order-independent)
- Bounded (0-1)
- Interpretable magnitude
- Can compare any two ages

**When to use**: To quantify magnitude of changes between specific age pairs

---

### 4. Welch's t-test (Script 05) ⚠️ SUPPLEMENTARY

**Use case**: Compare motif percentages between two ages

**Null hypothesis**: No difference in means

**Interpretation**: 
- p < 0.05: Significant difference in means
- p ≥ 0.05: No significant difference

**Advantages**: 
- Handles unequal variances
- Parametric (more power if data is normal)

**When to use**: For pairwise comparisons when data may be normal

---

### 5. Kolmogorov-Smirnov Test (Script 05) ⚠️ SUPPLEMENTARY

**Use case**: Compare entire distributions between ages

**Null hypothesis**: Distributions are identical

**Interpretation**: 
- p < 0.05: Distributions differ significantly
- p ≥ 0.05: Distributions are similar

**Advantages**: 
- Non-parametric
- Sensitive to shape differences (not just means)

**When to use**: When interested in distribution shape, not just means

---

### 6. Chi-square Test (Script 04) ⚠️ DESCRIPTIVE

**Use case**: Test independence of target type proportions across ages

**Null hypothesis**: Proportions are independent of age

**Interpretation**: 
- p < 0.05: Proportions depend on age (changes over time)
- p ≥ 0.05: Proportions independent of age (stable)

**Advantages**: 
- Good for categorical/count data
- Tests overall pattern

**When to use**: For target count distributions, not motif-specific analysis

---

## Most Important Plots/DataFrames for Stability Analysis

### From Each Age Group

1. **Aggregate upsetplot CSV** (`*_ALL_*_filters_upsetplot_*.csv`)
   - **Why**: Contains observed/expected counts, effect sizes, and p-values for all motifs
   - **Examine**: 
     - Effect Size column: Magnitude of over/under-representation
     - P-value column: Statistical significance
     - Observed vs Expected: Raw representation
   - **Compare across ages**: Look for motifs with changing effect sizes or significance

2. **Effect significance plots** (`*_effect_significance_*.png`)
   - **Why**: Visual summary of significant motifs
   - **Examine**: Which motifs are in significant regions (top/bottom) at each age
   - **Compare across ages**: Do the same motifs remain significant?

### Cross-Age Analysis

1. **Script 01: Kruskal-Wallis Results** ⭐ MOST IMPORTANT
   - **Why**: Direct statistical test of temporal stability
   - **Examine**: 
     - P-values for each motif
     - H-statistics (effect size)
     - Motifs with p < 0.05 show significant changes
   - **Interpretation**: Low p-value = unstable, high p-value = stable

2. **Script 01: Bar plots with SEM**
   - **Why**: Visual representation of frequency changes with error bars
   - **Examine**: 
     - Trends across ages
     - Overlap of error bars (indicates stability)
     - Motifs with large SEM (high variability)

3. **Script 07: Combined effect sizes CSV** ⭐ MOST IMPORTANT
   - **Why**: Effect size trajectories for each motif
   - **Examine**: 
     - Effect Size column across Stage column
     - Trends: increasing, decreasing, or stable
     - Magnitude of changes
   - **Compare**: Uniform vs region-specific models

4. **Script 07: Transition significance CSV** ⭐ MOST IMPORTANT
   - **Why**: Identifies which transitions are significant
   - **Examine**: 
     - P-value for each Transition (P3_to_P12, P12_to_P20, P20_to_P60)
     - Motifs with significant transitions
   - **Interpretation**: Which stage transitions show the most change?

5. **Script 05: Motif percentage matrix** ⚠️ SECONDARY
   - **Why**: Quantitative percentages for each motif at each age
   - **Examine**: 
     - Percentage changes across columns (ages)
     - Motifs with large percentage swings
   - **Calculate**: Percent change between ages

6. **Script 05: Transition significance summary** ⚠️ SECONDARY
   - **Why**: Detailed transition analysis with JSD
   - **Examine**: 
     - JSD values for each transition
     - Significance flags
   - **Interpretation**: Magnitude and significance of changes

---

## Interpretation Guidelines

### Overall Stability Assessment

1. **Start with Script 01 Kruskal-Wallis results**:
   - Count motifs with p < 0.05: High count = unstable system
   - Count motifs with p ≥ 0.05: High count = stable system
   - Look at distribution of p-values: Many low p-values = unstable

2. **Examine Script 07 transition significance**:
   - Which transitions have the most significant changes?
   - Are changes concentrated in specific transitions (e.g., early vs late)?
   - Count significant transitions per motif

3. **Check effect size magnitudes** (Script 07):
   - Large effect sizes (>1 or <-1) indicate substantial changes
   - Small effect sizes (<0.5) indicate minor changes
   - Look for consistent trends (increasing/decreasing)

4. **Compare models** (Uniform vs Region-Specific):
   - Do both models agree on stability?
   - Are there motifs where models disagree?
   - Agreement suggests robust findings

### Motif-Specific Stability

1. **Stable motifs** (p ≥ 0.05 in Kruskal-Wallis):
   - Consistent representation across ages
   - Small effect size changes
   - No significant transitions

2. **Unstable motifs** (p < 0.05 in Kruskal-Wallis):
   - Changing representation across ages
   - Large effect size changes
   - Significant transitions

3. **Early-changing motifs** (significant P3→P12 transition):
   - Changes occur early in development
   - May stabilize later

4. **Late-changing motifs** (significant P20→P60 transition):
   - Changes occur late in development
   - May be maturation effects

### Magnitude of Changes

1. **JSD interpretation**:
   - < 0.05: Very stable (minimal change)
   - 0.05-0.1: Moderately stable (small changes)
   - 0.1-0.2: Moderately unstable (noticeable changes)
   - > 0.2: Unstable (large changes)

2. **Effect size interpretation**:
   - |Effect Size| < 0.5: Small change
   - |Effect Size| 0.5-1.0: Moderate change
   - |Effect Size| > 1.0: Large change

---

## Summary: Answering "Do Results Change Significantly Over Time?"

### Primary Evidence (Scripts 01 & 07)

1. **Script 01 Kruskal-Wallis Test**:
   - **Question**: Do motif frequencies differ across ages?
   - **Answer**: Count motifs with p < 0.05
   - **Interpretation**: Many significant = unstable, few significant = stable

2. **Script 07 Transition Significance**:
   - **Question**: Which stage transitions show changes?
   - **Answer**: Count significant transitions (p < 0.05)
   - **Interpretation**: Many transitions = unstable, few transitions = stable

3. **Script 07 Effect Size Trajectories**:
   - **Question**: How large are the changes?
   - **Answer**: Examine effect size magnitudes and trends
   - **Interpretation**: Large effect sizes = substantial changes

### Secondary Evidence (Scripts 05 & 06)

1. **Script 05 Percentage Changes**:
   - **Question**: What are the quantitative changes?
   - **Answer**: Calculate percent change from percentage matrix
   - **Interpretation**: Large percent changes = unstable

2. **Script 06 Divergence Patterns**:
   - **Question**: Which transitions show high divergence?
   - **Answer**: Examine JSD values in plots
   - **Interpretation**: High JSD = unstable transition

### Recommended Analysis Workflow

1. **Start with Script 01**: Get overall Kruskal-Wallis results
2. **Examine Script 07**: Identify which transitions are significant
3. **Check Effect Sizes**: Quantify magnitude of changes
4. **Compare Models**: Verify consistency between uniform and region-specific
5. **Detail with Script 05**: Get specific percentage changes
6. **Visualize with Script 06**: See divergence patterns

### Key Files to Extract

1. Script 01: Kruskal-Wallis results table
2. Script 07: `combined_effect_sizes_*.csv` and `transition_significance.csv`
3. Script 05: `motif_percent_matrix_by_age_*.csv`
4. Script 05: `motif_transition_significance_summary_*.txt`
5. Age-specific: Aggregate upsetplot CSVs from each age

---

## Model Comparison

The pipeline uses two models:
- **Uniform Model**: Single edge probability (pₑ) for all regions
- **Region-Specific Model**: Region-specific projection probabilities

**For stability analysis**: Compare results from both models
- **Agreement**: Both models show same stability patterns → robust finding
- **Disagreement**: Models differ → investigate further (may indicate model sensitivity)

**Key files to compare**:
- Script 01 outputs: `{model}/` subdirectories
- Script 07 outputs: `{model}/` subdirectories
- All other scripts: Model-specific outputs

---

## Conclusion

To answer "Do results change significantly over time?":

1. **Primary evidence**: Script 01 (Kruskal-Wallis) and Script 07 (transitions)
2. **Key metrics**: P-values, effect sizes, JSD values
3. **Key files**: Kruskal-Wallis results, transition significance CSV, effect sizes CSV
4. **Interpretation**: Many significant results = unstable, few significant = stable
5. **Validation**: Compare uniform and region-specific models

The most important analyses are:
- **Script 01**: Overall temporal stability test
- **Script 07**: Specific transition identification
- **Script 05**: Detailed quantitative changes
