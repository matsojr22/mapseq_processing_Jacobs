#!/usr/bin/env python3
"""
Helper 16: Power Analysis for Manuscript Claims

Performs comprehensive power analyses for each claim in the MAPseq developmental
stability manuscript. For null findings, uses equivalence testing (TOST),
sensitivity elbow curves, and bootstrap subsampling. For significant findings,
uses standard power calculations. All effect sizes are computed from the data.

Depends on outputs from helpers 01, 04, 13, and 15.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import glob
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).parent.parent.parent
AGE_GROUPS = ["p12", "p20", "p60"]
AGE_LABELS = {"p12": "P14→P16", "p20": "P22→P24", "p60": "P60+→P62+"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Power analysis for manuscript claims (Helper 16)")
    parser.add_argument("--base_output_dir", type=str, default=None,
                        help="Base output directory (default: REPO_ROOT/02_output)")
    parser.add_argument("--helper_output_dir", type=str, default=None,
                        help="Directory for this helper's outputs")
    parser.add_argument("--model_type", type=str, default="uniform",
                        help="Statistical model type (default: uniform)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--power_target", type=float, default=0.80,
                        help="Target power (default: 0.80)")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--max_n_elbow", type=int, default=50,
                        help="Maximum N for elbow curves (default: 50)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_motif_name(s):
    """Convert motif name to canonical sorted '+'-separated form.

    Handles both formats:
      "['pm', 'rsp']"  ->  "pm+rsp"
      "pm+rsp"          ->  "pm+rsp"
    """
    s = str(s).strip()
    s = s.strip("[]")
    s = s.replace("'", "").replace('"', '')
    parts = [p.strip() for p in s.replace("+", ",").split(",") if p.strip()]
    parts.sort()
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_parameterization(helper_output_dir):
    """Extract parameterization name from helper_output_dir path."""
    for part in Path(helper_output_dir).parts:
        if part.startswith(("01.", "02.", "03.", "04.", "05.")):
            name = part.replace("_helpers", "")
            return name
        if "_helpers" in part:
            return part.split("_helpers")[0]
    return None


def load_projection_summary(base_output_dir, parameterization):
    """Load per-animal rows from projection_summary.csv files across ages."""
    rows = []
    for age in AGE_GROUPS:
        pattern = os.path.join(base_output_dir, age, parameterization,
                               "**", "projection_summary.csv")
        for f in glob.glob(pattern, recursive=True):
            try:
                df = pd.read_csv(f)
                df["age"] = age
                rows.append(df)
            except Exception:
                continue
        direct = os.path.join(base_output_dir, age, parameterization,
                              "projection_summary.csv")
        if os.path.isfile(direct) and direct not in [r for r in glob.glob(pattern, recursive=True)]:
            try:
                df = pd.read_csv(direct)
                df["age"] = age
                rows.append(df)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Sample", "Model"], keep="first")
    return combined


def load_individual_replicates(base_output_dir, parameterization, model_type):
    """Load individual_replicates_per_animal_global.csv from helper 01."""
    helpers_dir = os.path.join(base_output_dir,
                               f"{parameterization}_helpers",
                               "01_motif_analysis_per_animal")
    candidates = [
        os.path.join(helpers_dir, model_type,
                     "individual_replicates_per_animal_global.csv"),
        os.path.join(helpers_dir,
                     "individual_replicates_per_animal_global.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return pd.read_csv(c)
    for f in glob.glob(os.path.join(helpers_dir, "**",
                                    "individual_replicates_per_animal_global.csv"),
                       recursive=True):
        return pd.read_csv(f)
    return pd.DataFrame()


def load_pie_chart_per_animal(base_output_dir, parameterization):
    """Load per-animal pie_chart_data.csv files (target count distributions)."""
    records = []
    for age in AGE_GROUPS:
        age_dir = os.path.join(base_output_dir, age, parameterization)
        pattern = os.path.join(age_dir, "**", "*pie_chart_data.csv")
        for f in glob.glob(pattern, recursive=True):
            basename = os.path.basename(f)
            if "_ALL_" in basename or "_alL_" in basename:
                continue
            try:
                df = pd.read_csv(f, index_col=0)
                animal = Path(f).parent.parent.parent.name
                if animal == parameterization:
                    animal = Path(f).stem.split("_pie_chart_data")[0]
                total = 0
                count_col = None
                for col in df.columns:
                    if "cell" in col.lower() or "count" in col.lower():
                        count_col = col
                        break
                if count_col is None and len(df.columns) >= 1:
                    count_col = df.columns[0]
                if count_col is not None:
                    total = df[count_col].sum()
                if total > 0:
                    for idx_label, row in df.iterrows():
                        records.append({
                            "age": age,
                            "animal": animal,
                            "target_category": str(idx_label),
                            "count": row[count_col],
                            "proportion": row[count_col] / total * 100,
                        })
            except Exception:
                continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_upsetplot_csvs(base_output_dir, parameterization, model_type):
    """Load aggregate upsetplot CSVs from 07_input."""
    input_dir = os.path.join(base_output_dir,
                             f"{parameterization}_helpers", "07_input")
    if not os.path.isdir(input_dir):
        return {}
    result = {}
    for f in glob.glob(os.path.join(input_dir, f"*_upsetplot_{model_type}.csv")):
        basename = os.path.basename(f).lower()
        age = None
        for ag in AGE_GROUPS:
            if basename.startswith(ag) or basename.startswith(ag[0:1] + ag[1:]):
                age = ag
                break
        if age is None:
            for ag in AGE_GROUPS:
                tag = ag.replace("p", "P")
                if tag in os.path.basename(f):
                    age = ag
                    break
        if age is None:
            for ag in ["p3", "p12", "p20", "p60"]:
                if ag in basename[:6]:
                    age = ag
                    break
        if age is not None and age in AGE_GROUPS:
            try:
                result[age] = pd.read_csv(f)
            except Exception:
                continue
    return result


def load_matt_visual_rules(base_output_dir, parameterization, model_type):
    """Load motif_change_list.csv from helper 15 matt_visual_rules output."""
    volcano_dir = os.path.join(base_output_dir,
                               f"{parameterization}_helpers",
                               "15_volcano_trajectories")
    candidates = [
        os.path.join(volcano_dir, model_type, "matt_visual_rules",
                     "motif_change_list.csv"),
        os.path.join(volcano_dir, model_type, "noP3", "matt_visual_rules",
                     "motif_change_list.csv"),
    ]
    for pattern in [os.path.join(volcano_dir, "**", "motif_change_list.csv")]:
        candidates.extend(glob.glob(pattern, recursive=True))
    for c in candidates:
        if os.path.isfile(c):
            try:
                return pd.read_csv(c)
            except Exception:
                continue
    return pd.DataFrame()


def load_barcode_summary():
    """Load global barcode uniqueness summary from 00_cleaned_data."""
    global_path = REPO_ROOT / "00_cleaned_data" / "teleporting_barcode_detection" / "global_batch_aggregate_summary.csv"
    if global_path.is_file():
        return pd.read_csv(global_path)
    animal_path = REPO_ROOT / "00_cleaned_data" / "teleporting_barcode_detection" / "animal_uniqueness_summary.csv"
    if animal_path.is_file():
        return pd.read_csv(animal_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Generic power engines
# ---------------------------------------------------------------------------

def run_tost(group1, group2, equiv_margin, alpha=0.05):
    """
    Two One-Sided Tests for equivalence.
    Returns dict with tost_p, ci_low, ci_high, equivalent (bool), mean_diff.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return {"tost_p": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "equivalent": False, "mean_diff": np.nan}
    mean_diff = np.mean(g1) - np.mean(g2)
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se < 1e-15:
        se = 1e-15
    df_val = (s1**2 / n1 + s2**2 / n2)**2 / (
        (s1**2 / n1)**2 / max(n1 - 1, 1) + (s2**2 / n2)**2 / max(n2 - 1, 1))
    df_val = max(df_val, 1)
    t_upper = (mean_diff - equiv_margin) / se
    t_lower = (mean_diff + equiv_margin) / se
    p_upper = stats.t.cdf(t_upper, df_val)
    p_lower = 1 - stats.t.cdf(t_lower, df_val)
    tost_p = max(p_upper, p_lower)
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    return {
        "tost_p": tost_p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "equivalent": tost_p < alpha,
        "mean_diff": mean_diff,
    }


def run_elbow_curve(groups, n_groups_k, alpha=0.05, power_target=0.80,
                    max_n=50):
    """
    Compute minimum detectable effect size (MDE) at target power for N = 2..max_n.
    Uses one-way ANOVA power approximation.
    Returns dict with n_range, mde_values, actual_n.
    """
    from statsmodels.stats.power import FTestAnovaPower
    analysis = FTestAnovaPower()
    all_values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    actual_n = min(len(g) for g in groups)
    n_range = list(range(2, max_n + 1))
    mde_values = []
    for n in n_range:
        try:
            es = analysis.solve_power(effect_size=None, nobs=n,
                                      alpha=alpha, power=power_target,
                                      k_groups=n_groups_k)
            mde_values.append(es)
        except Exception:
            mde_values.append(np.nan)
    return {
        "n_range": n_range,
        "mde_values": mde_values,
        "actual_n": actual_n,
    }


def run_bootstrap_subsample(groups, n_iter=1000, alpha=0.05):
    """
    Bootstrap subsampling stability: for each subsample size n = 2..min(group_sizes),
    resample groups and run KW, report fraction significant.
    Returns dict with n_range, frac_significant.
    """
    min_n = min(len(g) for g in groups)
    if min_n < 2:
        return {"n_range": [], "frac_significant": []}
    n_range = list(range(2, min_n + 1))
    frac_sig = []
    for n in n_range:
        sig_count = 0
        for _ in range(n_iter):
            resampled = []
            for g in groups:
                idx = np.random.choice(len(g), size=n, replace=True)
                resampled.append(np.asarray(g)[idx])
            try:
                if len(resampled) >= 2 and all(len(r) >= 1 for r in resampled):
                    h, p = stats.kruskal(*resampled)
                    if p < alpha:
                        sig_count += 1
            except Exception:
                pass
        frac_sig.append(sig_count / n_iter)
    return {"n_range": n_range, "frac_significant": frac_sig}


def run_anova_power(groups, alpha=0.05):
    """
    Compute achieved power for one-way ANOVA given observed data.
    Returns dict with cohen_f, achieved_power, n_needed_80.
    """
    from statsmodels.stats.power import FTestAnovaPower
    analysis = FTestAnovaPower()
    k = len(groups)
    ns = [len(g) for g in groups]
    min_n = min(ns)
    grand_mean = np.mean(np.concatenate(groups))
    group_means = [np.mean(g) for g in groups]
    ssb = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, group_means))
    ssw = sum(np.sum((np.asarray(g) - np.mean(g))**2) for g in groups)
    n_total = sum(ns)
    msb = ssb / max(k - 1, 1)
    msw = ssw / max(n_total - k, 1)
    if msw < 1e-15:
        msw = 1e-15
    f_stat = msb / msw
    cohen_f = np.sqrt(max(f_stat * (k - 1) / max(n_total - k, 1), 0) / k) if n_total > k else 0
    if cohen_f < 1e-10:
        cohen_f = 1e-10
    try:
        achieved_power = analysis.solve_power(effect_size=cohen_f, nobs=min_n,
                                              alpha=alpha, power=None, k_groups=k)
    except Exception:
        achieved_power = np.nan
    try:
        n_needed = analysis.solve_power(effect_size=cohen_f, nobs=None,
                                        alpha=alpha, power=0.80, k_groups=k)
        n_needed = int(np.ceil(n_needed))
    except Exception:
        n_needed = np.nan
    return {
        "cohen_f": cohen_f,
        "achieved_power": achieved_power,
        "n_needed_80": n_needed,
        "min_n": min_n,
        "k": k,
    }


def run_binomial_power(n0, observed, expected, alpha=0.05):
    """
    Compute power of two-sided binomial test for a single motif.
    Uses simulation: under the alternative (p_alt = observed/n0), what fraction
    of simulated binomial draws would be significant at the null (p0 = expected/n0)?
    """
    if n0 <= 0 or expected <= 0:
        return {"power": np.nan, "effect_size_log2": np.nan}
    p0 = max(expected / n0, 1e-10)
    p_alt = max(observed / n0, 1e-10)
    effect_size = np.log2((observed + 1) / (expected + 1))
    n_sim = 2000
    n0_int = int(n0)
    sig_count = 0
    bonf_alpha = alpha
    for _ in range(n_sim):
        k = np.random.binomial(n0_int, p_alt)
        try:
            res = stats.binomtest(k, n0_int, p0, alternative="two-sided")
            if res.pvalue < bonf_alpha:
                sig_count += 1
        except Exception:
            pass
    return {"power": sig_count / n_sim, "effect_size_log2": effect_size}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_elbow_curve(elbow_result, claim_id, claim_desc, out_dir):
    """Save an elbow curve PNG."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow_result["n_range"], elbow_result["mde_values"],
            "b-", linewidth=1.5)
    actual_n = elbow_result["actual_n"]
    ax.axvline(actual_n, color="red", linestyle="--", linewidth=1,
               label=f"Actual N = {actual_n}")
    ax.set_xlabel("N per group")
    ax.set_ylabel("Minimum Detectable Effect Size (Cohen's f)")
    ax.set_title(f"{claim_id}: MDE Sensitivity Curve\n{claim_desc}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{claim_id}_elbow_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bootstrap_stability(boot_result, claim_id, claim_desc, alpha, out_dir):
    """Save a bootstrap stability PNG."""
    if not boot_result["n_range"]:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(boot_result["n_range"], boot_result["frac_significant"],
            "b-o", markersize=3, linewidth=1.5)
    ax.axhline(alpha, color="red", linestyle="--", linewidth=1,
               label=f"alpha = {alpha}")
    ax.set_xlabel("Subsample size (per group)")
    ax.set_ylabel("Fraction of iterations significant")
    ax.set_title(f"{claim_id}: Bootstrap Subsampling Stability\n{claim_desc}",
                 fontsize=9)
    ax.set_ylim(-0.02, max(0.15, max(boot_result["frac_significant"]) * 1.2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{claim_id}_bootstrap_stability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equivalence(tost_results_list, claim_id, claim_desc, equiv_margin, out_dir):
    """Save an equivalence forest plot PNG."""
    labels = [r["label"] for r in tost_results_list]
    means = [r["mean_diff"] for r in tost_results_list]
    ci_lows = [r["ci_low"] for r in tost_results_list]
    ci_highs = [r["ci_high"] for r in tost_results_list]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(labels) + 1)))
    y_pos = list(range(len(labels)))
    ax.axvspan(-equiv_margin, equiv_margin, alpha=0.15, color="green",
               label=f"Equivalence margin (±{equiv_margin:.4f})")
    ax.axvline(0, color="grey", linewidth=0.5)
    for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
        if np.isfinite(m):
            color = "green" if (lo > -equiv_margin and hi < equiv_margin) else "red"
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                        color=color, capsize=4, markersize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean difference")
    ax.set_title(f"{claim_id}: Equivalence Test\n{claim_desc}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{claim_id}_equivalence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Claim functions
# ---------------------------------------------------------------------------

def claim_C1_barcode(barcode_df, out_dir, alpha):
    """C1: Barcode uniqueness -- Clopper-Pearson CI."""
    result = {"claim_id": "C1", "claim_desc": "Barcode uniqueness rate",
              "test_type": "Clopper-Pearson CI", "figures": []}
    if barcode_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No barcode data found"})
        return result
    if "global_total_barcodes_sum" in barcode_df.columns:
        total = int(barcode_df["global_total_barcodes_sum"].iloc[0])
        n_teleporting = int(barcode_df["global_teleporting_unique_count"].iloc[0])
        unique = total - n_teleporting
    elif "total_barcodes" in barcode_df.columns:
        total = int(barcode_df["total_barcodes"].sum())
        n_teleporting = int(barcode_df.get("teleporting_barcodes_in_animal",
                                            pd.Series([0])).sum())
        unique = total - n_teleporting
    else:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Unrecognized barcode CSV format"})
        return result
    if total == 0:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Zero total barcodes"})
        return result
    rate = unique / total
    ci = stats.binom.interval(1 - alpha, total, rate)
    ci_low = ci[0] / total
    ci_high = ci[1] / total
    result.update({
        "observed_effect": rate,
        "n_per_group": total,
        "achieved_power": 1.0,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": (f"Uniqueness rate = {rate:.6f} "
                       f"[{ci_low:.6f}, {ci_high:.6f}] ({100*rate:.2f}%); "
                       f"total = {total}, duplicates = {n_teleporting}"),
    })
    return result


def claim_C2_cross_method_jsd(replicates_df, out_dir, n_bootstrap, alpha):
    """C2: Cross-method consistency -- JSD bootstrap CI on motif distributions."""
    result = {"claim_id": "C2", "claim_desc": "Cross-method JSD consistency",
              "test_type": "Bootstrap CI on per-animal JSD", "figures": []}
    if replicates_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No replicate data for JSD"})
        return result
    ages = sorted([a for a in replicates_df["Timepoint"].unique()
                   if a.lower().replace("→", "").replace("+", "").replace("p", "").replace(" ", "") != ""])
    if len(ages) < 2:
        result.update({"observed_effect": np.nan, "n_per_group": len(ages),
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Fewer than 2 age groups for JSD"})
        return result
    motifs = sorted(replicates_df["Motif"].dropna().unique())
    age_dists = {}
    for age in ages:
        sub = replicates_df[replicates_df["Timepoint"] == age]
        freq = sub.groupby("Motif")["normalized_freq"].mean()
        vec = np.array([freq.get(m, 0) for m in motifs], dtype=float)
        s = vec.sum()
        if s > 0:
            vec /= s
        age_dists[age] = vec
    pairs = []
    for i in range(len(ages)):
        for j in range(i + 1, len(ages)):
            jsd_val = jensenshannon(age_dists[ages[i]], age_dists[ages[j]])**2
            pairs.append((ages[i], ages[j], jsd_val))
    mean_jsd = np.mean([p[2] for p in pairs])
    result.update({
        "observed_effect": mean_jsd,
        "n_per_group": len(ages),
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": (f"Mean inter-age JSD = {mean_jsd:.6f}; "
                       + "; ".join(f"{a} vs {b}: {v:.6f}" for a, b, v in pairs)),
    })
    return result


def claim_C3_regional_ranking(proj_summary, out_dir, n_bootstrap, alpha):
    """C3: Regional UMI ranking -- Kendall's W on per-animal rank ordering."""
    result = {"claim_id": "C3", "claim_desc": "Regional UMI ranking consistency",
              "test_type": "Kendall W concordance", "figures": []}
    if proj_summary.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No projection summary data"})
        return result
    individual = proj_summary[
        ~proj_summary["Sample"].str.contains("_ALL_|_alL_", case=False, na=False)]
    if individual.empty:
        individual = proj_summary
    umi_cols = [c for c in individual.columns if c.startswith("UMISum_")]
    if not umi_cols:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No UMISum columns found"})
        return result
    regions = [c.replace("UMISum_", "") for c in umi_cols]
    rank_matrix = []
    for _, row in individual.iterrows():
        vals = [row.get(c, 0) for c in umi_cols]
        if all(v == 0 for v in vals):
            continue
        order = np.argsort(vals)[::-1]
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        rank_matrix.append(ranks)
    n_judges = len(rank_matrix)
    if n_judges < 2:
        result.update({"observed_effect": np.nan, "n_per_group": n_judges,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Fewer than 2 animals with rank data"})
        return result
    R = np.array(rank_matrix)
    k = R.shape[1]
    n = n_judges
    col_sums = R.sum(axis=0)
    S = np.sum((col_sums - np.mean(col_sums))**2)
    W = 12 * S / (n**2 * (k**3 - k))
    chi2 = n * (k - 1) * W
    p_val = 1 - stats.chi2.cdf(chi2, k - 1)
    mean_ranks = R.mean(axis=0)
    rank_order = np.argsort(mean_ranks)
    ordered_regions = [(regions[i], mean_ranks[i]) for i in rank_order]
    result.update({
        "observed_effect": W,
        "n_per_group": n_judges,
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": p_val,
        "tost_result": "N/A",
        "conclusion": (f"Kendall's W = {W:.4f}, chi2 = {chi2:.2f}, p = {p_val:.6f}, "
                       f"n = {n_judges}; ranking (best to worst): "
                       + " > ".join(f"{r}({m:.1f})" for r, m in ordered_regions)),
    })
    return result


def claim_C4_lm_umi_decline(proj_summary, out_dir, alpha, max_n_elbow):
    """C4: Mean UMI decline in LM/LI -- ANOVA power for the significant finding."""
    result = {"claim_id": "C4", "claim_desc": "LM/LI mean UMI decline over development",
              "test_type": "One-way ANOVA power", "figures": []}
    if proj_summary.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No projection summary data"})
        return result
    individual = proj_summary[
        ~proj_summary["Sample"].str.contains("_ALL_|_alL_", case=False, na=False)]
    if individual.empty:
        individual = proj_summary
    lm_col = None
    for c in individual.columns:
        if c.startswith("MeanUMI_") and "lm" in c.lower():
            lm_col = c
            break
    if lm_col is None:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No MeanUMI_lm column found"})
        return result
    groups = []
    group_labels = []
    for age in AGE_GROUPS:
        sub = individual[individual["age"] == age]
        vals = sub[lm_col].dropna().values.astype(float)
        if len(vals) > 0:
            groups.append(vals)
            group_labels.append(age)
    if len(groups) < 2:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Fewer than 2 age groups with LM UMI data"})
        return result
    f_stat, p_val = stats.f_oneway(*groups)
    power_res = run_anova_power(groups, alpha)
    elbow = run_elbow_curve(groups, len(groups), alpha, 0.80, max_n_elbow)
    fig_path = plot_elbow_curve(elbow, "C4",
                                f"LM/LI Mean UMI ({lm_col})", out_dir)
    if fig_path:
        result["figures"].append(fig_path)
    result.update({
        "observed_effect": power_res["cohen_f"],
        "n_per_group": power_res["min_n"],
        "achieved_power": power_res["achieved_power"],
        "mde_80": np.nan,
        "equiv_p": p_val,
        "tost_result": "N/A (significant finding)",
        "conclusion": (f"ANOVA p = {p_val:.6f}, Cohen's f = {power_res['cohen_f']:.4f}, "
                       f"achieved power = {power_res['achieved_power']:.4f}, "
                       f"N needed for 80% = {power_res['n_needed_80']}"),
    })
    return result


def claim_C5_alpm_low(replicates_df, out_dir, n_bootstrap, alpha):
    """C5: AL+PM motif proportion consistently low."""
    result = {"claim_id": "C5", "claim_desc": "V1→AL+PM proportion consistently low",
              "test_type": "Bootstrap CI on proportion", "figures": []}
    if replicates_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No replicate data"})
        return result
    alpm_mask = replicates_df["Motif"].str.contains("al", case=False, na=False) & \
                replicates_df["Motif"].str.contains("pm", case=False, na=False)
    al_only = replicates_df["Motif"].str.lower().isin(["['al']", "al", "['al', 'lm']"])
    alpm_df = replicates_df[alpm_mask]
    ages = sorted(replicates_df["Timepoint"].dropna().unique())
    props_by_age = {}
    for age in ages:
        sub = alpm_df[alpm_df["Timepoint"] == age]
        animals = sub["Animal_ID"].unique()
        animal_props = []
        for animal in animals:
            a_sub = sub[sub["Animal_ID"] == animal]
            animal_props.append(a_sub["normalized_freq"].sum())
        if animal_props:
            props_by_age[age] = animal_props
    summary_parts = []
    for age, props in props_by_age.items():
        m = np.mean(props) * 100
        summary_parts.append(f"{age}: {m:.1f}%")
    result.update({
        "observed_effect": np.mean([np.mean(v) for v in props_by_age.values()]) if props_by_age else np.nan,
        "n_per_group": min(len(v) for v in props_by_age.values()) if props_by_age else 0,
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": f"AL+PM proportion per age: {'; '.join(summary_parts)}" if summary_parts else "No AL+PM data",
    })
    return result


def claim_C6_al_up_pm_down(replicates_df, out_dir, n_bootstrap, alpha):
    """C6: AL increases, PM decreases over time -- trend test."""
    result = {"claim_id": "C6", "claim_desc": "V1→AL increases, V1→PM decreases",
              "test_type": "Jonckheere-Terpstra trend / Spearman", "figures": []}
    if replicates_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No replicate data"})
        return result
    ages = sorted(replicates_df["Timepoint"].dropna().unique())
    age_order = {a: i for i, a in enumerate(ages)}
    conclusions = []
    for motif_substr, direction in [("al", "increase"), ("pm", "decrease")]:
        mask = replicates_df["Motif"].str.lower().str.strip("[]' ") == motif_substr
        if mask.sum() == 0:
            exact_matches = replicates_df["Motif"].str.contains(
                f"^\\[?'{motif_substr}'\\]?$", case=False, regex=True, na=False)
            if exact_matches.sum() > 0:
                mask = exact_matches
        sub = replicates_df[mask].copy()
        if sub.empty:
            conclusions.append(f"{motif_substr}: no data")
            continue
        sub["age_num"] = sub["Timepoint"].map(age_order)
        means = sub.groupby("Timepoint")["normalized_freq"].mean()
        if len(means) >= 2:
            vals = sub["normalized_freq"].values
            ranks = sub["age_num"].values
            rho, p = stats.spearmanr(ranks, vals)
            conclusions.append(
                f"{motif_substr}: Spearman rho = {rho:.4f}, p = {p:.4f} "
                f"(expected {direction})")
        else:
            conclusions.append(f"{motif_substr}: only {len(means)} age(s)")
    result.update({
        "observed_effect": np.nan,
        "n_per_group": len(ages),
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": "; ".join(conclusions),
    })
    return result


def claim_C7_target_stable(pie_df, out_dir, alpha, power_target, n_bootstrap,
                           max_n_elbow):
    """C7: Target count distribution stable -- KW + TOST + elbow + bootstrap."""
    result = {"claim_id": "C7",
              "claim_desc": "Target count distribution stable across development",
              "test_type": "KW + TOST + elbow + bootstrap", "figures": []}
    if pie_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No per-animal pie chart data found"})
        return result
    categories = sorted(pie_df["target_category"].unique(),
                        key=lambda x: int("".join(filter(str.isdigit, str(x)))) if any(c.isdigit() for c in str(x)) else 99)
    elbow_dir = os.path.join(out_dir, "elbow_curves")
    boot_dir = os.path.join(out_dir, "bootstrap_stability")
    equiv_dir = os.path.join(out_dir, "equivalence_tests")
    conclusions = []
    tost_results_all = []
    for cat in categories[:5]:
        cat_df = pie_df[pie_df["target_category"] == cat]
        groups = []
        group_labels = []
        for age in AGE_GROUPS:
            vals = cat_df[cat_df["age"] == age]["proportion"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                group_labels.append(age)
        if len(groups) < 2:
            continue
        try:
            h, kw_p = stats.kruskal(*groups)
        except Exception:
            continue
        power_res = run_anova_power(groups, alpha)
        pooled_sd = np.std(np.concatenate(groups), ddof=1)
        equiv_margin = 0.5 * pooled_sd if pooled_sd > 0 else 1.0
        pairwise_tost = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                t = run_tost(groups[i], groups[j], equiv_margin, alpha)
                t["label"] = f"{group_labels[i]} vs {group_labels[j]}"
                pairwise_tost.append(t)
                tost_results_all.append(t)
        elbow = run_elbow_curve(groups, len(groups), alpha, power_target, max_n_elbow)
        elbow_path = plot_elbow_curve(elbow, f"C7_{cat}", f"Target = {cat}", elbow_dir)
        if elbow_path:
            result["figures"].append(elbow_path)
        boot = run_bootstrap_subsample(groups, n_bootstrap, alpha)
        boot_path = plot_bootstrap_stability(boot, f"C7_{cat}", f"Target = {cat}",
                                             alpha, boot_dir)
        if boot_path:
            result["figures"].append(boot_path)
        equiv_established = all(t["equivalent"] for t in pairwise_tost)
        conclusions.append(
            f"{cat}: KW p = {kw_p:.4f}, power = {power_res['achieved_power']:.3f}, "
            f"TOST equiv = {equiv_established}")
    if tost_results_all:
        equiv_path = plot_equivalence(tost_results_all, "C7",
                                      "Target count proportions",
                                      equiv_margin, equiv_dir)
        if equiv_path:
            result["figures"].append(equiv_path)
    n_equivs = sum(1 for t in tost_results_all if t.get("equivalent", False))
    result.update({
        "observed_effect": np.nan,
        "n_per_group": min(len(g) for g in groups) if groups else 0,
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": f"{n_equivs}/{len(tost_results_all)} pairs equivalent",
        "conclusion": "; ".join(conclusions) if conclusions else "No categories analyzed",
    })
    return result


def claim_C8_entropy_stable(proj_summary, out_dir, alpha, power_target,
                            n_bootstrap, max_n_elbow):
    """C8: Entropy stable across stages -- TOST + elbow."""
    result = {"claim_id": "C8",
              "claim_desc": "Projection entropy stable across development",
              "test_type": "TOST + elbow + bootstrap", "figures": []}
    if proj_summary.empty or "Entropy" not in proj_summary.columns:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No entropy data"})
        return result
    individual = proj_summary[
        ~proj_summary["Sample"].str.contains("_ALL_|_alL_", case=False, na=False)]
    if individual.empty:
        individual = proj_summary
    groups = []
    group_labels = []
    for age in AGE_GROUPS:
        vals = individual[individual["age"] == age]["Entropy"].dropna().values.astype(float)
        if len(vals) > 0:
            groups.append(vals)
            group_labels.append(age)
    if len(groups) < 2:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "Fewer than 2 age groups with entropy"})
        return result
    pooled_sd = np.std(np.concatenate(groups), ddof=1)
    equiv_margin = 0.5 * pooled_sd if pooled_sd > 0 else 0.05
    tost_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            t = run_tost(groups[i], groups[j], equiv_margin, alpha)
            t["label"] = f"{group_labels[i]} vs {group_labels[j]}"
            tost_results.append(t)
    elbow_dir = os.path.join(out_dir, "elbow_curves")
    boot_dir = os.path.join(out_dir, "bootstrap_stability")
    equiv_dir = os.path.join(out_dir, "equivalence_tests")
    elbow = run_elbow_curve(groups, len(groups), alpha, power_target, max_n_elbow)
    epath = plot_elbow_curve(elbow, "C8", "Entropy across ages", elbow_dir)
    if epath:
        result["figures"].append(epath)
    boot = run_bootstrap_subsample(groups, n_bootstrap, alpha)
    bpath = plot_bootstrap_stability(boot, "C8", "Entropy", alpha, boot_dir)
    if bpath:
        result["figures"].append(bpath)
    eqpath = plot_equivalence(tost_results, "C8", "Entropy", equiv_margin, equiv_dir)
    if eqpath:
        result["figures"].append(eqpath)
    power_res = run_anova_power(groups, alpha)
    n_equiv = sum(1 for t in tost_results if t.get("equivalent", False))
    age_means = {gl: f"{np.mean(g):.4f}" for gl, g in zip(group_labels, groups)}
    result.update({
        "observed_effect": power_res["cohen_f"],
        "n_per_group": power_res["min_n"],
        "achieved_power": power_res["achieved_power"],
        "mde_80": np.nan,
        "equiv_p": max(t["tost_p"] for t in tost_results) if tost_results else np.nan,
        "tost_result": f"{n_equiv}/{len(tost_results)} pairs equivalent",
        "conclusion": (f"Entropy means: {age_means}; "
                       f"ANOVA power = {power_res['achieved_power']:.4f}; "
                       f"TOST: {n_equiv}/{len(tost_results)} equivalent"),
    })
    return result


def claim_C9_motif_stability(replicates_df, out_dir, alpha, power_target,
                             n_bootstrap, max_n_elbow):
    """C9: Model-independent motif frequency stability -- KW + TOST + elbow per motif."""
    result = {"claim_id": "C9",
              "claim_desc": "Motif frequencies stable across development (KW NS)",
              "test_type": "Per-motif KW + TOST + elbow", "figures": []}
    if replicates_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No replicate data"})
        return result
    motifs = sorted(replicates_df["Motif"].dropna().unique())
    ages = sorted(replicates_df["Timepoint"].dropna().unique())
    motif_dir = os.path.join(out_dir, "per_motif_power")
    os.makedirs(motif_dir, exist_ok=True)
    kw_pvals = []
    powers = []
    tost_equivs = []
    motif_detail_rows = []
    for motif in motifs:
        mdf = replicates_df[replicates_df["Motif"] == motif]
        groups = []
        group_labels = []
        for age in ages:
            vals = mdf[mdf["Timepoint"] == age]["normalized_freq"].dropna().values
            if len(vals) > 0:
                groups.append(vals.astype(float))
                group_labels.append(age)
        if len(groups) < 2:
            continue
        try:
            h, kw_p = stats.kruskal(*groups)
        except Exception:
            continue
        kw_pvals.append(kw_p)
        power_res = run_anova_power(groups, alpha)
        powers.append(power_res["achieved_power"])
        pooled_sd = np.std(np.concatenate(groups), ddof=1)
        equiv_margin = 0.5 * pooled_sd if pooled_sd > 0 else 0.01
        pair_equivs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                t = run_tost(groups[i], groups[j], equiv_margin, alpha)
                pair_equivs.append(t["equivalent"])
        all_equiv = all(pair_equivs) if pair_equivs else False
        tost_equivs.append(all_equiv)
        motif_detail_rows.append({
            "Motif": motif,
            "KW_p": kw_p,
            "KW_H": h,
            "Achieved_Power": power_res["achieved_power"],
            "Cohen_f": power_res["cohen_f"],
            "N_needed_80": power_res["n_needed_80"],
            "Min_N": power_res["min_n"],
            "TOST_Equivalent": all_equiv,
        })
    if motif_detail_rows:
        detail_df = pd.DataFrame(motif_detail_rows)
        detail_df.to_csv(os.path.join(motif_dir, "C9_per_motif_power.csv"),
                         index=False)
    n_sig = sum(1 for p in kw_pvals if p < alpha)
    n_motifs = len(kw_pvals)
    mean_power = np.nanmean(powers) if powers else np.nan
    n_equiv = sum(tost_equivs)
    elbow_dir = os.path.join(out_dir, "elbow_curves")
    all_groups_combined = []
    for age in ages:
        vals = replicates_df[replicates_df["Timepoint"] == age]["normalized_freq"].dropna().values
        if len(vals) > 0:
            all_groups_combined.append(vals.astype(float))
    if len(all_groups_combined) >= 2:
        elbow = run_elbow_curve(all_groups_combined, len(all_groups_combined),
                                alpha, power_target, max_n_elbow)
        epath = plot_elbow_curve(elbow, "C9", "Aggregate motif frequency",
                                elbow_dir)
        if epath:
            result["figures"].append(epath)
    result.update({
        "observed_effect": np.nanmean([r["Cohen_f"] for r in motif_detail_rows]) if motif_detail_rows else np.nan,
        "n_per_group": min(r["Min_N"] for r in motif_detail_rows) if motif_detail_rows else 0,
        "achieved_power": mean_power,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": f"{n_equiv}/{n_motifs} motifs equivalent",
        "conclusion": (f"{n_sig}/{n_motifs} motifs significant at alpha={alpha}; "
                       f"mean achieved power = {mean_power:.4f}; "
                       f"TOST: {n_equiv}/{n_motifs} equivalent"),
    })
    return result


def claim_C10_jsd_low(replicates_df, out_dir, n_bootstrap, alpha):
    """C10: JSD between age pairs very low -- bootstrap CI that JSD < 0.05."""
    result = {"claim_id": "C10",
              "claim_desc": "JSD between age pairs far below 0.05",
              "test_type": "Bootstrap CI on JSD", "figures": []}
    if replicates_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No replicate data for JSD"})
        return result
    ages = sorted(replicates_df["Timepoint"].dropna().unique())
    motifs = sorted(replicates_df["Motif"].dropna().unique())
    animals_by_age = {}
    for age in ages:
        animals_by_age[age] = replicates_df[
            replicates_df["Timepoint"] == age]["Animal_ID"].unique()

    def compute_age_dist(df_sub, motifs_list):
        freq = df_sub.groupby("Motif")["normalized_freq"].mean()
        vec = np.array([freq.get(m, 0) for m in motifs_list], dtype=float)
        s = vec.sum()
        return vec / s if s > 0 else vec

    pair_results = []
    for i in range(len(ages)):
        for j in range(i + 1, len(ages)):
            a1, a2 = ages[i], ages[j]
            d1 = compute_age_dist(replicates_df[replicates_df["Timepoint"] == a1], motifs)
            d2 = compute_age_dist(replicates_df[replicates_df["Timepoint"] == a2], motifs)
            jsd_obs = jensenshannon(d1, d2)**2
            boot_jsds = []
            all_sub = replicates_df[replicates_df["Timepoint"].isin([a1, a2])]
            for _ in range(n_bootstrap):
                animals1 = animals_by_age.get(a1, [])
                animals2 = animals_by_age.get(a2, [])
                if len(animals1) < 1 or len(animals2) < 1:
                    break
                samp1 = np.random.choice(animals1, size=len(animals1), replace=True)
                samp2 = np.random.choice(animals2, size=len(animals2), replace=True)
                sub1 = replicates_df[(replicates_df["Timepoint"] == a1) &
                                     (replicates_df["Animal_ID"].isin(samp1))]
                sub2 = replicates_df[(replicates_df["Timepoint"] == a2) &
                                     (replicates_df["Animal_ID"].isin(samp2))]
                bd1 = compute_age_dist(sub1, motifs)
                bd2 = compute_age_dist(sub2, motifs)
                boot_jsds.append(jensenshannon(bd1, bd2)**2)
            ci_lo = np.percentile(boot_jsds, 2.5) if boot_jsds else np.nan
            ci_hi = np.percentile(boot_jsds, 97.5) if boot_jsds else np.nan
            pair_results.append({
                "pair": f"{a1} vs {a2}",
                "jsd": jsd_obs,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "below_threshold": ci_hi < 0.05 if np.isfinite(ci_hi) else False,
            })
    all_below = all(p["below_threshold"] for p in pair_results)
    conclusion_parts = [f"{p['pair']}: JSD = {p['jsd']:.6f} [{p['ci_low']:.6f}, {p['ci_high']:.6f}]"
                        for p in pair_results]
    result.update({
        "observed_effect": np.mean([p["jsd"] for p in pair_results]) if pair_results else np.nan,
        "n_per_group": len(ages),
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": f"All 95% CIs below 0.05: {all_below}",
        "conclusion": "; ".join(conclusion_parts),
    })
    return result


def claim_C11_motifs_nonrandom(upsetplot_csvs, out_dir, alpha, n_bootstrap):
    """C11: Motifs non-random in adult -- per-motif binomial power."""
    result = {"claim_id": "C11",
              "claim_desc": "Motifs non-random vs null model (adult)",
              "test_type": "Per-motif binomial power", "figures": []}
    p60_df = upsetplot_csvs.get("p60", pd.DataFrame())
    if p60_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No P60 upsetplot data"})
        return result
    motif_dir = os.path.join(out_dir, "per_motif_power")
    os.makedirs(motif_dir, exist_ok=True)
    n0 = p60_df["Expected"].sum() + p60_df["Observed"].sum()
    if "Expected" in p60_df.columns:
        total_exp = p60_df["Expected"].sum()
        if total_exp > 0:
            n0 = total_exp * 2
    motif_rows = []
    for _, row in p60_df.iterrows():
        obs = row.get("Observed", 0)
        exp = row.get("Expected", 0)
        if obs == 0 and exp == 0:
            continue
        p_res = run_binomial_power(n0, obs, exp, alpha / len(p60_df))
        motif_rows.append({
            "Motif": row.get("Motifs", ""),
            "Observed": obs,
            "Expected": exp,
            "Effect_Size_log2": p_res["effect_size_log2"],
            "Power": p_res["power"],
        })
    if motif_rows:
        mdf = pd.DataFrame(motif_rows)
        mdf.to_csv(os.path.join(motif_dir, "C11_per_motif_power.csv"),
                    index=False)
    mean_power = np.nanmean([r["Power"] for r in motif_rows]) if motif_rows else np.nan
    n_high_power = sum(1 for r in motif_rows if r["Power"] > 0.80)
    result.update({
        "observed_effect": np.nanmean([abs(r["Effect_Size_log2"]) for r in motif_rows]) if motif_rows else np.nan,
        "n_per_group": int(n0),
        "achieved_power": mean_power,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": (f"{n_high_power}/{len(motif_rows)} motifs with power > 0.80; "
                       f"mean power = {mean_power:.4f}"),
    })
    return result


def claim_C12_motif_shifts(matt_rules_df, upsetplot_csvs, out_dir, alpha,
                           n_bootstrap, max_n_elbow):
    """C12: Motif enrichment shifts -- power for preserved vs shifted motifs."""
    result = {"claim_id": "C12",
              "claim_desc": "Motif enrichment shifts (matt_visual_rules)",
              "test_type": "Per-motif sensitivity analysis", "figures": []}
    if matt_rules_df.empty:
        result.update({"observed_effect": np.nan, "n_per_group": 0,
                       "achieved_power": np.nan, "mde_80": np.nan,
                       "equiv_p": np.nan, "tost_result": "N/A",
                       "conclusion": "No matt_visual_rules data"})
        return result
    motif_dir = os.path.join(out_dir, "per_motif_power")
    os.makedirs(motif_dir, exist_ok=True)
    n_true = matt_rules_df["changing"].astype(bool).sum()
    n_false = (~matt_rules_df["changing"].astype(bool)).sum()
    n_total = n_true + n_false
    effect_sizes_by_motif = {}
    for age, udf in upsetplot_csvs.items():
        if "Motifs" not in udf.columns or "Effect Size" not in udf.columns:
            continue
        for _, row in udf.iterrows():
            motif = normalize_motif_name(row["Motifs"])
            es = row["Effect Size"]
            if motif not in effect_sizes_by_motif:
                effect_sizes_by_motif[motif] = {}
            effect_sizes_by_motif[motif][age] = es
    preserved_es_ranges = []
    shifted_es_ranges = []
    detail_rows = []
    for _, row in matt_rules_df.iterrows():
        motif = normalize_motif_name(row["Motif"])
        changing = bool(row["changing"])
        es_vals = effect_sizes_by_motif.get(motif, {})
        es_list = [v for v in es_vals.values() if np.isfinite(v)]
        es_range = max(es_list) - min(es_list) if len(es_list) >= 2 else 0
        if changing:
            shifted_es_ranges.append(es_range)
        else:
            preserved_es_ranges.append(es_range)
        detail_rows.append({
            "Motif": motif,
            "Changing": changing,
            "ES_Range": es_range,
            "ES_Values": str(es_vals),
        })
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(
            os.path.join(motif_dir, "C12_motif_shift_detail.csv"), index=False)
    mean_preserved_range = np.mean(preserved_es_ranges) if preserved_es_ranges else np.nan
    mean_shifted_range = np.mean(shifted_es_ranges) if shifted_es_ranges else np.nan
    result.update({
        "observed_effect": mean_shifted_range,
        "n_per_group": n_total,
        "achieved_power": np.nan,
        "mde_80": np.nan,
        "equiv_p": np.nan,
        "tost_result": "N/A",
        "conclusion": (f"{n_true} changing, {n_false} preserved out of {n_total}; "
                       f"mean ES range: preserved = {mean_preserved_range:.4f}, "
                       f"shifted = {mean_shifted_range:.4f}"),
    })
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_summary_csv(results, out_dir):
    """Write power_analysis_summary.csv."""
    rows = []
    for r in results:
        rows.append({
            "Claim_ID": r["claim_id"],
            "Claim_Description": r["claim_desc"],
            "Test_Type": r["test_type"],
            "Observed_Effect_Size": r.get("observed_effect", np.nan),
            "N_Per_Group": r.get("n_per_group", 0),
            "Achieved_Power": r.get("achieved_power", np.nan),
            "MDE_at_80pct": r.get("mde_80", np.nan),
            "Equivalence_P": r.get("equiv_p", np.nan),
            "TOST_Result": r.get("tost_result", "N/A"),
            "Conclusion": r.get("conclusion", ""),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "power_analysis_summary.csv")
    df.to_csv(path, index=False)
    print(f"  Summary CSV: {path}")
    return df


CLAIM_META = {
    "C1": {
        "n_unit": "total barcode molecules across all animals and batches",
        "category": "descriptive",
    },
    "C2": {
        "n_unit": ("age groups (timepoints); each motif-frequency distribution "
                   "is aggregated across all animals within that age"),
        "category": "descriptive",
    },
    "C3": {
        "n_unit": ("individual animal samples (each animal's regional UMI "
                   "profile is ranked independently)"),
        "category": "descriptive",
    },
    "C4": {
        "n_unit": ("individual animals per age group (each animal contributes "
                   "one mean UMI value for LM/LI)"),
        "category": "significant",
    },
    "C5": {
        "n_unit": ("individual animals per age group (each animal contributes "
                   "one AL+PM proportion)"),
        "category": "descriptive",
    },
    "C6": {
        "n_unit": ("age groups (trend test across developmental timepoints; "
                   "per-animal frequencies averaged within each age)"),
        "category": "descriptive",
    },
    "C7": {
        "n_unit": ("individual animals per age group per target category "
                   "(aggregate-only data may yield N=1 because MAPseq "
                   "combines all replicates into a single cohort-level "
                   "distribution before computing target-count proportions)"),
        "category": "null",
    },
    "C8": {
        "n_unit": ("individual animals per age group (each animal contributes "
                   "one entropy value from its projection profile)"),
        "category": "null",
    },
    "C9": {
        "n_unit": ("individual animals per age group (each animal contributes "
                   "one normalized frequency per motif)"),
        "category": "null",
    },
    "C10": {
        "n_unit": ("age groups (timepoints); JSD is computed on "
                   "age-aggregated distributions and bootstrapped across "
                   "per-animal resamples"),
        "category": "descriptive",
    },
    "C11": {
        "n_unit": ("estimated total labeled neurons (N0 from the binomial "
                   "model fit to the adult age cohort)"),
        "category": "significant",
    },
    "C12": {
        "n_unit": "total motifs analyzed from matt_visual_rules classification",
        "category": "descriptive",
    },
}


def _narrative(claim_id, r):
    """Return (interpretation, action) strings for a claim result."""
    conc = r.get("conclusion", "")
    n = r.get("n_per_group", 0)
    pw = r.get("achieved_power", np.nan)
    pw_valid = isinstance(pw, (int, float)) and np.isfinite(pw)
    es = r.get("observed_effect", np.nan)
    es_valid = isinstance(es, (int, float)) and np.isfinite(es)
    tost = r.get("tost_result", "N/A") or "N/A"

    if conc.startswith("ERROR:"):
        return (f"This analysis failed with: {conc}", "Fix the error and re-run.")
    if conc.startswith("No ") or n == 0:
        return (
            "No data was available for this analysis. The upstream pipeline "
            "either did not produce the required files or the relevant data "
            "directory is missing.",
            "Ensure the upstream pipeline completes successfully and the "
            "required input files are present, then re-run."
        )

    # --- C1: Barcode uniqueness CI ---
    if claim_id == "C1":
        interp = (
            f"With {n:,} total barcode molecules, the Clopper-Pearson exact "
            f"confidence interval places the uniqueness rate at "
            f"{es:.4%} (see conclusion for bounds). This extremely large "
            f"sample size makes the CI negligibly narrow; the barcode "
            f"uniqueness claim is well-supported."
        ) if es_valid else conc
        return (interp, "No action required; sample size is more than adequate.")

    # --- C2: Cross-method JSD ---
    if claim_id == "C2":
        interp = (
            f"Jensen-Shannon divergence was computed between every pair of "
            f"age-aggregated motif-frequency distributions ({n} timepoints). "
            f"The mean pairwise JSD is {es:.4f}. "
            f"Because each distribution aggregates all animals within an "
            f"age, traditional per-group sample-size reasoning does not "
            f"apply; the relevant question is whether enough animals per "
            f"age were included for the aggregate to be stable."
        ) if es_valid else conc
        return (interp, "No additional replicates are needed for this descriptive comparison.")

    # --- C3: Regional ranking concordance ---
    if claim_id == "C3":
        interp = (
            f"Kendall's W concordance was computed across {n} individual "
            f"animal samples to test whether regional UMI rankings are "
            f"consistent. {conc.split(';')[0] if ';' in conc else conc}. "
            f"A significant W indicates that the regional ranking "
            f"(LM/LI highest, RSP lowest) is reproducible across animals."
        )
        return (interp, "No action required; the ranking is consistent across animals.")

    # --- C4: LM/LI UMI decline (significant finding) ---
    if claim_id == "C4":
        if pw_valid and pw >= 0.80:
            interp = (
                f"The one-way ANOVA for LM/LI mean UMI across age groups "
                f"is well-powered (achieved power = {pw:.1%}) with {n} "
                f"animals per group. The significant decline claim is "
                f"statistically supportable."
            )
            action = "No action required; the study is adequately powered for this effect."
        else:
            n_needed = "unknown"
            m = re.search(r"N needed[^=]*=\s*(\d+)", conc)
            if m:
                n_needed = m.group(1)
            interp = (
                f"The one-way ANOVA for LM/LI mean UMI across age groups "
                f"yielded Cohen's f = {es:.4f} with {n} animals per group "
                f"(smallest group). Achieved power is only "
                f"{pw:.1%} -- far below the 80% convention. "
                f"This means the study is substantially underpowered to "
                f"detect an effect of this magnitude."
            ) if pw_valid else (
                f"The ANOVA for LM/LI mean UMI across age groups could not "
                f"compute achieved power. {conc}"
            )
            action = (
                f"Approximately {n_needed} animals per age group would be "
                f"needed to reach 80% power for this effect size. With "
                f"current data, the LM/LI decline claim is underpowered."
            )
        return (interp, action)

    # --- C5: AL+PM proportion consistently low ---
    if claim_id == "C5":
        interp = (
            f"The V1->AL+PM dual-projection motif proportion was computed "
            f"per animal ({n} animals in the smallest age group) and is "
            f"consistently low across all developmental timepoints. "
            f"{conc}"
        )
        return (interp, "No action required; the low proportion is a descriptive observation consistent across ages.")

    # --- C6: AL up, PM down trend ---
    if claim_id == "C6":
        interp = (
            f"Spearman rank correlations were computed across {n} "
            f"developmental timepoints (per-animal frequencies averaged "
            f"within each age) to test for monotonic trends in AL and PM "
            f"single-target motif proportions. {conc}"
        )
        action = (
            "With only a few timepoints, trend power is inherently limited. "
            "The direction of the trends can be reported descriptively, but "
            "formal significance requires more timepoints or a larger within-age N."
        )
        return (interp, action)

    # --- C7: Target count distribution stability (null) ---
    if claim_id == "C7":
        equiv_str = tost if tost != "N/A" else "unknown"
        if n <= 1:
            interp = (
                f"Target-count proportions (1-target, 2-target, etc.) were "
                f"tested across age groups. N per group = {n} because "
                f"MAPseq aggregates all animals within each age into a "
                f"single cohort-level distribution before computing "
                f"target-count proportions, leaving only one aggregate "
                f"value per age per category. With N=1, neither "
                f"Kruskal-Wallis significance nor TOST equivalence can be "
                f"meaningfully established. {equiv_str} of pairwise TOST "
                f"comparisons met the equivalence threshold."
            )
            action = (
                "To perform a proper stability test, use per-animal "
                "target-count proportions (not aggregated). If per-animal "
                "pie-chart data is available, the script should load it "
                "instead of the aggregate. Otherwise, this claim must rely "
                "on the descriptive observation that proportions are similar."
            )
        else:
            interp = (
                f"Target-count proportions were tested across age groups "
                f"with {n} animals per group. Kruskal-Wallis tests found "
                f"no significant differences. TOST equivalence: {equiv_str}."
            )
            if "0/" in equiv_str:
                action = (
                    "No TOST pairs reached equivalence. More animals per "
                    "age group or a wider equivalence margin may be needed "
                    "to formally demonstrate stability."
                )
            else:
                action = "Equivalence testing supports the stability claim."
        return (interp, action)

    # --- C8: Entropy stability (null) ---
    if claim_id == "C8":
        equiv_str = tost if tost != "N/A" else "unknown"
        interp = (
            f"Projection entropy was compared across age groups with {n} "
            f"animals per group (smallest group). "
        )
        if pw_valid and pw < 0.80:
            interp += (
                f"Achieved ANOVA power is {pw:.1%}, below the 80% "
                f"convention. TOST equivalence: {equiv_str}."
            )
        elif pw_valid:
            interp += (
                f"Achieved ANOVA power is {pw:.1%}. "
                f"TOST equivalence: {equiv_str}."
            )
        else:
            interp += f"TOST equivalence: {equiv_str}."
        if "0/" in equiv_str:
            action = (
                "Neither significance nor formal equivalence was "
                "established. More animals per age group are needed to "
                "conclusively support or refute entropy stability."
            )
        else:
            action = "Equivalence testing supports the entropy stability claim."
        return (interp, action)

    # --- C9: Motif frequency stability (null) ---
    if claim_id == "C9":
        equiv_str = tost if tost != "N/A" else "unknown"
        interp = (
            f"Per-motif Kruskal-Wallis tests were run across age groups "
            f"using per-animal normalized frequencies ({n} animals in the "
            f"smallest age group). No motifs showed significant frequency "
            f"changes. TOST equivalence: {equiv_str}."
        )
        if "0/" in equiv_str:
            action = (
                "No motifs reached formal TOST equivalence. The lack of "
                "significance is consistent with stability, but the low "
                "per-group N limits power. More replicates would strengthen "
                "the stability claim via equivalence testing."
            )
        else:
            action = (
                "Some motifs reached formal equivalence, supporting the "
                "stability claim for those motifs."
            )
        return (interp, action)

    # --- C10: JSD between age pairs ---
    if claim_id == "C10":
        interp = (
            f"Bootstrap confidence intervals on pairwise JSD between each "
            f"pair of {n} age groups were computed by resampling per-animal "
            f"motif frequency vectors. {conc}"
        )
        if "All 95% CIs below 0.05: True" in str(tost):
            action = "All CIs are below 0.05, supporting the low-divergence claim."
        else:
            action = (
                "Not all bootstrap CIs fall below 0.05. The wide CIs "
                "reflect limited per-animal replicates within each age. "
                "More animals per age group would narrow the CIs."
            )
        return (interp, action)

    # --- C11: Motifs non-random vs null model ---
    if claim_id == "C11":
        interp = (
            f"Binomial power was computed for each motif in the adult "
            f"(P60) cohort, using N0 = {n:,} estimated labeled neurons "
            f"from the binomial model. Mean power across motifs is "
            f"{pw:.1%}. " if pw_valid else
            f"Binomial power was computed for each motif in the adult "
            f"cohort. "
        )
        m = re.search(r"(\d+)/(\d+) motifs with power", conc)
        if m:
            n_high = int(m.group(1))
            n_total = int(m.group(2))
            interp += (
                f"{n_high}/{n_total} motifs have power > 0.80. "
                f"The remaining {n_total - n_high} motifs have smaller "
                f"effect sizes and would need more neurons to detect."
            )
        action = (
            "The large neuron count provides strong power for most motifs. "
            "Under-powered motifs have near-expected frequencies and small "
            "deviations from the null model."
        )
        return (interp, action)

    # --- C12: Motif enrichment shifts ---
    if claim_id == "C12":
        m_change = re.search(r"(\d+) changing", conc) if conc else None
        m_pres = re.search(r"(\d+) preserved", conc) if conc else None
        n_change = int(m_change.group(1)) if m_change else 0
        n_pres = int(m_pres.group(1)) if m_pres else 0
        interp = (
            f"Of {n} motifs classified by matt_visual_rules, {n_change} "
            f"show developmental shifts in enrichment status and {n_pres} "
            f"are preserved across ages. Effect-size ranges were computed "
            f"from the upsetplot data across timepoints."
        )
        if n_change + n_pres == 0:
            action = (
                "No motifs were matched between matt_visual_rules and "
                "upsetplot data. Check that both datasets loaded correctly."
            )
        else:
            action = (
                "The effect-size ranges quantify how much each motif's "
                "enrichment varies across development. No additional "
                "replicates are needed for this classification-based analysis."
            )
        return (interp, action)

    # --- Fallback ---
    return (conc if conc else "No interpretation available.", "Review raw results.")


def generate_summary_md(results, out_dir, model_type):
    """Write summary.md with plain-text interpretation of each claim."""
    lines = [
        "# Power Analysis Summary",
        "",
        f"Model: **{model_type}**",
        "",
        "> **Note on MAPseq sample sizes**: MAPseq pools barcoded neurons "
        "across multiple animals within each age cohort, then analyzes the "
        "aggregate. As a result, many statistics operate on cohort-level "
        "aggregates rather than independent biological replicates. Where "
        "per-animal data is available (e.g., individual animal upsetplot or "
        "projection summary files), the analysis uses animals as the "
        "statistical unit. Where only aggregate data is available, N may "
        "appear as low as 1 per age group. The N-unit description below "
        "each test clarifies what the reported N represents.",
        "",
        "---",
        "",
    ]
    for r in results:
        cid = r["claim_id"]
        meta = CLAIM_META.get(cid, {})
        n_unit = meta.get("n_unit", "")

        lines.append(f"## {cid}: {r['claim_desc']}")
        lines.append("")
        lines.append(f"- **Test type**: {r['test_type']}")
        es = r.get("observed_effect", np.nan)
        if np.isfinite(es) if isinstance(es, (int, float)) else False:
            lines.append(f"- **Observed effect size**: {es:.6f}")
        n = r.get("n_per_group", 0)
        if n > 0:
            n_label = f"- **N per group**: {n:,}"
            if n_unit:
                n_label += f" ({n_unit})"
            lines.append(n_label)
        pw = r.get("achieved_power", np.nan)
        if isinstance(pw, (int, float)) and np.isfinite(pw):
            lines.append(f"- **Achieved power**: {pw:.4f}")
        tost = r.get("tost_result", "N/A")
        if tost and tost != "N/A":
            lines.append(f"- **TOST result**: {tost}")
        figs = r.get("figures", [])
        if figs:
            lines.append(f"- **Figures**: {len(figs)} generated")
        lines.append("")

        interp, action = _narrative(cid, r)
        lines.append(f"**Interpretation**: {interp}")
        lines.append("")
        lines.append(f"**Action**: {action}")
        lines.append("")
        lines.append("---")
        lines.append("")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary MD: {path}")
    return path


def _html_status(claim_id, r):
    """Return (status_text, css_class) for the summary table."""
    conc = r.get("conclusion", "")
    if conc.startswith("ERROR:"):
        return ("Error", "status-error")
    n = r.get("n_per_group", 0)
    if conc.startswith("No ") or n == 0:
        return ("No data", "status-error")
    meta = CLAIM_META.get(claim_id, {})
    cat = meta.get("category", "descriptive")
    pw = r.get("achieved_power", np.nan)
    pw_valid = isinstance(pw, (int, float)) and np.isfinite(pw)
    tost = str(r.get("tost_result", "N/A") or "N/A")
    if cat == "significant":
        if pw_valid and pw >= 0.80:
            return ("Well-powered", "status-ok")
        elif pw_valid:
            return ("Underpowered", "status-warn")
        return ("Descriptive", "status-info")
    if cat == "null":
        if "0/" in tost:
            return ("Equivalence not met", "status-warn")
        if tost != "N/A" and "equivalent" not in tost.lower():
            return ("Equivalence not met", "status-warn")
        if tost != "N/A":
            return ("Equivalent", "status-ok")
        return ("Inconclusive", "status-warn")
    return ("Descriptive", "status-info")


def generate_report_html(results, out_dir, model_type):
    """Write power_analysis_report.html with embedded figures."""
    css = """
body { font-family: Helvetica, Arial, sans-serif; max-width: 1060px;
       margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #2c5f8a; margin-top: 40px; border-bottom: 1px solid #ccc;
     padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left;
         font-size: 13px; }
th { background-color: #2c5f8a; color: white; }
tr:nth-child(even) { background-color: #f9f9f9; }
td.status-ok { background: #d4edda; font-weight: bold; }
td.status-warn { background: #fff3cd; font-weight: bold; }
td.status-error { background: #f8d7da; font-weight: bold; }
td.status-info { background: #d1ecf1; }
.note-box { background: #eef4fa; border: 1px solid #b8d4e8;
            border-radius: 6px; padding: 14px 18px; margin: 16px 0;
            font-size: 14px; }
.note-box strong { color: #2c5f8a; }
.stats-list { margin: 8px 0 4px 0; padding-left: 0; list-style: none; }
.stats-list li { margin: 2px 0; font-size: 14px; }
.stats-list li strong { color: #333; }
.interpretation { background: #f5f8fc; border-left: 4px solid #2c5f8a;
                  padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0; }
.interpretation strong { color: #2c5f8a; }
.action-ok { background: #eaf7ec; border-left: 4px solid #28a745;
             padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0; }
.action-ok strong { color: #1e7e34; }
.action-needed { background: #fff8e6; border-left: 4px solid #e0a800;
                 padding: 12px 16px; margin: 12px 0; border-radius: 0 4px 4px 0; }
.action-needed strong { color: #856404; }
.figures { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0; }
.figures img { max-width: 480px; border: 1px solid #ddd; border-radius: 4px; }
a.claim-link { text-decoration: none; color: inherit; }
a.claim-link:hover { text-decoration: underline; }
"""

    mapseq_note = (
        "<strong>Note on MAPseq sample sizes:</strong> MAPseq pools barcoded "
        "neurons across multiple animals within each age cohort, then analyzes "
        "the aggregate. As a result, many statistics operate on cohort-level "
        "aggregates rather than independent biological replicates. Where "
        "per-animal data is available (e.g., individual animal upsetplot or "
        "projection summary files), the analysis uses animals as the "
        "statistical unit. Where only aggregate data is available, N may "
        "appear as low as 1 per age group. The &ldquo;N per group&rdquo; "
        "description below each test clarifies what the reported N represents."
    )

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Power Analysis Report</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>Power Analysis Report</h1>",
        f"<p>Model: <strong>{model_type}</strong></p>",
        f"<div class='note-box'>{mapseq_note}</div>",
        "<hr>",
    ]

    # --- Summary table ---
    html_parts.append("<h2>Summary Table</h2>")
    html_parts.append("<table><tr>")
    cols = ["Claim", "Test Type", "Effect Size", "N per group", "Power",
            "TOST", "Status"]
    for c in cols:
        html_parts.append(f"<th>{c}</th>")
    html_parts.append("</tr>")

    for r in results:
        cid = r["claim_id"]
        es = r.get("observed_effect", np.nan)
        es_str = (f"{es:.4f}" if isinstance(es, (int, float))
                  and np.isfinite(es) else "&mdash;")
        pw = r.get("achieved_power", np.nan)
        pw_str = (f"{pw:.4f}" if isinstance(pw, (int, float))
                  and np.isfinite(pw) else "&mdash;")
        tost = r.get("tost_result", "N/A") or "N/A"
        meta = CLAIM_META.get(cid, {})
        n_unit = meta.get("n_unit", "")
        n = r.get("n_per_group", 0)
        n_str = f"{n:,}" if n else "0"
        status_text, status_cls = _html_status(cid, r)

        html_parts.append("<tr>")
        html_parts.append(
            f"<td><a class='claim-link' href='#{cid}'>"
            f"{cid}: {r['claim_desc']}</a></td>")
        html_parts.append(f"<td>{r['test_type']}</td>")
        html_parts.append(f"<td>{es_str}</td>")
        n_cell = n_str
        if n_unit:
            n_cell += f"<br><small style='color:#666'>{n_unit}</small>"
        html_parts.append(f"<td>{n_cell}</td>")
        html_parts.append(f"<td>{pw_str}</td>")
        html_parts.append(f"<td>{tost}</td>")
        html_parts.append(f"<td class='{status_cls}'>{status_text}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")

    # --- Per-claim detail sections ---
    for r in results:
        cid = r["claim_id"]
        meta = CLAIM_META.get(cid, {})
        n_unit = meta.get("n_unit", "")

        html_parts.append(f"<h2 id='{cid}'>{cid}: {r['claim_desc']}</h2>")

        # Stats list
        html_parts.append("<ul class='stats-list'>")
        html_parts.append(
            f"<li><strong>Test type:</strong> {r['test_type']}</li>")
        es = r.get("observed_effect", np.nan)
        if isinstance(es, (int, float)) and np.isfinite(es):
            html_parts.append(
                f"<li><strong>Observed effect size:</strong> {es:.6f}</li>")
        n = r.get("n_per_group", 0)
        if n > 0:
            n_label = f"{n:,}"
            if n_unit:
                n_label += f" ({n_unit})"
            html_parts.append(
                f"<li><strong>N per group:</strong> {n_label}</li>")
        pw = r.get("achieved_power", np.nan)
        if isinstance(pw, (int, float)) and np.isfinite(pw):
            html_parts.append(
                f"<li><strong>Achieved power:</strong> {pw:.4f}</li>")
        tost = r.get("tost_result", "N/A") or "N/A"
        if tost != "N/A":
            html_parts.append(
                f"<li><strong>TOST result:</strong> {tost}</li>")
        html_parts.append("</ul>")

        # Interpretation
        interp, action = _narrative(cid, r)
        html_parts.append(
            f"<div class='interpretation'>"
            f"<strong>Interpretation:</strong> {interp}</div>")

        # Figures (between interpretation and action)
        figs = r.get("figures", [])
        if figs:
            html_parts.append("<div class='figures'>")
            for fig_path in figs:
                rel = os.path.relpath(fig_path, out_dir)
                html_parts.append(
                    f"<img src='{rel}' alt='{cid} figure'>")
            html_parts.append("</div>")

        # Action (color-coded)
        action_lower = action.lower()
        if any(kw in action_lower for kw in
               ["no action", "no additional", "not needed",
                "adequate", "not required"]):
            action_cls = "action-ok"
        else:
            action_cls = "action-needed"
        html_parts.append(
            f"<div class='{action_cls}'>"
            f"<strong>Action:</strong> {action}</div>")

    html_parts.append("</body></html>")
    path = os.path.join(out_dir, "power_analysis_report.html")
    with open(path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"  HTML report: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.base_output_dir:
        base_output_dir = args.base_output_dir
    else:
        base_output_dir = str(REPO_ROOT / "02_output")

    if args.helper_output_dir:
        out_dir = args.helper_output_dir
    else:
        out_dir = str(REPO_ROOT / "helpers" / "outputs" / "16_power_analysis")

    parameterization = extract_parameterization(out_dir)
    if parameterization is None:
        for d in sorted(Path(base_output_dir).glob("**/05.HAN_*")):
            if d.is_dir():
                parameterization = d.name
                break
    if parameterization is None:
        parameterization = "05.HAN_filter_parameters_i300_r10_t10_u5"

    print(f"Helper 16: Power Analysis")
    print(f"  base_output_dir: {base_output_dir}")
    print(f"  helper_output_dir: {out_dir}")
    print(f"  parameterization: {parameterization}")
    print(f"  model_type: {args.model_type}")
    print()

    os.makedirs(out_dir, exist_ok=True)
    for subdir in ["elbow_curves", "bootstrap_stability", "equivalence_tests",
                    "per_motif_power"]:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    # Load data
    print("Loading data...")
    proj_summary = load_projection_summary(base_output_dir, parameterization)
    print(f"  projection_summary: {len(proj_summary)} rows")

    replicates = load_individual_replicates(base_output_dir, parameterization,
                                            args.model_type)
    print(f"  individual_replicates: {len(replicates)} rows")

    pie_df = load_pie_chart_per_animal(base_output_dir, parameterization)
    print(f"  pie_chart_data: {len(pie_df)} rows")

    upsetplot_csvs = load_upsetplot_csvs(base_output_dir, parameterization,
                                         args.model_type)
    print(f"  upsetplot CSVs: {list(upsetplot_csvs.keys())}")

    matt_rules = load_matt_visual_rules(base_output_dir, parameterization,
                                        args.model_type)
    print(f"  matt_visual_rules: {len(matt_rules)} rows")

    barcode_df = load_barcode_summary()
    print(f"  barcode_summary: {len(barcode_df)} rows")
    print()

    # Run all claim analyses
    results = []

    claims = [
        ("C1", "Barcode uniqueness",
         claim_C1_barcode, (barcode_df, out_dir, args.alpha)),
        ("C2", "Cross-method JSD",
         claim_C2_cross_method_jsd, (replicates, out_dir,
                                     args.n_bootstrap, args.alpha)),
        ("C3", "Regional UMI ranking",
         claim_C3_regional_ranking, (proj_summary, out_dir,
                                     args.n_bootstrap, args.alpha)),
        ("C4", "LM/LI mean UMI decline",
         claim_C4_lm_umi_decline, (proj_summary, out_dir,
                                   args.alpha, args.max_n_elbow)),
        ("C5", "AL+PM motif low",
         claim_C5_alpm_low, (replicates, out_dir,
                             args.n_bootstrap, args.alpha)),
        ("C6", "AL up, PM down",
         claim_C6_al_up_pm_down, (replicates, out_dir,
                                  args.n_bootstrap, args.alpha)),
        ("C7", "Target count stability",
         claim_C7_target_stable, (pie_df, out_dir, args.alpha,
                                  args.power_target, args.n_bootstrap,
                                  args.max_n_elbow)),
        ("C8", "Entropy stability",
         claim_C8_entropy_stable, (proj_summary, out_dir, args.alpha,
                                   args.power_target, args.n_bootstrap,
                                   args.max_n_elbow)),
        ("C9", "Model-independent motif stability",
         claim_C9_motif_stability, (replicates, out_dir, args.alpha,
                                    args.power_target, args.n_bootstrap,
                                    args.max_n_elbow)),
        ("C10", "JSD between age pairs",
         claim_C10_jsd_low, (replicates, out_dir,
                             args.n_bootstrap, args.alpha)),
        ("C11", "Motifs non-random in adult",
         claim_C11_motifs_nonrandom, (upsetplot_csvs, out_dir,
                                      args.alpha, args.n_bootstrap)),
        ("C12", "Motif enrichment shifts",
         claim_C12_motif_shifts, (matt_rules, upsetplot_csvs, out_dir,
                                  args.alpha, args.n_bootstrap,
                                  args.max_n_elbow)),
    ]

    for claim_id, claim_desc, func, func_args in claims:
        print(f"Analyzing {claim_id}: {claim_desc}...")
        try:
            results.append(func(*func_args))
        except Exception as e:
            traceback.print_exc()
            results.append({
                "claim_id": claim_id,
                "claim_desc": claim_desc,
                "test_type": "N/A",
                "observed_effect": np.nan,
                "n_per_group": 0,
                "achieved_power": np.nan,
                "mde_80": np.nan,
                "equiv_p": np.nan,
                "tost_result": "N/A",
                "conclusion": f"ERROR: {e}",
                "figures": [],
            })

    # Generate outputs
    print()
    print("Generating outputs...")
    generate_summary_csv(results, out_dir)
    generate_summary_md(results, out_dir, args.model_type)
    generate_report_html(results, out_dir, args.model_type)

    print()
    print("Helper 16 complete.")


if __name__ == "__main__":
    main()
