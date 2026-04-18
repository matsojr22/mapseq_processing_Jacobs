"""
Helper 18: mean-JSD-based transition testing.

This helper focuses on motif-level "JS Divergence" values reported by helper 05
in motif_transition_significance_summary_{model}.txt (JS^2 on a 2-bin motif vs rest
vector). It compares transition means (P12 vs P20 vs P20 vs P60) using:
- permutation test on delta of means
- bootstrap CI on delta of means
- chi-squared test on binned JS values (transition x bin contingency)

Outputs are written under the provided helper_output_dir, one subdir per model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


RISK_AND_CLARITY = """
================================================================================
RISK AND CLARITY (READ THIS)
================================================================================
1) This helper uses motif-level JS Divergence values from helper 05
   (motif_transition_significance_summary_*.txt). Those values are:
     jensenshannon([p, 1-p], [q, 1-q])**2
   where p and q are motif percentages at two ages. This is a squared JS on a
   2-bin "motif vs rest" vector.

2) These motif-level JS^2 values (and their mean) are NOT the same as helper 01's
   global distribution JSD between ages. Do not compare the scalar magnitudes
   directly across helpers.

3) The permutation/bootstrap here tests whether the mean motif-level JS^2 differs
   between transitions (P12_vs_P20 vs P20_vs_P60), treating motifs as the unit
   of analysis (one JS per motif per transition).

4) The chi-squared test here is on a binned JS category table (transition x JS_bin),
   not on aggregate motif counts.
================================================================================
""".strip()


TRANSITIONS = [
    ("P12_vs_P20", "P12", "P20"),
    ("P20_vs_P60", "P20", "P60"),
]


def parse_transition_js_values(text: str) -> dict[str, list[float]]:
    """Parse helper 05 transition text -> transition key -> list of JS values (may include nan)."""
    sections: dict[str, list[float]] = {}
    current = None
    header_re = re.compile(r"^(P\d+)\s+vs\s+(P\d+)\s*$")
    js_re = re.compile(r"JS Divergence = ([^,]+)")

    for line in text.splitlines():
        m = header_re.match(line.strip())
        if m:
            current = f"{m.group(1)}_vs_{m.group(2)}"
            sections[current] = []
            continue
        if current and "JS Divergence" in line:
            jm = js_re.search(line)
            if not jm:
                continue
            raw = jm.group(1).strip()
            # Some older outputs can end up with a trailing ')' (e.g. 'nan)').
            raw_clean = raw.strip().rstrip(")")
            if raw_clean.lower() == "nan":
                sections[current].append(float("nan"))
            else:
                sections[current].append(float(raw_clean))

    return sections


def apply_nan_policy(vals: list[float], nan_policy: str) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if nan_policy == "finite_only":
        return arr[~np.isnan(arr)]
    if nan_policy == "nan_as_zero":
        out = arr.copy()
        out[np.isnan(out)] = 0.0
        return out
    raise ValueError(f"Unknown nan_policy: {nan_policy}")


def permutation_pvalue_delta_means(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_perm: int
) -> tuple[float, float]:
    """Two-sided permutation test for delta = mean(b)-mean(a). Returns (delta_obs, p)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    delta_obs = float(np.mean(b) - np.mean(a))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        a_p = perm[:n_a]
        b_p = perm[n_a:]
        delta_p = float(np.mean(b_p) - np.mean(a_p))
        if abs(delta_p) >= abs(delta_obs) - 1e-15:
            exceed += 1
    p = (1 + exceed) / (1 + n_perm)
    return delta_obs, float(p)


def bootstrap_ci_delta_means(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for delta=mean(b)-mean(a)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        deltas[i] = float(np.mean(b_s) - np.mean(a_s))
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return lo, hi


def make_bins(values: np.ndarray, binning: str, n_bins: int = 4) -> tuple[np.ndarray, str]:
    """
    Return integer bin ids and a human-readable description.
    Uses pooled values across transitions for stability.
    """
    if len(values) == 0:
        return np.array([], dtype=int), "no_values"
    if binning == "quantile":
        # qcut can fail with many ties -> fall back.
        try:
            cats = pd.qcut(values, q=n_bins, duplicates="drop")
            # codes: -1 indicates NaN, but values should be finite here
            codes = cats.codes.astype(int)
            return codes, f"quantile_qcut(q={n_bins}, actual_bins={len(cats.categories)})"
        except Exception:
            pass

    # fixed fallback: bins chosen for the observed scale in helper 05 outputs
    edges = np.array([0.0, 5e-4, 1e-3, 2e-3, 5e-3, np.inf], dtype=float)
    codes = np.digitize(values, edges, right=False) - 1
    return codes.astype(int), "fixed_edges([0,0.0005,0.001,0.002,0.005,inf])"


def chi2_on_binned_js(
    a: np.ndarray, b: np.ndarray, binning: str
) -> tuple[float, int, float, str, pd.DataFrame]:
    """
    Chi-squared test on transition x JS_bin table.
    Returns (chi2, df, p, binning_note, contingency_df).
    """
    pooled = np.concatenate([a, b]) if (len(a) and len(b)) else np.array([], dtype=float)
    if len(pooled) == 0:
        return float("nan"), 0, float("nan"), "no_data", pd.DataFrame()

    bin_ids, note = make_bins(pooled, binning=binning)
    # split back to transitions
    a_bins = bin_ids[: len(a)]
    b_bins = bin_ids[len(a) :]
    max_bin = int(bin_ids.max()) if len(bin_ids) else -1
    n_bins = max_bin + 1
    table = np.zeros((2, n_bins), dtype=int)
    for x in a_bins:
        table[0, int(x)] += 1
    for x in b_bins:
        table[1, int(x)] += 1
    chi2, p, dof, expected = chi2_contingency(table)

    cols = [f"bin_{i}" for i in range(n_bins)]
    contingency = pd.DataFrame(table, index=["P12_vs_P20", "P20_vs_P60"], columns=cols)
    return float(chi2), int(dof), float(p), note, contingency


def discover_models(dir_05: Path, only: list[str] | None) -> list[str]:
    out: list[str] = []
    for p in sorted(dir_05.glob("motif_transition_significance_summary_*.txt")):
        name = p.name.replace("motif_transition_significance_summary_", "").replace(".txt", "")
        if only and name not in only:
            continue
        out.append(name)
    return out


def main() -> None:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Helper 18: mean-JSD-based transition testing")
    parser.add_argument(
        "--helper_output_dir",
        type=str,
        required=True,
        help="Output directory for helper 18 (e.g. .../_helpers/18_mean_jsd_transition_tests). "
        "Parent must contain 05_motif_analysis.",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list (default: discover)")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--nan_policy",
        type=str,
        default="both",
        choices=["both", "finite_only", "nan_as_zero"],
        help="How to treat JS Divergence=nan rows.",
    )
    parser.add_argument(
        "--binning",
        type=str,
        default="quantile",
        choices=["quantile", "fixed"],
        help="Binning strategy for chi-squared contingency.",
    )
    args = parser.parse_args()

    out_root = Path(args.helper_output_dir).resolve()
    helpers_root = out_root.parent
    dir_05 = helpers_root / "05_motif_analysis"
    if not dir_05.is_dir():
        raise FileNotFoundError(f"Missing helper 05 directory: {dir_05}")
    out_root.mkdir(parents=True, exist_ok=True)

    only = [m.strip() for m in args.models.split(",")] if args.models else None
    models = discover_models(dir_05, only)
    if not models:
        raise SystemExit("No models found (missing motif_transition_significance_summary_*.txt).")

    rng = np.random.default_rng(args.random_seed)

    for model in models:
        model_out = out_root / model
        model_out.mkdir(parents=True, exist_ok=True)

        trans_txt = dir_05 / f"motif_transition_significance_summary_{model}.txt"
        body = trans_txt.read_text(encoding="utf-8") if trans_txt.is_file() else ""
        sections = parse_transition_js_values(body)

        # Build motif-level values dataframe
        rows = []
        for key, a, b in TRANSITIONS:
            vals = sections.get(key, [])
            for v in vals:
                rows.append({"model": model, "transition": key, "js": v})
        values_df = pd.DataFrame(rows)
        values_df.to_csv(model_out / f"mean_jsd_transition_values_{model}.csv", index=False)

        # Per-policy analysis
        policies = ["finite_only", "nan_as_zero"] if args.nan_policy == "both" else [args.nan_policy]
        infer_rows = []
        summary_lines = [
            f"Helper 18 mean-JSD transition summary — model: {model}",
            "",
            RISK_AND_CLARITY,
            "",
            f"Source: {trans_txt}",
            "",
        ]

        for pol in policies:
            a_vals = apply_nan_policy(sections.get("P12_vs_P20", []), pol)
            b_vals = apply_nan_policy(sections.get("P20_vs_P60", []), pol)
            mean_a = float(np.mean(a_vals)) if len(a_vals) else float("nan")
            mean_b = float(np.mean(b_vals)) if len(b_vals) else float("nan")
            summary_lines.append(f"nan_policy = {pol}")
            summary_lines.append(f"  P12_vs_P20: mean={mean_a:.6f}, n={len(a_vals)}")
            summary_lines.append(f"  P20_vs_P60: mean={mean_b:.6f}, n={len(b_vals)}")

            delta_obs, p_perm = permutation_pvalue_delta_means(a_vals, b_vals, rng, args.n_perm)
            ci_lo, ci_hi = bootstrap_ci_delta_means(a_vals, b_vals, rng, args.n_boot)
            chi2, dof, p_chi, bin_note, cont = chi2_on_binned_js(
                a_vals[~np.isnan(a_vals)], b_vals[~np.isnan(b_vals)], binning=args.binning
            )

            # Write contingency for this policy
            if not cont.empty:
                cont.to_csv(model_out / f"mean_jsd_chi2_contingency_{model}_{pol}.csv")

            infer_rows.append(
                {
                    "model": model,
                    "nan_policy": pol,
                    "mean_P12_vs_P20": mean_a,
                    "mean_P20_vs_P60": mean_b,
                    "delta_mean": delta_obs,
                    "perm_n": args.n_perm,
                    "perm_pvalue_two_sided": p_perm,
                    "boot_n": args.n_boot,
                    "boot_ci_lo": ci_lo,
                    "boot_ci_hi": ci_hi,
                    "chi2_statistic": chi2,
                    "chi2_df": dof,
                    "chi2_pvalue": p_chi,
                    "chi2_binning": bin_note,
                    "random_seed": args.random_seed,
                }
            )
            summary_lines.append("")

        infer_df = pd.DataFrame(infer_rows)
        infer_df.to_csv(model_out / f"mean_jsd_permutation_bootstrap_{model}.csv", index=False)

        # Summaries
        (model_out / f"mean_jsd_transition_summary_{model}.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        lines_inf = [
            f"Helper 18 inference summary — model: {model}",
            "",
            RISK_AND_CLARITY,
            "",
            "Question: do motif-level JS^2 values differ in mean between P12_vs_P20 and P20_vs_P60?",
            "Permutation test: shuffles motif-level JS values between transitions (two-sided on delta of means).",
            "Bootstrap: percentile CI on delta of means (resample within transition).",
            "Chi-squared: transition x binned(JS) contingency.",
            "",
        ]
        for _, row in infer_df.iterrows():
            pol = row["nan_policy"]
            lines_inf.append(f"nan_policy = {pol}")
            lines_inf.append(
                f"  delta_mean (P20_vs_P60 - P12_vs_P20) = {row['delta_mean']:.6f}"
            )
            lines_inf.append(
                f"  permutation p (two-sided, n={int(row['perm_n'])}) = {row['perm_pvalue_two_sided']:.6e}"
            )
            lines_inf.append(
                f"  bootstrap 95% CI (n={int(row['boot_n'])}) = [{row['boot_ci_lo']:.6f}, {row['boot_ci_hi']:.6f}]"
            )
            lines_inf.append(
                f"  chi2 p (binning={row['chi2_binning']}) = {row['chi2_pvalue']:.6e}"
            )
            lines_inf.append("")

        (model_out / f"mean_jsd_inference_summary_{model}.txt").write_text(
            "\n".join(lines_inf) + "\n", encoding="utf-8"
        )

    print(f"Done. Outputs under: {out_root}")


if __name__ == "__main__":
    main()

