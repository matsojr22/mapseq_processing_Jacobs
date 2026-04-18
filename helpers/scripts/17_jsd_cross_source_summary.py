"""
Helper 17: Cross-source JSD summary, mean per-motif JS from helper 05 text,
and homogeneity tests (chi-squared + Monte Carlo) for P12–P20 and P20–P60.

Requires sibling outputs from helpers 01 and 05 under the same *_helpers parent.
"""

from __future__ import annotations

import argparse
import ast
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

RISK_AND_CLARITY = """
================================================================================
RISK AND CLARITY (READ THIS)
================================================================================
1) Helper 01 (distribution_jsd_pairwise.csv) reports scipy.spatial.distance.
   jensenshannon distance between MEAN per-animal frequency vectors over ALL
   motifs (global row). Values are typically much larger than helper 05 lines.

2) Helper 05 (motif_transition_significance_summary_*.txt) reports, for EACH
   motif separately, jensenshannon([p_motif, 1-p_motif], [q_motif, 1-q_motif])**2
   — i.e. squared JS on a 2-bin (motif vs rest) vector. The arithmetic mean of
   those per-motif values is NOT the same quantity as helper 01 global JSD and
   is not on the same scale; do not compare the two scalars directly.

3) Chi-squared and Monte Carlo tests below test HOMOGENEITY of the AGGREGATE
   motif count distribution between two ages (rebuilt from the same aggregate
   upsetplot CSVs helper 05 uses). They do not test helper 01 JSD and they do
   not average per-motif JS from helper 05.

4) Sparse motifs: rows with expected frequency < 5 (under the chi-squared model)
   are merged until min expected >= 5 or only 2 rows remain; the same collapsed
   table is used for the Monte Carlo statistic.
================================================================================
"""


def parse_helpers_layout(helper_output_dir: Path) -> tuple[Path, Path, Path]:
    out = Path(helper_output_dir).resolve()
    parent = out.parent
    d01 = parent / "01_motif_analysis_per_animal"
    d05 = parent / "05_motif_analysis"
    if not d01.is_dir():
        raise FileNotFoundError(f"Missing helper 01 directory: {d01}")
    if not d05.is_dir():
        raise FileNotFoundError(f"Missing helper 05 directory: {d05}")
    return out, d01, d05


def parse_parameterization_filter(helper_output_dir: Path) -> tuple[str | None, str | None]:
    for part in Path(helper_output_dir).resolve().parts:
        if part.startswith(("01.", "02.", "03.", "04.", "05.")):
            if "_helpers" in part:
                parameterization_filter = part.split("_helpers")[0]
            else:
                parameterization_filter = part
            if "minimal" in part:
                filter_type = "minimal"
            elif "medium" in part:
                filter_type = "medium"
            elif "strict" in part:
                filter_type = "strict"
            elif "extreme" in part:
                filter_type = "extreme"
            elif "HAN" in part:
                filter_type = "HAN"
            else:
                filter_type = None
            return parameterization_filter, filter_type
    return None, None


def upsetplot_globs_for_age(
    output_dir: Path,
    age: str,
    model_type: str,
    parameterization_filter: str | None,
    filter_type: str | None,
) -> list[str]:
    age_lower = age.lower()
    if parameterization_filter and filter_type:
        param_path = output_dir / age_lower / parameterization_filter
        if not param_path.exists():
            return []
        return [
            str(param_path / "analysis" / model_type / f"*ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"*alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"{age.upper()}_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"{age.lower()}_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"{age.upper()}_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"{age.lower()}_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"P60_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / model_type / f"P60_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
            str(param_path / "analysis" / f"*ALL_{filter_type}_filters_upsetplot.csv"),
            str(param_path / "analysis" / f"*alL_{filter_type}_filters_upsetplot.csv"),
        ]
    return [
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"*ALL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"*alL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"{age.upper()}_ALL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"{age.lower()}_ALL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"{age.upper()}_alL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / model_type / f"{age.lower()}_alL_HAN_filters_upsetplot_{model_type}.csv"),
        str(output_dir / age_lower / "**" / "analysis" / f"*ALL_HAN_filters_upsetplot.csv"),
        str(output_dir / age_lower / "**" / "analysis" / f"*alL_HAN_filters_upsetplot.csv"),
    ]


def load_age_motif_observed_counts(
    output_dir: Path,
    age: str,
    model_type: str,
    parameterization_filter: str | None,
    filter_type: str | None,
) -> dict[str, float]:
    patterns = upsetplot_globs_for_age(
        output_dir, age, model_type, parameterization_filter, filter_type
    )
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = list(set(files))
    if not files:
        return {}
    file_path = files[0]
    df_temp = pd.read_csv(file_path)
    rows_to_drop = []
    for idx, row in df_temp.iterrows():
        motifs_str = row["Motifs"]
        observed_count = float(row["Observed"])
        if observed_count == 0.0:
            rows_to_drop.append(idx)
            continue
        try:
            if isinstance(motifs_str, str):
                motifs_list = ast.literal_eval(motifs_str)
            else:
                motifs_list = motifs_str
            if not isinstance(motifs_list, list):
                rows_to_drop.append(idx)
            elif len(motifs_list) == 0:
                rows_to_drop.append(idx)
            elif all(not m or str(m).strip() == "" for m in motifs_list):
                rows_to_drop.append(idx)
        except Exception:
            if observed_count == 0.0:
                rows_to_drop.append(idx)
    df_filtered = df_temp.drop(rows_to_drop)
    counts: dict[str, float] = {}
    for _, row in df_filtered.iterrows():
        motifs_str = row["Motifs"]
        try:
            if isinstance(motifs_str, str):
                motifs_list = ast.literal_eval(motifs_str)
            else:
                motifs_list = motifs_str
            if not isinstance(motifs_list, list) or len(motifs_list) == 0:
                continue
            if all(not m or str(m).strip() == "" for m in motifs_list):
                continue
            motif_key = str(sorted(motifs_list))
            obs = float(row["Observed"])
            counts[motif_key] = counts.get(motif_key, 0.0) + obs
        except Exception:
            continue
    return counts


def build_kx2_counts(
    ca: dict[str, float], cb: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    keys = sorted(set(ca) | set(cb))
    col0 = [ca.get(k, 0.0) for k in keys]
    col1 = [cb.get(k, 0.0) for k in keys]
    return np.array([col0, col1], dtype=float).T, keys


def collapse_low_expected(
    table: np.ndarray, labels: list[str], min_exp: float = 5.0
) -> tuple[np.ndarray, list[str], str]:
    obs = np.array(table, dtype=float)
    labs = list(labels)
    note_parts = []
    while obs.shape[0] > 2:
        _, _, _, expected = chi2_contingency(obs)
        if expected.min() >= min_exp:
            break
        row_sums = obs.sum(axis=1)
        i = int(np.argmin(row_sums))
        j = i + 1 if i < obs.shape[0] - 1 else i - 1
        new_row = obs[i] + obs[j]
        mask = np.ones(obs.shape[0], dtype=bool)
        mask[i] = mask[j] = False
        obs = np.vstack([obs[mask], new_row])
        merged = f"{labs[i]} + {labs[j]}"
        labs = [labs[k] for k in range(len(labs)) if k not in (i, j)] + [merged]
        note_parts.append(f"merged({i},{j})")
    detail = "; ".join(note_parts) if note_parts else "no merges"
    return obs, labs, detail


def pearson_chi2_stat_only(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    return float(chi2)


def monte_carlo_homogeneity_pvalue(
    collapsed: np.ndarray, rng: np.random.Generator, n_sim: int
) -> float:
    n1 = int(round(collapsed[:, 0].sum()))
    n2 = int(round(collapsed[:, 1].sum()))
    p_hat = collapsed.sum(axis=1) / (n1 + n2)
    p_hat = np.clip(p_hat, 1e-15, 1.0)
    p_hat = p_hat / p_hat.sum()
    obs_stat = pearson_chi2_stat_only(collapsed)
    exceed = 0
    for _ in range(n_sim):
        c1 = rng.multinomial(n1, p_hat)
        c2 = rng.multinomial(n2, p_hat)
        sim = np.column_stack([c1, c2])
        if pearson_chi2_stat_only(sim) >= obs_stat - 1e-12:
            exceed += 1
    return (1 + exceed) / (1 + n_sim)


def parse_transition_js_values(text: str) -> dict[str, list[float]]:
    sections: dict[str, list[float]] = {}
    current = None
    header_re = re.compile(r"^(P\d+)\s+vs\s+(P\d+)\s*$")
    line_re = re.compile(r"JS Divergence = ([^,]+)")
    for line in text.splitlines():
        m = header_re.match(line.strip())
        if m:
            current = f"{m.group(1)}_vs_{m.group(2)}"
            sections[current] = []
            continue
        if current and "JS Divergence" in line:
            jm = line_re.search(line)
            if jm:
                val = jm.group(1).strip()
                if val.lower() == "nan":
                    sections[current].append(float("nan"))
                else:
                    sections[current].append(float(val))
    return sections


def mean_jsd_report_for_transition(vals: list[float]) -> tuple[float, float, int, int]:
    n_total = len(vals)
    finite = [v for v in vals if not np.isnan(v)]
    n_f = len(finite)
    mean_f = float(np.mean(finite)) if n_f else float("nan")
    mean_z = float(np.mean([0.0 if np.isnan(v) else v for v in vals])) if vals else float("nan")
    return mean_f, mean_z, n_f, n_total


def write_mean_jsd_txt(
    path: Path,
    model: str,
    source_txt: Path,
    sections: dict[str, list[float]],
    transitions_report: list[tuple[str, str]],
) -> None:
    lines = [
        "=" * 80,
        "PRODUCED BY helpers/scripts/17_jsd_cross_source_summary.py",
        "=" * 80,
        f"Model: {model}",
        f"Source: {source_txt}",
        "",
        "Per transition: mean of per-motif JS Divergence lines (helper 05; squared JS on 2 bins).",
        "",
    ]
    for key, title in transitions_report:
        vals = sections.get(key, [])
        mf, mz, nf, nt = mean_jsd_report_for_transition(vals)
        lines.append(title)
        lines.append(f"  Mean JS divergence, finite only: {mf:.6f}  (n={nf}/{nt})")
        lines.append(f"  Mean JS divergence, nan as 0.0:   {mz:.6f}  (n={nt})")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_global_jsd_row(df: pd.DataFrame, a: str, b: str) -> float | None:
    g = df[(df["normalization"] == "global") & (df["domain"] == "all")]
    for _, row in g.iterrows():
        x, y = row["timepoint_a"], row["timepoint_b"]
        if {x, y} == {a, b}:
            return float(row["jsd"])
    return None


def discover_models(dir_01: Path, only: list[str] | None) -> list[str]:
    out = []
    for d in sorted(dir_01.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "cross_age" / "distribution_jsd_pairwise.csv").is_file():
            continue
        name = d.name
        if only and name not in only:
            continue
        out.append(name)
    return out


def process_model(
    model: str,
    base_output: Path,
    dir_01: Path,
    dir_05: Path,
    out_dir: Path,
    parameterization_filter: str | None,
    filter_type: str | None,
    n_sim: int,
    rng: np.random.Generator,
    random_seed: int,
) -> None:
    model_out = out_dir / model
    model_out.mkdir(parents=True, exist_ok=True)

    jsd_csv = dir_01 / model / "cross_age" / "distribution_jsd_pairwise.csv"
    run_summary = dir_01 / model / "cross_age" / "analysis_run_summary.txt"
    trans_txt = dir_05 / f"motif_transition_significance_summary_{model}.txt"

    df_jsd = pd.read_csv(jsd_csv)
    jsd_p12_p20 = get_global_jsd_row(df_jsd, "P12", "P20")
    jsd_p20_p60 = get_global_jsd_row(df_jsd, "P20", "P60")

    trans_body = trans_txt.read_text(encoding="utf-8") if trans_txt.is_file() else ""
    sections = parse_transition_js_values(trans_body)

    transitions_report = [
        ("P12_vs_P20", "P12 vs P20"),
        ("P20_vs_P60", "P20 vs P60"),
    ]
    write_mean_jsd_txt(
        model_out / f"motif_transition_mean_jsd_{model}.txt",
        model,
        trans_txt,
        sections,
        transitions_report,
    )

    chi_rows = []
    mc_rows = []
    pair_specs = [("P12", "P20", "p12", "p20"), ("P20", "P60", "p20", "p60")]

    for lab_a, lab_b, age_a, age_b in pair_specs:
        ca = load_age_motif_observed_counts(
            base_output, age_a, model, parameterization_filter, filter_type
        )
        cb = load_age_motif_observed_counts(
            base_output, age_b, model, parameterization_filter, filter_type
        )
        if not ca or not cb:
            chi_rows.append(
                {
                    "transition": f"{lab_a}_vs_{lab_b}",
                    "error": "missing aggregate counts for one age",
                }
            )
            mc_rows.append(
                {
                    "transition": f"{lab_a}_vs_{lab_b}",
                    "observed_chi2_statistic": np.nan,
                    "B": n_sim,
                    "pvalue_mc": np.nan,
                    "random_seed": random_seed,
                }
            )
            continue
        raw, keys = build_kx2_counts(ca, cb)
        collapsed, _, merge_note = collapse_low_expected(raw, keys)
        chi2, p_asym, dof, _exp = chi2_contingency(collapsed)
        obs_stat = float(chi2)
        p_mc = monte_carlo_homogeneity_pvalue(collapsed, rng, n_sim)
        chi_rows.append(
            {
                "transition": f"{lab_a}_vs_{lab_b}",
                "chi2_statistic": obs_stat,
                "df": dof,
                "pvalue_asymptotic": float(p_asym),
                "n_motif_rows_collapsed": collapsed.shape[0],
                "N_age_a": int(collapsed[:, 0].sum()),
                "N_age_b": int(collapsed[:, 1].sum()),
                "collapse_merge_note": merge_note,
            }
        )
        mc_rows.append(
            {
                "transition": f"{lab_a}_vs_{lab_b}",
                "observed_chi2_statistic": obs_stat,
                "B": n_sim,
                "pvalue_mc": float(p_mc),
                "random_seed": random_seed,
            }
        )

    pd.DataFrame(chi_rows).to_csv(
        model_out / f"transition_homogeneity_chi2_{model}.csv", index=False
    )
    pd.DataFrame(mc_rows).to_csv(
        model_out / f"transition_homogeneity_permutation_{model}.csv", index=False
    )

    global_csv_excerpt = df_jsd[
        (df_jsd["normalization"] == "global") & (df_jsd["domain"] == "all")
    ].to_string(index=False)
    run_excerpt = ""
    if run_summary.is_file():
        run_excerpt = run_summary.read_text(encoding="utf-8")

    mean_p12_p20 = sections.get("P12_vs_P20", [])
    mean_p20_p60 = sections.get("P20_vs_P60", [])
    mf12, mz12, nf12, nt12 = mean_jsd_report_for_transition(mean_p12_p20)
    mf20, mz20, nf20, nt20 = mean_jsd_report_for_transition(mean_p20_p60)

    summary_a = [
        f"Helper 17 cross-source JSD summary — model: {model}",
        "",
        RISK_AND_CLARITY.strip(),
        "",
        "Paths:",
        f"  helper 01 JSD CSV: {jsd_csv}",
        f"  helper 01 run log: {run_summary}",
        f"  helper 05 transitions: {trans_txt}",
        "",
        "=== Global JSD (helper 01, distribution_jsd_pairwise.csv, global/all) ===",
        f"P12 vs P20: {jsd_p12_p20}",
        f"P20 vs P60: {jsd_p20_p60}",
        "",
        "Full global pairwise rows:",
        global_csv_excerpt,
        "",
        "=== analysis_run_summary.txt (helper 01, full copy) ===",
        run_excerpt if run_excerpt else "(file missing)",
        "",
        "=== Mean per-motif JS (helper 05 text; see motif_transition_mean_jsd_*.txt) ===",
        f"P12 vs P20: mean finite={mf12:.6f} (n={nf12}/{nt12}); nan->0 mean={mz12:.6f}",
        f"P20 vs P60: mean finite={mf20:.6f} (n={nf20}/{nt20}); nan->0 mean={mz20:.6f}",
        "",
        "=== motif_transition_significance excerpt (first 40 lines) ===",
        "\n".join(trans_body.splitlines()[:40]),
    ]
    (model_out / f"jsd_cross_source_summary_{model}.txt").write_text(
        "\n".join(summary_a) + "\n", encoding="utf-8"
    )

    chi_df = pd.DataFrame(chi_rows)
    mc_df = pd.DataFrame(mc_rows)
    lines_b = [
        f"Helper 17 statistical inference — model: {model}",
        "",
        RISK_AND_CLARITY.strip(),
        "",
        "Null hypothesis (chi-squared and Monte Carlo): the same multinomial motif",
        "distribution generates counts at both ages (homogeneity), given fixed column totals.",
        "Tables use aggregate Observed counts from upsetplot CSVs (helper 05 logic).",
        "Low-expected rows were merged until min expected >= 5 or only 2 rows remain.",
        "",
    ]
    for _, row in chi_df.iterrows():
        lines_b.append(f"Transition: {row.get('transition', '')}")
        if "error" in row and pd.notna(row.get("error")):
            lines_b.append(f"  {row['error']}")
            lines_b.append("")
            continue
        cs = row.get("chi2_statistic", float("nan"))
        cp = row.get("pvalue_asymptotic", float("nan"))
        lines_b.append(
            f"  Pearson chi-squared (collapsed table): statistic={cs:.4f}, "
            f"df={row.get('df', '')}, asymptotic p={cp:.6e}"
        )
        mc = mc_df[mc_df["transition"] == row["transition"]]
        if len(mc):
            lines_b.append(
                f"  Monte Carlo p-value (same statistic, {n_sim} sims): {mc.iloc[0]['pvalue_mc']:.6e}"
            )
        lines_b.append(
            f"  Collapsed motif rows: {row.get('n_motif_rows_collapsed', '')}; "
            f"N ages: {row.get('N_age_a', '')}, {row.get('N_age_b', '')}"
        )
        lines_b.append(f"  Row merge log: {row.get('collapse_merge_note', '')}")
        lines_b.append("")
    (model_out / f"jsd_inference_summary_{model}.txt").write_text(
        "\n".join(lines_b) + "\n", encoding="utf-8"
    )


def main() -> None:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Helper 17: JSD cross-source summary and homogeneity tests")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=None,
        help="02_output root (default: REPO_ROOT/02_output)",
    )
    parser.add_argument(
        "--helper_output_dir",
        type=str,
        required=True,
        help="Output for helper 17 (e.g. .../_helpers/17_jsd_cross_source). "
        "Parent must contain 01_motif_analysis_per_animal and 05_motif_analysis.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all with helper 01 JSD CSV)",
    )
    parser.add_argument("--n_perm", type=int, default=10000, help="Monte Carlo replicates")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    base_output = Path(args.base_output_dir or (repo_root / "02_output")).resolve()
    out_root, dir_01, dir_05 = parse_helpers_layout(Path(args.helper_output_dir))

    param_f, ftype = parse_parameterization_filter(out_root)
    if param_f:
        print(f"Parameterization filter: {param_f}, filter type: {ftype}")

    only = [m.strip() for m in args.models.split(",")] if args.models else None
    models = discover_models(dir_01, only)
    if not models:
        raise SystemExit("No models found with cross_age/distribution_jsd_pairwise.csv")

    rng = np.random.default_rng(args.random_seed)
    for model in models:
        print(f"Processing {model}...")
        process_model(
            model,
            base_output,
            dir_01,
            dir_05,
            out_root,
            param_f,
            ftype,
            args.n_perm,
            rng,
            args.random_seed,
        )

    print(f"Done. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
