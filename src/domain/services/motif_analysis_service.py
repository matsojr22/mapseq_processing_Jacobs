import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.infrastructure.logger import Logger


class MotifAnalysisService:
    """Service for analyzing motifs and generating motif statistics"""

    def __init__(self, logger: Logger):
        self.logger = logger

    def analyze_motifs(
        self, matrix: np.ndarray, columns: List[str], pe_num: float, n0: float
    ) -> Dict[str, Any]:
        """
        Analyze motifs in the data matrix.

        Returns:
            Dictionary containing motif analysis results
        """
        try:
            self.logger.log_step("Motif analysis", "Analyzing motif patterns")

            # Convert matrix to DataFrame for processing
            df = pd.DataFrame(matrix, columns=columns)

            # Get target pie data
            target_pie_data = self._get_target_pie(df)

            # Get motif counts
            motif_counts = self._get_motif_counts(target_pie_data)

            # Get detailed motif analysis (placeholder for now)
            detailed_motif_analysis = self._get_detailed_motif_analysis(df)

            # Generate motif obs exp data
            motif_obs_exp_data = self._get_motif_obs_exp_data(df, pe_num, n0)

            return {
                "target_pie_data": target_pie_data,
                "motif_counts": motif_counts,
                "detailed_motif_analysis": detailed_motif_analysis,
                "motif_obs_exp_data": motif_obs_exp_data,
                "df": df,
                "normalized_matrix": df,  # Add normalized matrix for Extended Data Fig 10
                "n0": n0,  # Store n0 to avoid recomputation
            }

        except Exception as e:
            self.logger.log_error(e, "Analyzing motifs")
            raise

    def _get_target_pie(self, df: pd.DataFrame) -> np.ndarray:
        """
        For each cell (row), determine how many projections it makes
        """
        data = df.to_numpy(copy=True)
        cells, regions = data.shape
        res = []
        for cell in range(cells):
            num_targets = int(np.nonzero(data[cell])[0].shape[0])
            res.append(num_targets)
        return np.array(res)

    def _get_motif_counts(self, target_pie_data: np.ndarray) -> Dict[str, int]:
        """Get counts of cells by number of targets"""
        unique_targets, counts = np.unique(target_pie_data, return_counts=True)
        return {str(target): count for target, count in zip(unique_targets, counts)}

    def _get_detailed_motif_analysis(self, df: pd.DataFrame) -> List[List]:
        """
        Get detailed motif analysis data using the exact same logic as the original script.
        """
        # Generate all possible motifs using the same logic as original script
        motifs, motif_labels = self._gen_motifs(df.shape[1], df.columns)

        # Count motifs using the same logic as original script
        dcounts, cell_ids = self._count_motifs(df, motifs, return_ids=True)

        # Get detailed analysis using the same logic as original script
        return self._get_all_counts_nondf(df, motifs, dcounts, motif_labels)

    def _gen_motifs(
        self, r: int, labels: pd.Index
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Generate all possible motifs - same logic as original script"""
        from itertools import combinations

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1)
            )

        num_motifs = 2**r
        motifs = np.zeros((num_motifs, r)).astype(bool)
        motif_ids = list(powerset(np.arange(r)))
        motif_labels = []  # list of labels e.g. PFC-LH or PFC-LS-BNST

        for i in range(num_motifs):
            idx = motif_ids[i]
            motifs[i, idx] = True
            label = labels[np.array(idx)].tolist() if idx else [""]
            motif_labels.append(label)

        return motifs, motif_labels

    def _get_all_counts_nondf(
        self,
        df: pd.DataFrame,
        motifs: np.ndarray,
        counts: np.ndarray,
        labels: List[List[str]],
    ) -> List[List]:
        """
        Returns an array where each row is a motif and the columns are the counts of
        number of cells targeting each member of the motif (non-exclusive), total number of cells targeting any of
        the members of the motif, number of cells targeting all members of motif, and percentage exclusively targeting full
        motif (relative to any member of the motif)
        """
        retdf = []  # return list
        # each element is a list [Labels, R1 count, R2 count ... Rn count, Total Count, Motif Count, Motif Perc]
        for i, motif in enumerate(motifs):  # loop through motifs
            m = [index for (index, x) in enumerate(motif) if x]
            row = list(
                np.zeros(1 + len(m) + 3)
            )  # 1 (labels) + num-regions-in-motifs + 3 (total,motif count,motif perc)
            if len(m) < 1:
                continue
            sums = df.iloc[:, m].astype(bool).astype(int).sum().to_numpy()
            row[0] = labels[i]
            row[1 : len(m) + 1] = sums
            tot = sums.sum()

            # Prevent division by zero
            if tot == 0:
                row[len(m) + 1] = np.nan
                row[len(m) + 2] = np.nan
                row[len(m) + 3] = np.nan
            else:
                row[len(m) + 1] = tot
                row[len(m) + 2] = counts[i]
                row[len(m) + 3] = 100.0 * (counts[i] / tot)

            retdf.append(row)
        return retdf

    def _get_motif_obs_exp_data(
        self, df: pd.DataFrame, pe_num: float, n0: float
    ) -> Dict[str, Any]:
        """Generate motif observed vs expected data - same logic as original script"""
        # Generate motifs and counts
        motifs, motif_labels = self._gen_motifs(df.shape[1], df.columns)
        dcounts, cell_ids = self._count_motifs(df, motifs, return_ids=True)

        # Get expected counts
        exp_counts, motif_probs = self._get_expected_counts(motif_labels, pe_num, n0)

        # Create DataFrame with proper motif formatting using original logic
        def concatenate_list_data(slist, join="+"):
            result = []
            for i in slist:
                sub = ""
                for j in i:
                    if sub:
                        sub = sub + join + str(j)
                    else:
                        sub += str(j)
                result.append(sub)
            return result

        formatted_motifs = concatenate_list_data(motif_labels)

        df_obs_exp = pd.DataFrame(
            {
                "Motif": formatted_motifs,
                "Observed": dcounts,
                "Expected": exp_counts.astype(int),
            }
        )

        return {
            "df_obs_exp": df_obs_exp,
            "motif_labels": motif_labels,
            "dcounts": dcounts,
            "exp_counts": exp_counts,
            "motif_probs": motif_probs,
        }

    def _get_expected_counts(
        self, motif_labels: List[List[str]], prob_edge: float, n: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get expected counts - same logic as original script"""
        from sympy import N

        # Ensure variables are numeric
        prob_edge = (
            float(prob_edge.evalf())
            if hasattr(prob_edge, "evalf")
            else float(prob_edge)
        )
        n_motifs = len(motif_labels)
        res = np.zeros(n_motifs)
        probs = np.zeros(n_motifs)

        for i, motif in enumerate(motif_labels):
            e1 = int(len(motif))
            e2 = 7 - e1  # num_regions=7 as in original script
            p = (prob_edge**e1) * (1 - prob_edge) ** e2
            exp = float(N(p)) * n
            res[i] = exp
            probs[i] = p

        res[0] = 0  # Set first (null combination) to 0
        return res, probs

    def _count_motifs(
        self, df: pd.DataFrame, motifs: np.ndarray, return_ids: bool = False
    ):
        """Count motifs - same logic as original script"""
        cells, regions = df.shape
        data = df.to_numpy().astype(bool)
        counts = np.zeros(motifs.shape[0])
        cell_ids = []

        for i in range(motifs.shape[0]):  # loop through motifs
            cell_ids_ = []
            for j in range(data.shape[0]):  # loop through observed data cells X regions
                if np.array_equal(motifs[i], data[j]):
                    counts[i] = counts[i] + 1
                    cell_ids_.append(j)
            cell_ids.append(cell_ids_)

        if return_ids:
            return counts, cell_ids
        else:
            return counts

    def get_pie_chart_dataframe(self, target_pie_data: np.ndarray) -> pd.DataFrame:
        """Create the pie chart DataFrame in the original format"""
        unique_targets, counts = np.unique(target_pie_data, return_counts=True)

        c_row_names = ["1 target"]
        c_row_names += ["{} targets".format(i + 2) for i in range(len(counts) - 1)]

        return pd.DataFrame(counts, columns=["# Cells"], index=c_row_names)

    def get_motif_counts_dataframe(
        self, df: pd.DataFrame, motif_analysis_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create the motif counts DataFrame in the same format as original script"""
        # Use the detailed motif analysis data to create the same format as original
        detailed_analysis = motif_analysis_results["detailed_motif_analysis"]

        # Create DataFrame with the same structure as original
        motif_df = pd.DataFrame(
            columns=df.columns.tolist() + ["Total", "Motif Num", "Motif Perc"]
        )

        for i, row in enumerate(detailed_analysis):
            if len(row) < 4:  # Skip invalid rows
                continue

            motif_labels = row[0]
            if not motif_labels or motif_labels == [""]:
                continue

            # Create a row with the same structure as original
            df_row = {}

            # Set values for each region (leave as NaN initially, like original script)
            for col in df.columns:
                df_row[col] = np.nan

            # Set the motif values
            if len(row) > 1:
                # The row structure is: [labels, region1_count, region2_count, ..., total, motif_count, motif_perc]
                # We need to map the region counts to the correct columns
                motif_regions = motif_labels
                region_counts = (
                    row[1 : len(motif_regions) + 1]
                    if len(row) > len(motif_regions)
                    else []
                )

                for region, count in zip(motif_regions, region_counts):
                    if region in df.columns:
                        df_row[region] = float(count)

                # Set total, motif count, and motif percentage
                if len(row) > len(motif_regions) + 1:
                    df_row["Total"] = float(row[len(motif_regions) + 1])
                if len(row) > len(motif_regions) + 2:
                    df_row["Motif Num"] = float(row[len(motif_regions) + 2])
                if len(row) > len(motif_regions) + 3:
                    df_row["Motif Perc"] = float(row[len(motif_regions) + 3])

            motif_df.loc[i] = df_row

        return motif_df

    def get_upset_plot_data(
        self, df: pd.DataFrame, pe_num: float, n0: float
    ) -> Dict[str, Any]:
        """Generate upset plot data with same format as original script"""
        try:
            from scipy.stats import binomtest

            # Generate motifs and labels
            motifs, motif_labels = self._gen_motifs(df.shape[1], df.columns)

            # Count observed motifs
            dcounts, cell_ids = self._count_motifs(df, motifs, return_ids=True)

            # Calculate expected counts
            exp_counts, motif_probs = self._get_expected_counts(
                motif_labels, pe_num, n0
            )

            # Calculate effect sizes and p-values using original script logic
            def get_motif_sig_pts(
                dcounts,
                labels,
                prob_edge=pe_num,
                n0=n0,
                exclude_zeros=True,
                p_transform=lambda x: -1 * np.log10(x),
            ):
                num_motifs = dcounts.shape[0]
                expected, probs = self._get_expected_counts(labels, prob_edge, n0)
                assert dcounts.shape[0] == expected.shape[0]
                if exclude_zeros:
                    nonzid = np.nonzero(dcounts)[0]
                else:
                    nonzid = np.arange(dcounts.shape[0])
                num_nonzid_motifs = nonzid.shape[0]
                dcounts_ = dcounts[nonzid]
                expected_ = expected[nonzid]
                probs_ = probs[nonzid]
                # Effect size is log2(observed/expected)
                effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
                matches = np.zeros(num_nonzid_motifs)
                assert dcounts_.shape[0] == expected_.shape[0]
                dcounts_ = dcounts_.astype(int)
                for i in range(num_nonzid_motifs):
                    pi = max(probs_[i], 1e-10)  # avoid zero or very small probs
                    matches[i] = binomtest(int(dcounts_[i]), n=n0, p=pi).pvalue
                    matches[i] = max(matches[i], 1e-10)
                matches = p_transform(matches)
                return matches, effect_size

            # Calculate significance and effect sizes
            p_values, effect_sizes = get_motif_sig_pts(dcounts, motif_labels)

            # Calculate expected standard deviation
            expected_sd = np.sqrt(exp_counts * (1 - motif_probs))

            # Create degree and group columns
            degree = [len(motif) for motif in motif_labels]
            group = [1] * len(motif_labels)  # Default group

            # Format motifs for output
            def concatenate_list_data(slist, join="+"):
                result = []
                for i in slist:
                    sub = ""
                    for j in i:
                        if sub:
                            sub = sub + join + str(j)
                        else:
                            sub += str(j)
                    result.append(sub)
                return result

            formatted_motifs = concatenate_list_data(motif_labels)

            # Create DataFrame with original script format
            df_upset = pd.DataFrame(
                {
                    "Motifs": formatted_motifs,
                    "Observed": dcounts,
                    "Expected": exp_counts.astype(int),
                    "Expected SD": expected_sd,
                    "Effect Size": effect_sizes,
                    "P-value": p_values,
                    "Degree": degree,
                    "Group": group,
                }
            )

            return {
                "df_upset": df_upset,
                "motif_labels": motif_labels,
                "dcounts": dcounts,
                "exp_counts": exp_counts,
                "effect_sizes": effect_sizes,
                "p_values": p_values,
            }

        except Exception as e:
            self.logger.log_error(e, "Generating upset plot data")
            raise

    def prepare_upset_plot_data(
        self,
        df_obs_exp: pd.DataFrame,
        motif_probs: List[float],
        n0: float,
        motif_labels: List[List[str]],
    ) -> Dict[str, Any]:
        """
        Prepare data for upset plot visualization.

        Args:
            df_obs_exp: DataFrame with observed/expected motif data
            motif_probs: List of motif probabilities
            n0: Total observed cells

        Returns:
            Dict containing prepared upset plot data
        """
        try:
            from scipy.stats import binomtest

            # Prepare upset data - EXACT same as original script
            def prepare_upset_data(df):
                # mask1 = [i for (i,x) in enumerate(motif_labels) if len(x) > 1]
                mask1 = [i for (i, x) in enumerate(df["Degree"].to_list()) if x > 1]
                a = subset_list(df["Motifs"].to_list(), mask1)
                b = df["Observed"][mask1]
                c = df["Expected"][mask1]
                d = df["Expected SD"][mask1]
                e = df["Effect Size"][mask1]
                f = df["P-value"][mask1]
                g = df["Group"][mask1]
                mask2 = [i for i in range(b.shape[0]) if b.iloc[i] > 0]
                a = subset_list(a, mask2)
                b = b.iloc[mask2]
                b = b.to_numpy().astype(int)
                #
                c = c.iloc[mask2]
                c = c.to_numpy().astype(int)
                #
                d = d.iloc[mask2]
                #
                e = e.iloc[mask2]
                #
                f = f.iloc[mask2]
                #
                g = g.iloc[mask2]
                dfdata = pd.DataFrame(data=[a, b, c, d, e, f, g]).T
                dfdata.columns = [
                    "Motifs",
                    "Observed",
                    "Expected",
                    "Expected SD",
                    "Effect Size",
                    "P-value",
                    "Group",
                ]
                return dfdata

            def subset_list(lst, mask):
                return [lst[i] for i in mask]

            # Create the dfraw DataFrame - EXACT same as original
            # Use the passed motif_labels (lists of strings) instead of formatted motifs
            observed_counts = df_obs_exp["Observed"].tolist()
            expected_counts = df_obs_exp["Expected"].tolist()

            # Calculate expected SD, effect sizes, and p-values - EXACT same as original
            # Calculate expected SD
            expected_sd = []
            for i in range(len(motif_probs)):
                # Handle NaN values and negative numbers under square root
                prob = motif_probs[i] if not np.isnan(motif_probs[i]) else 0.0
                n0_val = n0 if not np.isnan(n0) else 0.0
                under_sqrt = prob * n0_val * (1 - prob)
                if under_sqrt < 0:
                    sd = 0.0
                else:
                    sd = np.sqrt(under_sqrt)
                expected_sd.append(sd)

            # Use EXACT same logic as original script - replicate get_motif_sig_pts
            def get_motif_sig_pts(
                dcounts,
                labels,
                prob_edge,
                n0,
                exclude_zeros=False,
                p_transform=lambda x: x,
            ):
                num_motifs = len(dcounts)
                expected, probs = self._get_expected_counts(labels, prob_edge, n0)
                assert dcounts.shape[0] == expected.shape[0]
                if exclude_zeros:
                    nonzid = np.nonzero(dcounts)[0]
                else:
                    nonzid = np.arange(dcounts.shape[0])
                num_nonzid_motifs = nonzid.shape[0]
                dcounts_ = dcounts[nonzid]
                expected_ = expected[nonzid]
                probs_ = probs[nonzid]
                # Effect size is log2(observed/expected)
                effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
                matches = np.zeros(num_nonzid_motifs)
                assert dcounts_.shape[0] == expected_.shape[0]
                dcounts_ = dcounts_.astype(int)
                for i in range(num_nonzid_motifs):
                    pi = max(probs_[i], 1e-10)  # avoid zero or very small probs
                    matches[i] = binomtest(int(dcounts_[i]), n=n0, p=pi).pvalue
                    matches[i] = max(matches[i], 1e-10)
                matches = p_transform(matches)
                # matches is the significance level
                res = zip(effect_size, matches)
                mlabels = [labels[h] for h in nonzid]
                return list(res), mlabels

            # Get effect sizes and p-values using EXACT same function as original
            dcounts_array = np.array(observed_counts)
            sigsraw, slabelsraw = get_motif_sig_pts(
                dcounts_array,
                motif_labels,
                0.2,
                n0,
                exclude_zeros=False,
                p_transform=lambda x: x,
            )
            effectsigsraw = np.array(sigsraw)

            # Calculate degree and group - EXACT same as original
            degree = [len(x) for x in motif_labels]
            degree[0] = 0

            group = []
            bonferroni_correction = len(motif_labels)  # EXACT same as original script
            for i in range(len(degree)):
                """
                Group 1: motifs significantly over represented
                Group 2: motifs significantly under-represented
                Group 3: motifs non-significantly over-represented
                Group 4: motifs non-significantly under-represented
                """
                grp = 0
                thr = 0.05 / bonferroni_correction
                if effectsigsraw[i, 0] > 0:  # over-represented
                    if effectsigsraw[i, 1] < thr:  # statistically significant
                        grp = 1  # significantly over-represented
                    else:
                        grp = 3  # non-significantly over-represented
                else:  # under-represented
                    if effectsigsraw[i, 1] < thr:  # statistically significant
                        grp = 2  # significantly under-represented
                    else:
                        grp = 4  # non-significantly under-represented
                group.append(grp)

            # Create DataFrame with same format as original script - EXACT same data
            dfraw = pd.DataFrame(
                data=[
                    motif_labels,
                    observed_counts,
                    expected_counts,
                    expected_sd,
                    effectsigsraw[:, 0],  # Effect sizes from get_motif_sig_pts
                    effectsigsraw[:, 1],  # P-values from get_motif_sig_pts
                    degree,
                    group,
                ]
            ).T
            dfraw.columns = [
                "Motifs",
                "Observed",
                "Expected",
                "Expected SD",
                "Effect Size",
                "P-value",
                "Degree",
                "Group",
            ]

            # Prepare upset data using the helper function
            dfdata = prepare_upset_data(dfraw)

            return {
                "dfraw": dfraw,
                "dfdata": dfdata,
                "motif_labels": motif_labels,
                "observed_counts": observed_counts,
                "expected_counts": expected_counts,
                "expected_sd": expected_sd,
                "effect_sizes": effectsigsraw[:, 0],
                "p_values": effectsigsraw[:, 1],
                "degree": degree,
                "group": group,
            }

        except Exception as e:
            self.logger.log_error(e, "Preparing upset plot data")
            raise

    def generate_motif_raw_data_files(
        self, matrix: np.ndarray, columns: List[str], sample_name: str
    ) -> Dict[str, Any]:
        """Generate motif raw data files using EXACT same logic as original script"""
        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix, columns=columns)

        # Generate motifs using the same logic as original script
        def gen_motifs(r, labels):
            num_motifs = 2**r
            motifs = np.zeros((num_motifs, r)).astype(bool)
            motif_ids = list(
                itertools.chain.from_iterable(
                    itertools.combinations(range(r), i) for i in range(r + 1)
                )
            )
            motif_labels = []
            for i in range(num_motifs):
                idx = motif_ids[i]
                motifs[i, idx] = True
                label = [labels[j] for j in idx] if idx else [""]
                motif_labels.append(label)
            return motifs, motif_labels

        def count_motifs(df, motifs, return_ids=False):
            cells, regions = df.shape
            data = df.to_numpy().astype(bool)
            counts = np.zeros(motifs.shape[0])
            cell_ids = []
            for i in range(motifs.shape[0]):
                cell_ids_ = []
                for j in range(data.shape[0]):
                    if np.array_equal(motifs[i], data[j]):
                        counts[i] = counts[i] + 1
                        cell_ids_.append(j)
                cell_ids.append(cell_ids_)
            if return_ids:
                return counts, cell_ids
            else:
                return counts

        # Generate motifs and count them
        motifs, motif_labels = gen_motifs(df.shape[1], df.columns)
        dcounts, cell_ids = count_motifs(df, motifs, return_ids=True)

        # Prepare data for each motif with non-zero counts
        motif_data_files = {}
        for i, (count, cell_id_list) in enumerate(zip(dcounts, cell_ids)):
            if (
                count > 0 and motif_labels[i] and motif_labels[i][0]
            ):  # Skip empty motifs
                # Create motif name
                motif_name = "_".join(motif_labels[i])
                fname = f"{sample_name}_{motif_name}_raw_data.csv"

                # Get the actual cells that have this motif - EXACT same as original
                if cell_id_list:
                    # Use EXACT same logic as original: get ALL columns for these cells
                    motif_cells = df.iloc[cell_id_list, :]
                    # Create index like original: cell_0, cell_1, etc.
                    index = [f"cell_{cid}" for cid in cell_id_list]
                    motif_cells.index = index

                    motif_data_files[fname] = motif_cells

        return motif_data_files

    def generate_per_cell_projection_strength_data(
        self, matrix: np.ndarray, columns: List[str]
    ) -> Dict[str, Any]:
        """Generate per-cell projection strength data using EXACT same logic as original script"""
        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix, columns=columns)

        # Generate motifs using the same logic as original script
        def gen_motifs(r, labels):
            num_motifs = 2**r
            motifs = np.zeros((num_motifs, r)).astype(bool)
            motif_ids = list(
                itertools.chain.from_iterable(
                    itertools.combinations(range(r), i) for i in range(r + 1)
                )
            )
            motif_labels = []
            for i in range(num_motifs):
                idx = motif_ids[i]
                motifs[i, idx] = True
                label = [labels[j] for j in idx] if idx else [""]
                motif_labels.append(label)
            return motifs, motif_labels

        def count_motifs(df, motifs, return_ids=False):
            cells, regions = df.shape
            data = df.to_numpy().astype(bool)
            counts = np.zeros(motifs.shape[0])
            cell_ids = []
            for i in range(motifs.shape[0]):
                cell_ids_ = []
                for j in range(data.shape[0]):
                    if np.array_equal(motifs[i], data[j]):
                        counts[i] = counts[i] + 1
                        cell_ids_.append(j)
                cell_ids.append(cell_ids_)
            if return_ids:
                return counts, cell_ids
            else:
                return counts

        # Generate motifs and counts
        motifs, motif_labels = gen_motifs(df.shape[1], df.columns)
        dcounts, cell_ids = count_motifs(df, motifs, return_ids=True)

        # Filter to only show motifs with two or more regions (hide_singlets=True)
        mask = [i for (i, l) in enumerate(motif_labels) if len(l) > 1]
        cell_ids = [cell_ids[i] for i in mask]
        dcounts = [dcounts[i] for i in mask]
        motif_labels = [motif_labels[i] for i in mask]

        # Filter to only motifs with cells
        non0cell_ids = [(i, x) for (i, x) in enumerate(cell_ids) if len(x) > 0]

        return {
            "df": df,
            "motif_labels": motif_labels,
            "dcounts": dcounts,
            "cell_ids": cell_ids,
            "non0cell_ids": non0cell_ids,
            "has_multi_region_motifs": len(non0cell_ids) > 0,
        }
