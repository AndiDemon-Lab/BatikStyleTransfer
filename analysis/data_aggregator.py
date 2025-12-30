import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class ResultsAggregator:
    """
    Aggregate and analyze results from batch NST experiments.
    """

    def __init__(self, results_dir):
        """
        Initialize aggregator with results directory.

        Args:
            results_dir: Path to directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = []
        self.df = None

    def load_all_experiments(self):
        """
        Load all experiment metadata from the results directory.
        """
        # Find all metadata.json files
        metadata_files = list(self.results_dir.rglob("metadata.json"))

        print(f"Found {len(metadata_files)} experiment metadata files")

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    # Skip if this is an aggregated results file
                    if "results" in data and isinstance(data["results"], list):
                        continue
                    self.experiments.append(data)
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")

        print(f"Loaded {len(self.experiments)} experiments")

    def create_dataframe(self):
        """
        Create pandas DataFrame from experiments.

        Returns:
            DataFrame with experiment data
        """
        rows = []

        for exp in self.experiments:
            # Extract relevant data
            row = {
                "model_name": exp.get("model_name"),
                "content_name": exp.get("content_name"),
                "style_name": exp.get("style_name"),
                "content_weight": exp.get("hyperparameters", {}).get("content_weight"),
                "style_weight": exp.get("hyperparameters", {}).get("style_weight"),
                "learning_rate": exp.get("hyperparameters", {}).get("learning_rate"),
                "max_epochs": exp.get("hyperparameters", {}).get("max_epochs"),
                "pooling": exp.get("hyperparameters", {}).get("pooling"),
                "training_time_seconds": exp.get("training_time_seconds"),
            }

            # Add final metrics
            final_metrics = exp.get("final_metrics", {})
            for metric_name, value in final_metrics.items():
                row[f"final_{metric_name}"] = value

            # Add loss tracking summary
            loss_tracking = exp.get("loss_tracking", {})
            summary = loss_tracking.get("summary", {})

            for metric_name, metric_stats in summary.items():
                if isinstance(metric_stats, dict):
                    for stat_name, stat_value in metric_stats.items():
                        row[f"{metric_name}_{stat_name}"] = stat_value

            rows.append(row)

        self.df = pd.DataFrame(rows)
        return self.df

    def compute_statistics_by_model(self):
        """
        Compute statistics grouped by model.

        Returns:
            Dictionary with statistics per model
        """
        if self.df is None:
            self.create_dataframe()

        stats_dict = {}

        # Metrics to analyze
        metrics = [
            "final_ssim",
            "final_psnr",
            "final_mse",
            "final_lpips",
            "training_time_seconds",
            "loss_final",
        ]

        for model in self.df["model_name"].unique():
            model_df = self.df[self.df["model_name"] == model]

            model_stats = {"n_experiments": len(model_df)}

            for metric in metrics:
                if metric in model_df.columns:
                    values = model_df[metric].dropna()
                    if len(values) > 0:
                        model_stats[metric] = {
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "median": float(values.median()),
                        }

            stats_dict[model] = model_stats

        return stats_dict

    def perform_anova(self, metric="final_ssim"):
        """
        Perform ANOVA test to compare models.

        Args:
            metric: Metric to compare

        Returns:
            Dictionary with ANOVA results
        """
        if self.df is None:
            self.create_dataframe()

        if metric not in self.df.columns:
            return None

        # Group data by model
        groups = []
        model_names = []

        for model in self.df["model_name"].unique():
            model_data = self.df[self.df["model_name"] == model][metric].dropna()
            if len(model_data) > 0:
                groups.append(model_data.values)
                model_names.append(model)

        if len(groups) < 2:
            return None

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            "metric": metric,
            "models": model_names,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def perform_pairwise_tests(self, metric="final_ssim"):
        """
        Perform pairwise t-tests between models.

        Args:
            metric: Metric to compare

        Returns:
            DataFrame with pairwise comparison results
        """
        if self.df is None:
            self.create_dataframe()

        if metric not in self.df.columns:
            return None

        models = self.df["model_name"].unique()
        results = []

        for i, model1 in enumerate(models):
            for model2 in models[i + 1 :]:
                data1 = self.df[self.df["model_name"] == model1][metric].dropna()
                data2 = self.df[self.df["model_name"] == model2][metric].dropna()

                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_value = stats.ttest_ind(data1, data2)

                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (
                            (len(data1) - 1) * data1.std() ** 2
                            + (len(data2) - 1) * data2.std() ** 2
                        )
                        / (len(data1) + len(data2) - 2)
                    )
                    cohens_d = (
                        (data1.mean() - data2.mean()) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    results.append(
                        {
                            "model1": model1,
                            "model2": model2,
                            "metric": metric,
                            "mean1": data1.mean(),
                            "mean2": data2.mean(),
                            "diff": data1.mean() - data2.mean(),
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "cohens_d": cohens_d,
                            "significant": p_value < 0.05,
                        }
                    )

        return pd.DataFrame(results)

    def compute_correlation_matrix(self):
        """
        Compute correlation matrix between metrics.

        Returns:
            Correlation matrix DataFrame
        """
        if self.df is None:
            self.create_dataframe()

        # Select numeric columns related to metrics
        metric_cols = [
            col
            for col in self.df.columns
            if any(
                x in col
                for x in ["final_", "loss_", "ssim", "psnr", "lpips", "mse", "time"]
            )
        ]

        metric_df = self.df[metric_cols].select_dtypes(include=[np.number])

        return metric_df.corr()

    def export_to_csv(self, output_path):
        """
        Export aggregated data to CSV.

        Args:
            output_path: Path to save CSV file
        """
        if self.df is None:
            self.create_dataframe()

        self.df.to_csv(output_path, index=False)
        print(f"Exported data to {output_path}")

    def export_to_excel(self, output_path):
        """
        Export comprehensive analysis to Excel with multiple sheets.

        Args:
            output_path: Path to save Excel file
        """
        if self.df is None:
            self.create_dataframe()

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Raw data
            self.df.to_excel(writer, sheet_name="Raw Data", index=False)

            # Statistics by model
            stats_dict = self.compute_statistics_by_model()
            stats_rows = []
            for model, stats in stats_dict.items():
                for metric, values in stats.items():
                    if isinstance(values, dict):
                        row = {"model": model, "metric": metric}
                        row.update(values)
                        stats_rows.append(row)

            stats_df = pd.DataFrame(stats_rows)
            stats_df.to_excel(writer, sheet_name="Statistics by Model", index=False)

            # Correlation matrix
            corr_matrix = self.compute_correlation_matrix()
            corr_matrix.to_excel(writer, sheet_name="Correlation Matrix")

            # ANOVA results for key metrics
            anova_results = []
            for metric in ["final_ssim", "final_psnr", "final_lpips", "final_mse"]:
                result = self.perform_anova(metric)
                if result:
                    anova_results.append(result)

            if anova_results:
                anova_df = pd.DataFrame(anova_results)
                anova_df.to_excel(writer, sheet_name="ANOVA Results", index=False)

            # Pairwise comparisons
            pairwise_results = []
            for metric in ["final_ssim", "final_psnr", "final_lpips"]:
                result = self.perform_pairwise_tests(metric)
                if result is not None and len(result) > 0:
                    pairwise_results.append(result)

            if pairwise_results:
                pairwise_df = pd.concat(pairwise_results, ignore_index=True)
                pairwise_df.to_excel(
                    writer, sheet_name="Pairwise Comparisons", index=False
                )

        print(f"Exported comprehensive analysis to {output_path}")

    def generate_latex_table(
        self,
        metrics=["final_ssim", "final_psnr", "final_lpips", "training_time_seconds"],
    ):
        """
        Generate LaTeX table for paper.

        Args:
            metrics: List of metrics to include

        Returns:
            LaTeX table string
        """
        if self.df is None:
            self.create_dataframe()

        stats_dict = self.compute_statistics_by_model()

        # Create table header
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += (
            "\\caption{Quantitative comparison of CNN architectures for batik NST}\n"
        )
        latex += "\\label{tab:quantitative_results}\n"
        latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\toprule\n"

        # Header row
        metric_names = {
            "final_ssim": "SSIM↑",
            "final_psnr": "PSNR (dB)↑",
            "final_lpips": "LPIPS↓",
            "final_mse": "MSE↓",
            "training_time_seconds": "Time (s)↓",
        }

        header = (
            "Model & "
            + " & ".join([metric_names.get(m, m) for m in metrics])
            + " \\\\\n"
        )
        latex += header
        latex += "\\midrule\n"

        # Data rows
        for model, stats in stats_dict.items():
            row = model
            for metric in metrics:
                if metric in stats and isinstance(stats[metric], dict):
                    mean = stats[metric]["mean"]
                    std = stats[metric]["std"]
                    row += f" & {mean:.3f}±{std:.3f}"
                else:
                    row += " & -"
            row += " \\\\\n"
            latex += row

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex


if __name__ == "__main__":
    # Example usage
    aggregator = ResultsAggregator("outputs/batch_experiments_test")
    aggregator.load_all_experiments()
    aggregator.create_dataframe()

    # Export to Excel
    aggregator.export_to_excel("outputs/analysis_results.xlsx")

    # Generate LaTeX table
    latex_table = aggregator.generate_latex_table()
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(latex_table)

    # Print statistics
    stats = aggregator.compute_statistics_by_model()
    print("\n" + "=" * 80)
    print("STATISTICS BY MODEL")
    print("=" * 80)
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        for metric, values in model_stats.items():
            if isinstance(values, dict):
                print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
