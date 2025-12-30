import json
from pathlib import Path
from run_paired_batch import PairedBatchRunner
from analysis.data_aggregator import ResultsAggregator


class ExperimentRunner:
    """
    Unified runner for all NST experiments.
    """

    def __init__(self, config_path="config_final.json"):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def run_full_batch(self):
        """
        Run full batch experiments with all models and pairs.

        Returns:
            List of experiment results
        """
        print("=" * 80)
        print("RUNNING FULL BATCH EXPERIMENTS")
        print("=" * 80)

        runner = PairedBatchRunner(self.config)
        results = runner.run_paired_batch()

        print(f"\n✓ Completed {len(results)} experiments")
        return results

    def run_ablation_study(self, ablation_config_path):
        """
        Run ablation study with specified configuration.

        Args:
            ablation_config_path: Path to ablation study config

        Returns:
            List of ablation results
        """
        print("=" * 80)
        print("RUNNING ABLATION STUDY")
        print("=" * 80)

        with open(ablation_config_path, "r") as f:
            ablation_config = json.load(f)

        runner = PairedBatchRunner(ablation_config)
        results = runner.run_paired_batch()

        print(f"\n✓ Completed ablation study: {len(results)} experiments")
        return results

    def analyze_results(self, results_dir="outputs/full_batch_experiments"):
        """
        Analyze experiment results and generate statistics.

        Args:
            results_dir: Directory containing experiment results

        Returns:
            Dictionary with analysis results
        """
        print("=" * 80)
        print("ANALYZING RESULTS")
        print("=" * 80)

        aggregator = ResultsAggregator(results_dir)
        aggregator.load_all_experiments()
        aggregator.create_dataframe()

        print(f"\nLoaded {len(aggregator.experiments)} experiments")

        # Export results
        aggregator.export_to_excel("outputs/full_analysis_results.xlsx")
        print("Exported to outputs/full_analysis_results.xlsx")

        # Compute statistics
        stats = aggregator.compute_statistics_by_model()

        print("\n" + "=" * 80)
        print("STATISTICS BY MODEL")
        print("=" * 80)

        for model, s in stats.items():
            print(f"\n{model.upper()}:")
            print(
                f"  SSIM: {s['final_ssim']['mean']:.4f} ± {s['final_ssim']['std']:.4f}"
            )
            print(
                f"  PSNR: {s['final_psnr']['mean']:.2f} ± {s['final_psnr']['std']:.2f} dB"
            )
            print(
                f"  LPIPS: {s['final_lpips']['mean']:.4f} ± {s['final_lpips']['std']:.4f}"
            )
            print(
                f"  Training Time: {s['training_time_seconds']['mean']:.1f}s ± {s['training_time_seconds']['std']:.1f}s"
            )

        # Generate LaTeX table
        latex = aggregator.generate_latex_table()
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        print(latex)

        return {"statistics": stats, "latex_table": latex, "aggregator": aggregator}

    def run_statistical_tests(self, results_dir="outputs/full_batch_experiments"):
        """
        Run statistical tests (ANOVA, pairwise comparisons).

        Args:
            results_dir: Directory containing experiment results

        Returns:
            Dictionary with statistical test results
        """
        print("=" * 80)
        print("RUNNING STATISTICAL TESTS")
        print("=" * 80)

        aggregator = ResultsAggregator(results_dir)
        aggregator.load_all_experiments()
        aggregator.create_dataframe()

        # ANOVA for SSIM
        anova_ssim = aggregator.perform_anova("final_ssim")
        print(
            f"\nANOVA (SSIM): F={anova_ssim['f_statistic']:.4f}, p={anova_ssim['p_value']:.4e}"
        )

        # Pairwise tests
        pairwise = aggregator.perform_pairwise_tests("final_ssim")
        print("\nPairwise comparisons (SSIM):")
        print(pairwise)

        return {"anova": anova_ssim, "pairwise": pairwise}


def main():
    """Main entry point for experiment runner."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python experiment_runner.py batch              # Run full batch")
        print("  python experiment_runner.py analyze            # Analyze results")
        print("  python experiment_runner.py stats              # Statistical tests")
        print("  python experiment_runner.py ablation <config>  # Ablation study")
        return

    command = sys.argv[1]
    runner = ExperimentRunner()

    if command == "batch":
        runner.run_full_batch()
    elif command == "analyze":
        runner.analyze_results()
    elif command == "stats":
        runner.run_statistical_tests()
    elif command == "ablation" and len(sys.argv) > 2:
        runner.run_ablation_study(sys.argv[2])
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
