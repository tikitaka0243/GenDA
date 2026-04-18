"""
Plot calling utilities for RMSE comparison across experiments.

Adjust paths and experiment names according to your setup.
"""


def rmse_comparison_plot(processor):
    """Generate RMSE comparison plots across different experiment configurations.

    Example usage:
        processor.plot_rmse_boxplot_comparison(
            rmse_files={
                "Experiment A": "./output/results/experiment_a/rmse_results_ensemble_10.csv",
                "Experiment B": "./output/results/experiment_b/rmse_results_ensemble_10.csv",
            },
            output_path="./output/plot/rmse_comparison/comparison.png",
            title="RMSE Comparison"
        )
    """
    pass
