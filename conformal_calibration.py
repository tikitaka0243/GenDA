"""
Conformal Calibration script for ocean data assimilation results.

This script performs conformal prediction, isotonic calibration,
rank histogram plotting, pixel distribution visualization, and
coverage calculation. Adjust paths and parameters according to your
experiment setup.
"""

from modules.conformal import ConformalPredictor
from modules.plot.plot_conformal import plot_combined_rank_histograms, plot_coverage_results_comparison


# ======================== Conformal Prediction ========================

conformal = ConformalPredictor(
    sample_dir='./output/results/YOUR_EXPERIMENT_DIR/samples',
    ground_truth_dir='./data/glorys12/test',
    n_bins=200,
    num_workers=64,
    sample_ratio=0.1,
    sample_seed=42
)

conformal.plot_rank_histogram(save_name='rank_histogram.png')

conformal.fit_isotonic_calibration(
    use_binned=False,
    method={
        'thetao': 'xgboost',
        'so': 'xgboost',
        'uo': 'sklearn',
        'vo': 'sklearn',
    },
    verbose=True,
)

model_path = './output/results/YOUR_EXPERIMENT_DIR/conformal/isotonic_calibration.pkl'

conformal.plot_calibrated_rank_histogram(model_path=model_path, num_workers=4)

conformal.visualize_pixel_distributions(
    num_pixels=5,
    save_name='pixel_distributions.pdf',
    model_path=model_path,
    date_str='20231120',
    depth_level=19,
    pixel_coords=[[128, 204], [116, 386], [99, 183], [78, 703], [284, 639]]
)
