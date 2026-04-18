"""
Post-processing script for ocean data assimilation results.

This script demonstrates how to use the post-processing and conformal
calibration utilities. Adjust paths and parameters according to your
experiment setup.
"""

from modules.postprocess import SamplePostProcessor
from modules.plot.plot_call import rmse_comparison_plot
from modules.conformal import ConformalPredictor
from modules.plot.plot_conformal import plot_combined_rank_histograms, plot_coverage_results_comparison


# ======================== Sample Post-Processing ========================

sample_dir = './output/results/YOUR_EXPERIMENT_DIR/samples'

processor = SamplePostProcessor(sample_dir=sample_dir)

# Randomly sample and visualize
# processor.random_sample_and_visualize(
#     n_samples=5,
#     seed=42,
#     export_netcdf=False
# )

# processor.random_sample_and_visualize_with_ground_truth(
#     n_samples=50,
#     seed=42
# )

# Visualize ensemble with ground truth for different regions
# _ = processor.visualize_ensemble_with_ground_truth(
#     ground_truth_dir='./data/glorys12/test',
#     n_samples=31,
#     seed=42,
#     region_extent=[
#         (-90, -40, 20, 70),    # North Atlantic region (non-polar)
#         (10, 60, -60, -10),    # Another region (non-polar)
#         (-180, 180, 60, 90),   # Arctic region (polar projection)
#         (-180, 180, -90, -60), # Antarctic region (polar projection)
#     ],
#     plot_mode='all',
#     save_format='jpg'
# )

# Visualize vertical profiles
# processor.visualize_vertical_profiles(
#     date_str='20231120',
#     ground_truth_dir='./data/glorys12/test',
#     save_format='jpg',
#     figsize=(14, 8),
#     dpi=900
# )

# Calculate RMSE
# processor.calculate_all_samples_rmse()
processor.calculate_ensemble_samples_rmse(
    ground_truth_dir='./data/glorys12/test',
    output_csv_name="rmse_results_ensemble_10.csv",
    num_ensemble_members=10
)
# processor.plot_srmse_boxplot_by_depth(figsize=(16, 16))
# processor.compute_spatial_ensemble_rmse(num_workers=8, sample_ratio=0.1)
# processor.plot_spatial_rmse_and_srmse(figsize=(9, 4))

# RMSE comparison across experiments
# rmse_comparison_plot(processor)


# ======================== Conformal Prediction ========================

conformal = ConformalPredictor(
    sample_dir='./output/results/YOUR_EXPERIMENT_DIR/samples',
    ground_truth_dir='./data/glorys12/test',
    n_bins=200,
    num_workers=64,
    sample_ratio=0.1,
    sample_seed=42
)

# conformal.plot_rank_histogram(save_name='rank_histogram.png')

# conformal.fit_isotonic_calibration(
#     use_binned=False,
#     method={
#         'thetao': 'xgboost',
#         'so': 'xgboost',
#         'uo': 'sklearn',
#         'vo': 'sklearn',
#     },
#     verbose=True,
# )

# model_path = './output/results/YOUR_EXPERIMENT_DIR/conformal/isotonic_calibration.pkl'

# conformal.plot_calibrated_rank_histogram(model_path=model_path, num_workers=4)

# conformal.visualize_pixel_distributions(
#     num_pixels=5,
#     save_name='pixel_distributions.pdf',
#     model_path=model_path,
#     date_str='20231120',
#     depth_level=19,
#     pixel_coords=[[128, 204], [116, 386], [99, 183], [78, 703], [284, 639]]
# )

# for confidence_level in [0.8, 0.9, 0.95, 0.99]:
#     conformal.calculate_coverage(
#         confidence_level=confidence_level,
#         model_path=model_path,
#         save_name=f'coverage_results_{confidence_level}.txt'
#     )

# plot_combined_rank_histograms(
#     cal_output_dir='./output/results/YOUR_CALIBRATION_DIR',
#     test_output_dir='./output/results/YOUR_TEST_DIR',
#     n_bins=200
# )

# plot_coverage_results_comparison(
#     files_by_group=files_by_group,
#     save_path='./output/results/conformal_coverage_comparison.pdf',
#     group_display_names={'test': 'Test', 'cal': 'Calibration'},
# )
