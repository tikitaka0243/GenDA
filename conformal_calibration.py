"""
Conformal Calibration script for ocean data assimilation results.

Workflow:
1. Plot pre-calibration rank histograms for both calibration and test sets.
2. Fit isotonic regression on the calibration set only.
3. Plot post-calibration rank histograms for both calibration and test sets.
4. Visualize pixel distributions (before and after calibration) for both sets.

Example usage:
    python conformal_calibration.py \\
        --cal_sample_dir ./output/results/genda/sample_cal_20260101_120000/samples \\
        --cal_gt_dir ./data/glorys12/cal \\
        --test_sample_dir ./output/results/genda/sample_20260101_120000/samples \\
        --test_gt_dir ./data/glorys12/test
"""

import argparse
import json
import os

from modules.conformal import ConformalPredictor
from modules.plot.plot_conformal import plot_combined_rank_histograms, plot_coverage_results_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conformal Calibration for ocean data assimilation results."
    )

    # Calibration set arguments
    parser.add_argument(
        "--cal_sample_dir", type=str, required=True,
        help="Path to the calibration set samples directory."
    )
    parser.add_argument(
        "--cal_gt_dir", type=str, default="./data/glorys12/cal",
        help="Path to the calibration set ground truth directory (default: ./data/glorys12/cal)."
    )

    # Test set arguments
    parser.add_argument(
        "--test_sample_dir", type=str, required=True,
        help="Path to the test set samples directory."
    )
    parser.add_argument(
        "--test_gt_dir", type=str, default="./data/glorys12/test",
        help="Path to the test set ground truth directory (default: ./data/glorys12/test)."
    )

    # ConformalPredictor common arguments
    parser.add_argument(
        "--n_bins", type=int, default=200,
        help="Number of bins for rank histogram (default: 200)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=64,
        help="Number of parallel workers for data loading (default: 64)."
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.1,
        help="Ratio of data to use (default: 0.1)."
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42,
        help="Random seed for sampling (default: 42)."
    )

    # Isotonic calibration arguments
    parser.add_argument(
        "--save_name", type=str, default="isotonic_calibration.pkl",
        help="Filename for the saved calibration model (default: isotonic_calibration.pkl)."
    )
    parser.add_argument(
        "--use_binned", action="store_true",
        help="Use binned data for calibration (default: False, uses raw PIT values)."
    )
    parser.add_argument(
        "--cal_method", type=str,
        default='{"thetao": "xgboost", "so": "xgboost", "uo": "sklearn", "vo": "sklearn"}',
        help="Calibration method per variable as a JSON dict or single method string."
    )
    parser.add_argument(
        "--cal_verbose", type=lambda x: x.lower() in ("true", "1", "yes"), default=True,
        help="Print detailed calibration progress (default: True)."
    )
    parser.add_argument(
        "--cal_batch_size", type=int, default=4096,
        help="Batch size for MLP training (default: 4096)."
    )
    parser.add_argument(
        "--cal_epochs", type=int, default=1000,
        help="Number of training epochs for MLP (default: 1000)."
    )

    # Calibrated rank histogram arguments
    parser.add_argument(
        "--cal_hist_save_name", type=str, default="calibrated_rank_histogram.png",
        help="Filename for the calibrated rank histogram plot (default: calibrated_rank_histogram.png)."
    )
    parser.add_argument(
        "--cal_hist_num_workers", type=int, default=4,
        help="Number of workers for calibrated histogram computation (default: 4)."
    )

    # Pixel distribution visualization arguments
    parser.add_argument(
        "--pixel_save_name", type=str, default="pixel_distributions.pdf",
        help="Base filename for the pixel distribution plot."
    )
    parser.add_argument(
        "--num_pixels", type=int, default=5,
        help="Number of random pixels to visualize (default: 5)."
    )
    parser.add_argument(
        "--pixel_date_str", type=str, default=None,
        help="Date string for pixel visualization in 'YYYYMMDD' format (default: None, random)."
    )
    parser.add_argument(
        "--pixel_depth_level", type=int, default=None,
        help="Depth level for pixel visualization (default: None, random)."
    )
    parser.add_argument(
        "--pixel_coords", type=str, default=None,
        help="List of (h, w) pixel coordinates as a JSON list, e.g., '[[128,204],[116,386]]'."
    )
    parser.add_argument(
        "--pixel_seed", type=int, default=42,
        help="Random seed for pixel selection (default: 42)."
    )

    return parser.parse_args()


def run_calibration_pipeline(args):
    """Run the full conformal calibration pipeline."""

    # Parse calibration method (JSON dict or single string)
    try:
        cal_method = json.loads(args.cal_method)
    except json.JSONDecodeError:
        cal_method = args.cal_method  # single string like 'sklearn'

    # Parse pixel coordinates (JSON list or None)
    pixel_coords = None
    if args.pixel_coords:
        pixel_coords = json.loads(args.pixel_coords)

    # ========================================================================
    # Step 1: Plot pre-calibration rank histograms
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Plotting pre-calibration rank histograms")
    print("=" * 60)

    # Calibration set
    print(f"\n--- Calibration set ---")
    cal_conformal = ConformalPredictor(
        sample_dir=args.cal_sample_dir,
        ground_truth_dir=args.cal_gt_dir,
        n_bins=args.n_bins,
        num_workers=args.num_workers,
        sample_ratio=args.sample_ratio,
        sample_seed=args.sample_seed
    )
    cal_conformal.plot_rank_histogram(save_name="rank_histogram_cal.png")

    # Test set (if provided)
    test_conformal = None
    if args.test_sample_dir:
        print(f"\n--- Test set ---")
        test_conformal = ConformalPredictor(
            sample_dir=args.test_sample_dir,
            ground_truth_dir=args.test_gt_dir,
            n_bins=args.n_bins,
            num_workers=args.num_workers,
            sample_ratio=args.sample_ratio,
            sample_seed=args.sample_seed
        )
        test_conformal.plot_rank_histogram(save_name="rank_histogram_test.png")

    # ========================================================================
    # Step 2: Fit isotonic regression on calibration set
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Fitting isotonic calibration on calibration set")
    print("=" * 60)

    model_dir = os.path.join(os.path.dirname(args.cal_sample_dir), "conformal")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.save_name)

    cal_conformal.fit_isotonic_calibration(
        save_name=args.save_name,
        use_binned=args.use_binned,
        method=cal_method,
        verbose=args.cal_verbose,
        batch_size=args.cal_batch_size,
        epochs=args.cal_epochs,
    )
    print(f"\nCalibration model saved to: {model_path}")

    # ========================================================================
    # Step 3: Plot post-calibration rank histograms
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Plotting post-calibration rank histograms")
    print("=" * 60)

    # Calibration set
    print(f"\n--- Calibration set (calibrated) ---")
    cal_conformal.plot_calibrated_rank_histogram(
        save_name="calibrated_rank_histogram_cal.png",
        model_path=model_path,
        num_workers=args.cal_hist_num_workers,
    )

    # Test set (if provided)
    if test_conformal is not None:
        print(f"\n--- Test set (calibrated) ---")
        test_conformal.plot_calibrated_rank_histogram(
            save_name="calibrated_rank_histogram_test.png",
            model_path=model_path,
            num_workers=args.cal_hist_num_workers,
        )

    # ========================================================================
    # Step 4: Visualize pixel distributions (before and after calibration)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Visualizing pixel distributions")
    print("=" * 60)

    # Calibration set
    print(f"\n--- Calibration set pixel distributions ---")
    cal_conformal.visualize_pixel_distributions(
        num_pixels=args.num_pixels,
        save_name=f"{args.pixel_save_name.replace('.pdf', '_cal.pdf')}",
        model_path=model_path,
        date_str=args.pixel_date_str,
        depth_level=args.pixel_depth_level,
        pixel_coords=pixel_coords,
        seed=args.pixel_seed,
    )

    # Test set (if provided)
    if test_conformal is not None:
        print(f"\n--- Test set pixel distributions ---")
        test_conformal.visualize_pixel_distributions(
            num_pixels=args.num_pixels,
            save_name=f"{args.pixel_save_name.replace('.pdf', '_test.pdf')}",
            model_path=model_path,
            date_str=args.pixel_date_str,
            depth_level=args.pixel_depth_level,
            pixel_coords=pixel_coords,
            seed=args.pixel_seed,
        )

    print("\n" + "=" * 60)
    print("Conformal calibration pipeline complete!")
    print("=" * 60)
    print(f"\nCalibration model: {model_path}")


def main():
    args = parse_args()
    run_calibration_pipeline(args)


if __name__ == "__main__":
    main()
