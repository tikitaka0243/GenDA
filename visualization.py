"""
Visualization script for ocean data assimilation results.

This script demonstrates how to use the post-processing visualization
utilities, including ensemble visualization and vertical profile plots.
Adjust paths and parameters according to your experiment setup.
"""

import argparse

from modules.postprocess import SamplePostProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ocean data assimilation sampling results.'
    )
    parser.add_argument(
        '--sample_dir', type=str, required=True,
        help='Path to the sampling output directory (e.g., ./output/results/genda/sample_YYYYMMDD_HHMMSS/samples)'
    )
    parser.add_argument(
        '--ground_truth_dir', type=str, default='./data/glorys12/test',
        help='Path to the ground truth NetCDF files (default: ./data/glorys12/test)'
    )
    parser.add_argument(
        '--n_samples', type=int, default=10,
        help='Number of ensemble samples to visualize (default: 10)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sample selection (default: 42)'
    )
    parser.add_argument(
        '--plot_mode', type=str, default='all',
        help="Plot mode for ensemble visualization: 'all', 'ensemble_mean', 'ensemble_std', 'ground_truth', 'error' (default: 'all')"
    )
    parser.add_argument(
        '--save_format', type=str, default='jpg',
        help='Image format for saved figures: jpg, png, pdf, etc. (default: jpg)'
    )
    parser.add_argument(
        '--date_str', type=str, default='20231120',
        help='Date string (YYYYMMDD) for vertical profile visualization (default: 20231120)'
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='DPI for saved figures (default: 300)'
    )
    args = parser.parse_args()

    processor = SamplePostProcessor(sample_dir=args.sample_dir)

    # Visualize ensemble with ground truth for different regions
    _ = processor.visualize_ensemble_with_ground_truth(
        ground_truth_dir=args.ground_truth_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        region_extent=[
            (-90, -40, 20, 70),    # North Atlantic region (non-polar)
            (10, 60, -60, -10),    # Another region (non-polar)
            (-180, 180, 60, 90),   # Arctic region (polar projection)
            (-180, 180, -90, -60), # Antarctic region (polar projection)
        ],
        plot_mode=args.plot_mode,
        save_format=args.save_format
    )

    # Visualize vertical profiles
    processor.visualize_vertical_profiles(
        date_str=args.date_str,
        ground_truth_dir=args.ground_truth_dir,
        save_format=args.save_format,
        figsize=(14, 8),
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()