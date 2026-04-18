import argparse
from modules.datasets import glorys12_dataset
from modules.datasets.load_argo import ArgoProcessorRaw, ArgoProcessorCSV
from modules.datasets.load_glorys12 import GLORYS12Processor
from modules.datasets.glorys12_dataset import test_glorys12_dataset, GLORYS12Dataset
from modules.networks.fibonacci import FibonacciSphere


parser = argparse.ArgumentParser(description='Process GLORYS12 ocean data')
parser.add_argument('--start-year', type=int, default=2012, 
                       help='Start year for data processing (inclusive)')
parser.add_argument('--end-year', type=int, default=2025,
                    help='End year for data processing (exclusive)')
parser.add_argument('--username', type=str, required=True,
                    help='Copernicus Marine username')
parser.add_argument('--password', type=str, required=True,
                    help='Copernicus Marine password')

args = parser.parse_args()

glorys12_processor = GLORYS12Processor(username=args.username, password=args.password)
# glorys12_processor.gen_all_metadata()
# glorys12_processor.dowload_and_process(year_range=range(args.start_year, args.end_year))
# glorys12_processor.dowload_and_process_first40_daily(year_range=range(args.start_year, args.end_year))
# glorys12_processor.plot_random_samples()
# glorys12_processor.visualize_all_depths(
#     year=2018,
#     month=6,
#     day=7,
#     output_dir='./output/plot/data/glorys12/all_depths',
#     data_type='train',
#     clean_mode=False,
#     depth=None,
#     add_noise=False,
#     noise_level=0.4
# )
# glorys12_processor.compute_test_statistics()

# glorys12_processor.compute_spatial_std_statistics(
#     data_type='test',
#     output_file='./data/glorys12/test_spatial_std.csv'
# )

# argo_processor_raw = ArgoProcessorRaw()
# argo_processor_raw.sample_and_inspect_files(n_per_prefix=25)

argo_processor_csv = ArgoProcessorCSV()
# argo_processor_csv.check_csv_folder_year_consistency()
# argo_processor_csv.convert_csv_to_profile_nc(year_folders=['csv_2021_2022', 'csv_2023_2024'])
# argo_processor_csv.visualize_random_nc_profiles(
#     n_samples=100,
#     output_dir='./output/plot/data/argo/random_profiles',
#     verbose=True
# )
# argo_processor_csv.analyze_daily_profile_counts()
# argo_processor_csv.analyze_netcdf_daily_profile_counts(figsize=(9, 4.5), dpi=900)
argo_processor_csv.analyze_netcdf_spatial_distribution(
    nc_file='./data/argo/argo_profiles.nc',
    output_path='./output/plot/data/argo/argo_spatial_distribution.png',
    year_range=(2012, 2023),
    figsize=(9, 5.5),
    dpi=900
)
# argo_processor_csv.count_profiles_per_year(year_folders=['csv_2023_2024_2'])
# argo_processor_csv.plot_profile_length_distribution(save_path='./output/plot/data/argo/profile_length_distribution.png')
# argo_processor_csv.process_argo_to_netcdf(output_file='./data/argo/argo_profiles.nc')
# argo_processor_csv.split_netcdf_by_date(
#     nc_file='./data/argo/argo_profiles.nc',
#     output_dir='./data/argo/argo_profiles_by_date',
#     filename_prefix='argo_profiles'
# )

dataset = GLORYS12Dataset(
    data_dir='./data/glorys12',
    argo_file='./data/argo/argo_profiles_by_date',
    argo_days=5
)
# dataset.compute_statistics(num_samples=10000)
# dataset.visualize_distribution(output_file='./output/plot/data/distribution.png', num_samples=1000)
# dataset.visualize_random_samples(
#     num_samples=5, 
#     output_dir='./output/plot/data/random_select_samples'
# )
# dataset.visualize_argo_spatial_distribution(
#     num_samples=10,
#     output_dir='./output/plot/data/random_select_samples_argo_spatial',
#     depth_index=20,
#     show_background=True,
#     figsize=(9, 5.5),
#     dpi=900
# )
# dataset.visualize_argo_spatial_distribution(
#     num_samples=10,
#     output_dir='./output/plot/data/random_select_samples_argo_spatial_no_background',
#     depth_index=20,
#     show_background=False
# )

# test_glorys12_dataset()

# fib_sphere = FibonacciSphere(num_points=1024)
# points = fib_sphere.generate_points()
# fib_sphere.plot(save_path="./output/plot/data/fibonacci/fibonacci_sphere_1024_3d.jpg", show=False)
# fib_sphere.plot_on_map(save_path="./output/plot/data/fibonacci/fibonacci_sphere_1024_map.jpg", show=False)
# fib_sphere.plot_grid_distribution(save_path="./output/plot/data/fibonacci/fibonacci_sphere_1024_dist.jpg", show=False)
