import math
import sys
import os
import shutil
import concurrent.futures

import torch
import numpy as np

import modules.util.misc as misc
import modules.util.lr_sched as lr_sched
from tqdm import tqdm
import copy
from modules.datasets.glorys12_dataset import GLORYS12Dataset
import xarray as xr


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Unpack batch: (glorys_data, argo_data, depth_index, mask, time_encoding)
        time_encoding = None
        if len(batch) == 5:
            x, argo_data, depth_index, mask, time_encoding = batch
        elif len(batch) == 4:
            x, argo_data, depth_index, mask = batch
        else:
            x = batch
            argo_data = None
            depth_index = None
            mask = None

        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # data already normalized to [-1, 1] by GLORYS12Dataset
        x = x.to(device, non_blocking=True).to(torch.float32)
        
        if depth_index is not None:
            depth_index = depth_index.to(device, non_blocking=True)
            
        if argo_data is not None:
            for k, v in argo_data.items():
                if isinstance(v, torch.Tensor):
                    argo_data[k] = v.to(device, non_blocking=True)
        
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        
        if time_encoding is not None:
            time_encoding = time_encoding.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, mse_loss, grad_loss = model(x, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)

        loss_value = loss.item()
        mse_loss_value = mse_loss.item()
        grad_loss_value = grad_loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mse_loss=mse_loss_value)
        metric_logger.update(grad_loss=grad_loss_value)
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        mse_loss_value_reduce = misc.all_reduce_mean(mse_loss_value)
        grad_loss_value_reduce = misc.all_reduce_mean(grad_loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('total_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_loss', mse_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('grad_loss', grad_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def save_batch_data(sampled_images_cpu, start_idx, world_size, local_rank, dataset, save_folder, repeat_idx=0, repeat_count=1, mask_cpu=None):
    """
    Helper function to save a batch of images to disk.
    This runs in a separate thread to avoid blocking the GPU.
    """
    # Helper to handle Subset wrappers (added for validation sampling)
    def resolve_dataset(ds, idx):
        if isinstance(ds, torch.utils.data.Subset):
            return resolve_dataset(ds.dataset, ds.indices[idx])
        return ds, idx

    # Get underlying dataset for methods like _denormalize_data
    underlying_dataset, _ = resolve_dataset(dataset, 0)

    # Denormalize ocean data from [-1, 1] to original physical range
    denormalized_images = underlying_dataset._denormalize_data(sampled_images_cpu)
    
    # Apply mask if provided (set land to NaN)
    if mask_cpu is not None:
        # mask_cpu: (B, 1, H, W) where 1 is ocean, 0 is land
        # We want to set land to NaN
        mask_bool = mask_cpu.to(dtype=torch.bool)
        denormalized_images = denormalized_images.masked_fill(~mask_bool, float('nan'))

    denormalized_images = denormalized_images.numpy().astype(np.float32)

    # distributed save images as .npy files (ocean data)
    for b_id in range(denormalized_images.shape[0]):
        # Calculate dataset index corresponding to this sample
        dataset_idx = (start_idx + b_id) * world_size + local_rank
        
        # Handle padding in DistributedSampler
        if dataset_idx >= len(dataset):
            dataset_idx = dataset_idx % len(dataset)
        
        # Get real dataset and index to access sample info
        real_dataset, real_idx = resolve_dataset(dataset, dataset_idx)

        # Get sample info for filename
        info = real_dataset.get_sample_info(real_idx)
        timestamp = info['timestamp']
        depth_idx = info['depth_index']
        
        if timestamp is not None and depth_idx is not None:
            date_str = timestamp.strftime('%Y%m%d')
            if repeat_count > 1:
                filename = '{}_d{:02d}_r{:02d}.nc'.format(date_str, depth_idx, repeat_idx)
            else:
                filename = '{}_d{:02d}.nc'.format(date_str, depth_idx)
        else:
            # Fallback to unique ID based on rank and local count
            img_id = local_rank * 100000 + start_idx + b_id
            if repeat_count > 1:
                filename = '{}_r{:02d}.nc'.format(str(img_id).zfill(6), repeat_idx)
            else:
                filename = '{}.nc'.format(str(img_id).zfill(6))
        
        # Save as NetCDF with compression and shuffle filter
        # Create xarray Dataset
        data_var = denormalized_images[b_id] # Shape (C, H, W)
        ds = xr.Dataset(
            data_vars={
                'data': (('variable', 'height', 'width'), data_var)
            },
            coords={
                'variable': np.arange(data_var.shape[0]),
                'height': np.arange(data_var.shape[1]),
                'width': np.arange(data_var.shape[2])
            }
        )

        # Define compression settings
        compression = {
            'zlib': True,           # Enable DEFLATE compression
            'complevel': 6,         # Compression level
            'shuffle': True,        # Enable byte-shuffle filter
            'dtype': 'float32'
        }
        
        encoding = {'data': compression}
        
        # Save to NetCDF
        output_path = os.path.join(save_folder, filename)
        ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')


def _get_existing_sample_keys(save_folder):
    """Get set of existing sample keys (date_depth) from save folder"""
    existing_keys = set()
    if not os.path.exists(save_folder):
        return existing_keys
    
    for filename in os.listdir(save_folder):
        if filename.endswith('.nc'):
            # Store full filename without extension
            key = os.path.splitext(filename)[0]
            existing_keys.add(key)
    return existing_keys


def _check_batch_already_sampled(dataset, start_idx, world_size, local_rank, 
                                  current_batch_size, num_samples_per_input, existing_keys):
    """Check if all samples in a batch have already been generated"""
    # Helper to handle Subset wrappers
    def resolve_dataset(ds, idx):
        if isinstance(ds, torch.utils.data.Subset):
            return resolve_dataset(ds.dataset, ds.indices[idx])
        return ds, idx
    
    underlying_dataset, _ = resolve_dataset(dataset, 0)
    
    all_sampled = True
    for b_id in range(current_batch_size):
        dataset_idx = (start_idx + b_id) * world_size + local_rank
        
        # Handle padding in DistributedSampler
        if dataset_idx >= len(dataset):
            dataset_idx = dataset_idx % len(dataset)
        
        real_dataset, real_idx = resolve_dataset(dataset, dataset_idx)
        info = real_dataset.get_sample_info(real_idx)
        timestamp = info['timestamp']
        depth_idx = info['depth_index']
        
        if timestamp is not None and depth_idx is not None:
            date_str = timestamp.strftime('%Y%m%d')
            base_key = '{}_d{:02d}'.format(date_str, depth_idx)
            
            # Check if all repeats exist
            for repeat_idx in range(num_samples_per_input):
                sample_key = base_key
                if num_samples_per_input > 1:
                    sample_key = '{}_r{:02d}'.format(base_key, repeat_idx)
                
                if sample_key not in existing_keys:
                    all_sampled = False
                    break
        else:
            all_sampled = False
            
        if not all_sampled:
            break
    
    return all_sampled


def evaluate(model_without_ddp, args, data_loader=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    
    if data_loader is None:
        raise ValueError("data_loader must be provided")
        
    dataset = data_loader.dataset

    save_folder = os.path.join(args.output_dir, "samples")
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Get existing samples for resume sampling
    existing_keys = _get_existing_sample_keys(save_folder)
    if len(existing_keys) > 0:
        print(f"Found {len(existing_keys)} existing samples in output directory")
    
    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # Create a ThreadPoolExecutor for asynchronous saving
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    futures = []

    # Iterate over dataloader
    generated_count = 0
    total_images_target = args.num_images
    skipped_batches = 0
    
    num_samples_per_input = getattr(args, 'num_samples_per_input', 1)
    if num_samples_per_input < 1:
        num_samples_per_input = 1

    for i, batch in enumerate(tqdm(data_loader, desc="Generating")):
        if not getattr(args, 'conditional', False) and generated_count * world_size >= total_images_target:
            break
            
        # Unpack batch: (glorys_data, argo_data, depth_index, mask, time_encoding)
        mask = None
        time_encoding = None
        if len(batch) == 5:
            gt_images, argo_data, depth_index, mask, time_encoding = batch
        elif len(batch) == 4:
            gt_images, argo_data, depth_index, mask = batch
        else:
            gt_images = batch
            argo_data = None
            depth_index = None
            
        current_batch_size = gt_images.shape[0]
        
        # Check if this batch has already been sampled (resume sampling mode)
        if len(existing_keys) > 0:
            is_sampled = _check_batch_already_sampled(
                dataset, generated_count, world_size, local_rank,
                current_batch_size, num_samples_per_input, existing_keys
            )
            if is_sampled:
                skipped_batches += 1
                generated_count += current_batch_size
                continue
        
        # Prepare conditions on device
        if depth_index is not None:
            depth_index = depth_index.to('cuda', non_blocking=True)
            
        if argo_data is not None:
            for k, v in argo_data.items():
                if isinstance(v, torch.Tensor):
                    argo_data[k] = v.to('cuda', non_blocking=True)
        
        if mask is not None:
            mask = mask.to('cuda', non_blocking=True)
        
        if time_encoding is not None:
            time_encoding = time_encoding.to('cuda', non_blocking=True)

        for repeat_idx in range(num_samples_per_input):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Use CFG generation if cfg_scale > 1.0
                cfg_scale = getattr(args, 'cfg_scale', 1.0)
                if cfg_scale > 1.0:
                    sampled_images = model_without_ddp.generate_cfg(
                        current_batch_size, 
                        device='cuda',
                        argo_data=argo_data,
                        depth_index=depth_index,
                        mask=mask,
                        time_encoding=time_encoding
                    )
                else:
                    sampled_images = model_without_ddp.generate(
                        current_batch_size, 
                        device='cuda',
                        argo_data=argo_data,
                        depth_index=depth_index,
                        mask=mask,
                        time_encoding=time_encoding
                    )

            sampled_images_cpu = sampled_images.detach().cpu()
            mask_cpu = mask.cpu() if mask is not None else None

            future = executor.submit(
                save_batch_data, 
                sampled_images_cpu, 
                generated_count, 
                world_size, 
                local_rank, 
                dataset, 
                save_folder,
                repeat_idx,
                num_samples_per_input,
                mask_cpu
            )
            futures.append(future)

            # Check for failures in background tasks and clean up completed ones
            # This ensures we fail fast if any task encounters an error
            pending_futures = []
            for f in futures:
                if f.done():
                    exc = f.exception()
                    if exc:
                        print(f"Error immediately detected in saving task: {exc}")
                        executor.shutdown(wait=False)
                        raise exc
                else:
                    pending_futures.append(f)
            futures = pending_futures
            
        generated_count += current_batch_size

    # Wait for all saving tasks to complete
    executor.shutdown(wait=True)

    # Check for exceptions
    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"Error in saving task: {e}")
            import traceback
            traceback.print_exc()

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    
    # Print resume sampling statistics
    if skipped_batches > 0:
        print(f"\nResume sampling statistics: Skipped {skipped_batches} already-sampled batches")

    # Note: FID and IS metrics are not applicable for ocean data
    # Remove the evaluation metrics computation

    torch.distributed.barrier()
