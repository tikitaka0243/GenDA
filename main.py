import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import yaml
import shutil
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from modules.datasets.glorys12_dataset import GLORYS12Dataset
import modules.util.misc as misc

import copy
from modules.flow_matching.engine import train_one_epoch, evaluate
from modules.flow_matching.denoiser import Denoiser


class TeeLogger:
    """Logger that writes to both file and stdout"""
    def __init__(self, log_file, mode='a'):
        self.file = open(log_file, mode, buffering=1)  # Line buffering
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


class ConfigNamespace:
    """Configuration namespace for YAML config"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config dict to ConfigNamespace for compatibility
    return ConfigNamespace(**config)


class OceanModelPipeline:
    """Complete pipeline for ocean data generation model training and sampling"""
    
    def __init__(self, args, config_path=None):
        self.args = args
        self.config_path = config_path
        self.device = None
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.data_loader_train = None
        self.log_writer = None
        self.seed = None
        self.tee_logger = None
        
        # Initialize distributed mode first to allow syncing timestamp
        misc.init_distributed_mode(self.args)
        
        # Handle resume sampling mode
        self._handle_resume_sampling()
        
        self._setup_output_directory()
        self._initialize()
    
    def _handle_resume_sampling(self):
        """Handle resume sampling: move samples from previous run to new output directory"""
        if not getattr(self.args, 'evaluate_gen', False):
            return
        
        resume_sampling_dir = getattr(self.args, 'resume_sampling_dir', None)
        if resume_sampling_dir is None or resume_sampling_dir == 'null':
            return
        
        if not os.path.exists(resume_sampling_dir):
            print(f"Warning: Resume sampling directory does not exist: {resume_sampling_dir}")
            return
        
        # Store resume sampling info for later use
        self.args._resume_sampling_source = resume_sampling_dir
        print(f"Resume sampling mode enabled. Previous samples will be moved from: {resume_sampling_dir}")
    
    def _move_existing_samples(self, new_samples_dir):
        """Move existing samples from resume directory to new output directory"""
        resume_dir = getattr(self.args, '_resume_sampling_source', None)
        if resume_dir is None:
            return
        
        old_samples_dir = os.path.join(resume_dir, 'samples')
        if not os.path.exists(old_samples_dir):
            print(f"Warning: Old samples directory does not exist: {old_samples_dir}")
            return
        
        # Create new samples directory if needed
        os.makedirs(new_samples_dir, exist_ok=True)
        
        # Count files to move
        files_to_move = [f for f in os.listdir(old_samples_dir) if f.endswith('.nc')]
        
        if len(files_to_move) == 0:
            print("No existing samples to move")
            return
        
        print(f"\n{'='*60}")
        print(f"Moving {len(files_to_move)} existing samples from previous run")
        print(f"From: {old_samples_dir}")
        print(f"To: {new_samples_dir}")
        print(f"{'='*60}\n")
        
        # Move files
        moved_count = 0
        for filename in files_to_move:
            src = os.path.join(old_samples_dir, filename)
            dst = os.path.join(new_samples_dir, filename)
            try:
                shutil.move(src, dst)
                moved_count += 1
            except Exception as e:
                print(f"Error moving file {filename}: {e}")
        
        print(f"Successfully moved {moved_count} sample files")
        
        # Also try to copy conformal directory if it exists
        old_conformal_dir = os.path.join(resume_dir, 'conformal')
        if os.path.exists(old_conformal_dir):
            new_conformal_dir = os.path.join(self.args.output_dir, 'conformal')
            try:
                shutil.copytree(old_conformal_dir, new_conformal_dir, dirs_exist_ok=True)
                print(f"Copied conformal directory to: {new_conformal_dir}")
            except Exception as e:
                print(f"Warning: Could not copy conformal directory: {e}")
    
    def _setup_output_directory(self):
        """Setup timestamped output directory and save config"""
        # Create timestamped output directory
        if misc.get_rank() == 0:
            current_time = time.time()
        else:
            current_time = 0.0
            
        if misc.is_dist_avail_and_initialized():
            # Broadcast timestamp to ensure all ranks use the same time
            t = torch.tensor(current_time, device='cuda')
            torch.distributed.broadcast(t, src=0)
            current_time = t.item()
            
        timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')
        base_output_dir = self.args.output_dir
        
        # Determine run_prefix: use custom value from config if specified, otherwise auto-determine
        run_prefix = getattr(self.args, 'run_prefix', None)
        if run_prefix is None:
            run_prefix = 'sample' if getattr(self.args, 'evaluate_gen', False) else 'run'
        
        timestamped_output_dir = os.path.join(base_output_dir, f'{run_prefix}_{timestamp}')
        self.args.output_dir = timestamped_output_dir
        
        # Create output directory structure
        Path(timestamped_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup text log file
        log_filename = f'{run_prefix}_{timestamp}.log'
        self.log_file_path = os.path.join(timestamped_output_dir, log_filename)
        
        # Redirect stdout to both console and log file
        self.tee_logger = TeeLogger(self.log_file_path, mode='w')
        sys.stdout = self.tee_logger
        sys.stderr = self.tee_logger
        
        print(f"Output directory: {timestamped_output_dir}")
        print(f"Log file: {self.log_file_path}")
        
        # Save config to the timestamped directory if config_path is provided
        if self.config_path:
            config_filename = os.path.basename(self.config_path)
            config_save_path = os.path.join(timestamped_output_dir, config_filename)
            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            shutil.copy2(self.config_path, config_save_path)
            print(f"Config saved to: {config_save_path}")
        
        # Backup all Python code files
        self._backup_code(timestamped_output_dir)
    
    def _backup_code(self, output_dir):
        """Backup all Python code files to output directory"""
        # Get project root directory (where main.py is located)
        project_root = os.path.dirname(os.path.realpath(__file__))
        
        # Create code backup directory
        code_backup_dir = os.path.join(output_dir, 'code_backup')
        os.makedirs(code_backup_dir, exist_ok=True)
        
        # Track number of files copied
        file_count = 0
        
        # Walk through project directory and copy all .py files
        for root, dirs, files in os.walk(project_root):
            # Skip output directory and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'output' and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    # Get source file path
                    src_file = os.path.join(root, file)
                    
                    # Calculate relative path to preserve directory structure
                    rel_path = os.path.relpath(src_file, project_root)
                    
                    # Create destination path
                    dst_file = os.path.join(code_backup_dir, rel_path)
                    
                    # Create destination directory if needed
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(src_file, dst_file)
                    file_count += 1
        
        print(f"Code backup completed: {file_count} Python files copied to {code_backup_dir}")
    
    def _initialize(self):
        """Initialize training environment"""
        # Initialize distributed mode (moved to __init__)
        # misc.init_distributed_mode(self.args)
        print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
        print("Arguments:\n{}".format(self.args).replace(', ', ',\n'))
        
        # Set device
        self.device = torch.device(self.args.device)
        
        # Enable TensorFloat32 for better performance on Ampere GPUs
        torch.set_float32_matmul_precision('high')
        print('TensorFloat32 enabled for float32 matrix multiplication')
        
        # Set seeds for reproducibility
        self.seed = self.args.seed + misc.get_rank()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cudnn.benchmark = True
        
        # Set up TensorBoard logging (only on main process)
        global_rank = misc.get_rank()
        if global_rank == 0 and self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)
            # Only create checkpoints directory and TensorBoard log in training mode
            if not getattr(self.args, 'evaluate_gen', False):
                # Create subdirectories for checkpoints and configs
                os.makedirs(os.path.join(self.args.output_dir, 'checkpoints'), exist_ok=True)
                # Log directory is handled by TensorBoard
                self.log_writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir))
            else:
                self.log_writer = None
        else:
            self.log_writer = None
    
    def _build_dataloader(self, is_train=True):
        """Build dataset and data loader"""
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        
        # Create GLORYS12 dataset
        argo_file = getattr(self.args, 'argo_file', None)
        if not getattr(self.args, 'conditional', False):
            argo_file = None
            
        dataset_type = 'train' if is_train else 'test'
        if not is_train:
            if hasattr(self.args, 'sampling_dataset') and self.args.sampling_dataset:
                dataset_type = self.args.sampling_dataset

        dataset = GLORYS12Dataset(
            data_dir=self.args.data_path,
            dataset_type=dataset_type,
            variables=['thetao', 'so', 'uo', 'vo'],
            normalize=True,  # Use normalization to [-1, 1]
            argo_file=argo_file,
            argo_days=getattr(self.args, 'argo_days', 1),
            max_argo_profiles_per_day=getattr(self.args, 'max_argo_profiles_per_day', None),
            normalization_method=getattr(self.args, 'normalization_method', 'minmax'),
            velocity_weight=getattr(self.args, 'velocity_weight', 1.0),
            use_time_embed=getattr(self.args, 'use_time_embed', False)
        )
        print(f"Dataset ({dataset_type}): {dataset}")

        # Subsample validation set if requested
        if not is_train and hasattr(self.args, 'val_sample_ratio') and self.args.val_sample_ratio < 1.0:
            total_size = len(dataset)
            subset_size = int(total_size * self.args.val_sample_ratio)
            if subset_size > 0:
                # Set seed for reproducibility
                seed = getattr(self.args, 'val_sample_seed', 42)
                g = torch.Generator()
                g.manual_seed(seed)
                indices = torch.randperm(total_size, generator=g)[:subset_size].tolist()
                
                dataset = torch.utils.data.Subset(dataset, indices)
                print(f"Subsampled validation dataset: {subset_size}/{total_size} samples (seed={seed})")
            else:
                print(f"Warning: val_sample_ratio {self.args.val_sample_ratio} resulted in 0 samples. Using full dataset.")
        
        # Create distributed sampler
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
        print(f"Sampler ({dataset_type}) =", sampler)
        
        # Import custom collate function
        from modules.datasets.glorys12_dataset import glorys_argo_collate_fn
        
        batch_size = self.args.batch_size if is_train else self.args.gen_bsz

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=is_train,
            collate_fn=glorys_argo_collate_fn,
            persistent_workers=self.args.num_workers > 0
        )
        return data_loader
    
    def _build_model(self):
        """Build model and optimizer"""
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False
        
        # Create denoiser model
        self.model = Denoiser(self.args)

        print("Model =", self.model)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
        
        self.model.to(self.device)
        
        # Calculate learning rate
        eff_batch_size = self.args.batch_size * misc.get_world_size()
        if self.args.lr is None:  # only base_lr (blr) is specified
            self.args.lr = self.args.blr * eff_batch_size / 256
        
        print("Base lr: {:.2e}".format(self.args.lr * 256 / eff_batch_size))
        print("Actual lr: {:.2e}".format(self.args.lr))
        print("Effective batch size: %d" % eff_batch_size)
        
        # Wrap model with DDP
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.gpu]
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model
        
        # Set up optimizer
        param_groups = misc.add_weight_decay(self.model_without_ddp, self.args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))
        print(self.optimizer)
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint if provided"""
        if self.args.resume:
            # Determine checkpoint filename from config
            checkpoint_name = getattr(self.args, 'checkpoint_name', 'checkpoint-last.pth')
            checkpoint_path = os.path.join(
                self.args.resume, "checkpoints", checkpoint_name
            )
            print(f"Loading checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            msg = self.model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            
            # Check for missing keys that are NOT grad_criterion (which can be ignored during inference)
            unexpected_missing = [k for k in msg.missing_keys if "grad_criterion" not in k]
            if unexpected_missing:
                raise RuntimeError(f"Missing keys in state_dict: {unexpected_missing}")
            
            if msg.missing_keys:
                print(f"Ignored missing keys (grad_criterion): {[k for k in msg.missing_keys if 'grad_criterion' in k]}")
            
            ema_state_dict1 = checkpoint['model_ema1']
            ema_state_dict2 = checkpoint['model_ema2']
            self.model_without_ddp.ema_params1 = [
                ema_state_dict1[name].cuda() 
                for name, _ in self.model_without_ddp.named_parameters()
            ]
            self.model_without_ddp.ema_params2 = [
                ema_state_dict2[name].cuda() 
                for name, _ in self.model_without_ddp.named_parameters()
            ]
            print("Resumed checkpoint from", self.args.resume)
            
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.args.start_epoch = checkpoint['epoch'] + 1
                print("Loaded optimizer & scaler state!")
            del checkpoint
        else:
            self.model_without_ddp.ema_params1 = copy.deepcopy(
                list(self.model_without_ddp.parameters())
            )
            self.model_without_ddp.ema_params2 = copy.deepcopy(
                list(self.model_without_ddp.parameters())
            )
            print("Training from scratch")
    
    def evaluate_generation(self):
        """Evaluate generation quality"""
        # Build dataloader for generation
        data_loader = self._build_dataloader(is_train=False)
        
        # Move existing samples to new output directory when in resume sampling mode
        save_folder = os.path.join(self.args.output_dir, "samples")
        self._move_existing_samples(save_folder)
        
        print("Evaluating checkpoint at {} epoch".format(self.args.start_epoch))
        with torch.random.fork_rng(devices=[self.args.gpu]):
            torch.manual_seed(self.seed)
            with torch.no_grad():
                evaluate(
                    self.model_without_ddp, 
                    self.args, 
                    data_loader
                )
        
        # Close log file after evaluation
        if self.tee_logger is not None:
            sys.stdout = self.tee_logger.stdout
            sys.stderr = self.tee_logger.stdout
            self.tee_logger.close()
    
    def train(self):
        """Main training loop"""
        print(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # Set epoch for sampler
            if self.args.distributed:
                self.data_loader_train.sampler.set_epoch(epoch)
            
            # Train one epoch
            train_one_epoch(
                self.model, 
                self.model_without_ddp, 
                self.data_loader_train, 
                self.optimizer, 
                self.device, 
                epoch, 
                log_writer=self.log_writer, 
                args=self.args
            )
            
            save_freq = getattr(self.args, 'save_checkpoint_freq', 10)
            if (epoch + 1) % save_freq == 0 and epoch > 0:
                misc.save_model(
                    args=self.args,
                    model_without_ddp=self.model_without_ddp,
                    optimizer=self.optimizer,
                    epoch=epoch
                )
            
            if misc.is_main_process() and self.log_writer is not None:
                self.log_writer.flush()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time:', total_time_str)
        
        # Close log file
        if self.tee_logger is not None:
            sys.stdout = self.tee_logger.stdout
            sys.stderr = self.tee_logger.stdout
            self.tee_logger.close()
    
    def run(self):
        """Run the complete training pipeline"""
        # Build model
        self._build_model()
        
        # Resume from checkpoint if needed
        self._resume_from_checkpoint()
        
        # Evaluate or train
        if self.args.evaluate_gen:
            self.evaluate_generation()
        else:
            self.data_loader_train = self._build_dataloader(is_train=True)
            self.train()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ocean Data Assimilation Model Training and Sampling')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'sample'],
        help='Run mode: train or sample (default: train)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/configs.yaml',
        help='Path to config file (default: ./configs/configs.yaml)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    cmd_args = parse_arguments()
    
    # Load configuration from YAML file
    config_path = cmd_args.config
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Running in {cmd_args.mode.upper()} mode")
    print(f"Loading config from: {config_path}")
    
    # Load configuration
    args = load_config(config_path)
    
    # Override evaluate_gen based on mode
    if cmd_args.mode == 'sample':
        args.evaluate_gen = True
    else:
        args.evaluate_gen = False
    
    # Set default checkpoint_name if not in config
    if not hasattr(args, 'checkpoint_name'):
        args.checkpoint_name = 'checkpoint-last.pth'
        print(f"Using default checkpoint: checkpoint-last.pth")
    else:
        print(f"Using checkpoint from config: {args.checkpoint_name}")
    
    # Create pipeline and run
    pipeline = OceanModelPipeline(args, config_path=config_path)
    pipeline.run()
