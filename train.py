# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import argparse
import logging
import os
import torch
import torch.utils
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
from accelerate import Accelerator
from models.mpdit import MPDiT_models
from diffusers.models import AutoencoderKL
import torchvision
import json
from tqdm import tqdm
import math
from evaluator import calc_all_metric
from otflow import OTFlow

#################################################################################
#                             Training Helper Functions                         #
#################################################################################



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

@torch.no_grad
def evaluator(args,
              flow,
              accelerator,
              model,
              vae_package,
              sample_dir,
              cfg,
              vae_batch_size=1):
    """
    Evaluates the current model by sampling images and calculating metrics.
    """
    world_size = accelerator.num_processes
    n = 32 # per process batch size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if accelerator.is_main_process:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if accelerator.is_main_process else pbar
    vae = vae_package["vae"]
    mean = vae_package["mean"]
    std = vae_package["std"]

    # Save samples directly to disk to avoid OOM
    all_samples_numpy = []

    for _ in pbar:
        x = torch.randn((n, args.in_channels, args.latent_size, args.latent_size), device=accelerator.device)
        samples = flow.sample_euler(model,
                                x = x,
                                class_idx=torch.randint(0, args.num_classes, (n,), device=accelerator.device),
                                device="cuda",
                                cfg=cfg,
                                num_steps=250)

        # Denormalize
        samples = samples * std + mean

        # Decode VAE in smaller batches to save memory
        decoded_samples = []
        for i in range(0, samples.shape[0], vae_batch_size):
            batch = samples[i:i+vae_batch_size]
            decoded_batch = vae.decode(batch).sample
            # Move to CPU and convert immediately to save GPU memory
            decoded_batch = torch.clamp(127.5 * decoded_batch + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)
            decoded_samples.append(decoded_batch)
            # Clear cache after each batch
            torch.cuda.empty_cache()

        # Concatenate and convert to numpy
        batch_numpy = torch.cat(decoded_samples, dim=0).numpy()
        all_samples_numpy.append(batch_numpy)

        # Clear GPU memory
        del samples, decoded_samples, batch_numpy
        torch.cuda.empty_cache()

    # Concatenate all samples
    all_sample_process = np.concatenate(all_samples_numpy, axis=0)
    del all_samples_numpy

    # Save to temporary file
    np.savez(f"{sample_dir}_temp_{accelerator.process_index}.npz", arr_0=all_sample_process)
    del all_sample_process

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if  accelerator.is_main_process:
        all_data = []
        for idx in range(world_size):
            with np.load(f"{sample_dir}_temp_{idx}.npz") as file:
                data = file["arr_0"]
                all_data.append(data)
            # delete temp file
            os.remove(f"{sample_dir}_temp_{idx}.npz")
        all_data = np.concatenate(all_data, 0)
        npz_path = f"{sample_dir}.npz"
        np.savez(npz_path, arr_0=all_data)
        print("Done Saving")

    dist.barrier()
    # compute all metric here
    if  accelerator.is_main_process:
        if args.resolution == 256:
            metric_dict = calc_all_metric(npz_path, "VIRTUAL_imagenet256_labeled.npz")
        elif args.resolution == 512:
            metric_dict = calc_all_metric(npz_path, "VIRTUAL_imagenet512.npz")
        os.remove(f"{sample_dir}.npz")
        torch.cuda.empty_cache()
    else:
        metric_dict = {}
    return metric_dict

    

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features).squeeze(0), torch.from_numpy(labels).squeeze(0)
    
class ImageNet_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, vae_type="vae"):
        # VAE
        self.base_dir = base_dir
        self.vae_path = os.path.join(base_dir, vae_type)
        self.meta_file = os.path.join(self.vae_path, "dataset.json")
        with open(self.meta_file, 'r') as file:
            self.meta_data = json.load(file)["labels"]

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        vae_file, label = self.meta_data[index]
        label = torch.tensor(int(label))
        vae_data = np.load(os.path.join(self.vae_path, vae_file))
        mean, std = np.array_split(vae_data, indices_or_sections=2, axis=0) 
        mean, std = torch.from_numpy(mean), torch.from_numpy(std)   
        sample = mean + torch.randn_like(mean) * std
        return sample, label

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    experiment_dir = f"{args.results_dir}/{args.exp_name}"  # Create an experiment folder
    eval_dir = f"{experiment_dir}/eval"
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"
        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    args.latent_size = int(args.resolution // args.vae_rate)  # latent size depends on data resolution
    if args.model in MPDiT_models.keys():
        model = MPDiT_models[args.model](
            input_size=args.latent_size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            num_condition_tokens=args.num_condition_tokens,
            time_emb=args.time_emb,
            skip_connection=args.skip_connection
        )
    else:
        raise ValueError(f"Model {args.model} not found in available models.")

    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)

    # Apply torch.compile if requested
    if args.compile:
        if accelerator.is_main_process:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.compile_mode)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Resume training from a checkpoint if provided and exists
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(device))
        init_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = checkpoint["train_steps"]
        if accelerator.is_main_process:
            logger.info("=> loaded checkpoint (epoch {})".format(init_epoch))
        del checkpoint
    else:
        init_epoch = 0
        train_steps = 0
        # Ensure EMA is initialized with synced weights
        update_ema(ema, model, decay=0)

    # Setup data:
    if args.resolution == 256:
        dataset = CustomDataset(args.feature_path + "/imagenet256_features",
                                args.feature_path + "/imagenet256_labels")
    elif args.resolution == 512:
        dataset = CustomDataset(args.feature_path + "/imagenet512_vae_features",
                                args.feature_path + "/imagenet512_labels")
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")
    
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)
    flow = OTFlow(model,
                ema,
                image_size=args.latent_size,
                in_channel=args.in_channels,
                p_std=args.p_std,
                p_mean=args.p_mean,
                time_sampler=args.time_sampler,
                reweight=args.reweight,
                loss_type=args.loss_type,
                )
    # mean, std for imagenet256 from Karras et al.
    mean = torch.tensor([5.81, 3.25, 0.12, -2.15]).view(1, -1, 1, 1).to(device)
    std =  torch.tensor([4.17, 4.62, 3.71, 3.28]).view(1, -1, 1, 1).to(device)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(init_epoch, args.epochs+1):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in tqdm(loader, total=len(loader)):
            x = x.to(device)
            y = y.to(device)
            x = (x - mean) / std
            loss = flow.loss(x, y)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model, decay=0.9999)  # Update EMA model with a decay factor

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.8f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
        if accelerator.is_main_process:
            if epoch % args.plot_every == 0:
                # sample from train model and ema model
                with torch.no_grad():
                    x_train = flow.sample_euler(model.module, device=device)
                    x_ema = flow.sample_euler(ema, device=device)
                    x = torch.cat([x_train, x_ema], dim=0)
                    # denormalize x to original range
                    x = x * std + mean
                    samples = [vae.decode(x[i].unsqueeze(0)).sample for i in range(x.shape[0])]
                    samples = torch.cat(samples, dim=0)
                    torchvision.utils.save_image(samples, f"{samples_dir}/epoch_{epoch:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))
                logger.info("Saved unguided image")
                # save CFG images
                with torch.no_grad():
                    x_train = flow.sample_euler(model.module, device=device, cfg=2)
                    x_ema = flow.sample_euler(ema, device=device, cfg=2)
                    x = torch.cat([x_train, x_ema], dim=0)
                    # denormalize x to original range
                    x = x * std + mean
                    samples = [vae.decode(x[i].unsqueeze(0)).sample for i in range(x.shape[0])]
                    samples = torch.cat(samples, dim=0)
                    torchvision.utils.save_image(samples, f"{samples_dir}/epoch_{epoch:07d}_cfg2.png", nrow=4, normalize=True, value_range=(-1, 1))
                logger.info("Saved cfg image")

                
            if epoch % args.ckpt_every == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "train_steps": train_steps,
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if epoch % args.ckpt_every == 0 and epoch >= 80:
            # After saving checkpoint, do evaluation here
            vae_package = {
                "vae": vae,
                "mean": mean,
                "std": std
            }
            
            for scale in [1.0, 1.5, 2.0]:
                metric_dict = evaluator(args,
                    flow, 
                    accelerator, 
                    ema, 
                    vae_package, 
                    eval_dir+f"_ep{epoch}_cfg{scale}",
                    cfg=scale)
                if accelerator.is_main_process:
                    with open(os.path.join(eval_dir, f"metrics_cfg{scale}_epoch{epoch}.json"), "w") as f:
                        json.dump(metric_dict, f, indent=4)
                torch.cuda.empty_cache()
            if accelerator.is_main_process:
                logger.info("Finish Evaluation")
            
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="features")
    parser.add_argument("--exp_name", type=str, default="meanflow")
    parser.add_argument("--plot_every", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(MPDiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae_rate", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20)
    parser.add_argument("--p_std", type=float, default=0.1)
    parser.add_argument("--p_mean", type=float, default=-0.4)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--time_sampler", type=str, choices=["lognormal", "uniform"], default="lognormal")
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--reweight", type=str, default="uniform")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer. Default is 1e-4.")
    parser.add_argument("--loss_type", type=str, default="l2", help="Loss type to use for training.")
    parser.add_argument("--time_emb", type=str, default="mlp", choices=["mlp", "fno"], help="Type of time embedding to use in DiT models.")
    parser.add_argument("--num_condition_tokens", type=int, default=16, help="Number of condition tokens for DiT models.")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile for model compilation.")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], help="Compilation mode for torch.compile.")
    parser.add_argument("--skip_connection", action="store_true", default=True, help="Enable skip connection in the model.")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512], help="Data resolution")
    args = parser.parse_args()
    main(args)
