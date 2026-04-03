# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
from datetime import timedelta
import torch
import torch.distributed as dist
from models.mpdit import MPDiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from otflow import OTFlow
from evaluator import calc_all_metric
import json
from torchvision.utils import save_image

def init_distributed_with_long_timeout():
    """Initialize distributed training with longer timeout."""
    # Set environment variables BEFORE dist.init_process_group
    os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour in seconds
    # Or use timeout parameter in init_process_group
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(hours=1)  # Increase from default 30min to 1 hour
    )
    print(f"Rank {dist.get_rank()}: Initialized with 1 hour timeout")
    

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


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    init_distributed_with_long_timeout()

    # Setup DDP:
    # dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    world_size = dist.get_world_size()
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Load model:
    args.latent_size = args.image_size // 8
    model = MPDiT_models[args.model](
        input_size=args.latent_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        num_condition_tokens=args.num_condition_tokens,
        time_emb=args.time_emb,
        skip_connection=args.skip_connection
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = torch.load(args.ckpt, map_location=f"cuda:{device}")["ema"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    mean = torch.tensor([5.81, 3.25, 0.12, -2.15]).view(1, -1, 1, 1).to(device)
    std =  torch.tensor([4.17, 4.62, 3.71, 3.28]).view(1, -1, 1, 1).to(device)

    otflow = OTFlow(model,
                    model,
                    image_size=args.latent_size,
                    in_channel=args.in_channels,)


    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples:
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    if args.sample_type == "ode":
        folder_name += "-ode"
    else:
        folder_name += "-sde"
        folder_name += f"-{args.diff_form}_{args.diff_norm}"
    exp_name = args.ckpt.split('/')[-3]
    sample_folder_dir = f"{args.sample_dir}/{exp_name}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # try to sample 8 image first to see if everything works
    ################################################################################
    if args.demo:
        if rank == 0:
            print("Testing sampling with 8 images...")
            n_test = 8
            z = torch.randn(n_test, model.in_channels, args.latent_size, args.latent_size, device=device)
            y = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
            # Sample from the flow matching
            if args.sample_type == "ode":
                samples = otflow.sample_euler(
                    model,
                    x=z,
                    class_idx=y,
                    device=device,
                    cfg=args.cfg_scale,
                    num_steps=args.num_sampling_steps,
                )
            elif args.sample_type == "sde":
                samples = otflow.sample_euler_maruyama(
                    model,
                    x=z,
                    class_idx=y,
                    device=device,
                    cfg=args.cfg_scale,
                    num_steps=args.num_sampling_steps,
                    diffusion_form=args.diff_form,
                    diffusion_norm=args.diff_norm
                )
            samples = samples * std + mean
            decoded_samples = []
            for i in range(0, samples.shape[0], 1):
                batch = samples[i:i+1]
                decoded_batch = vae.decode(batch).sample
                # Move to CPU and convert immediately to save GPU memory
                decoded_samples.append(decoded_batch)
            decoded_samples = torch.cat(decoded_samples, dim=0)
            #save sample images as grid
            save_image(decoded_samples, f"{sample_folder_dir}/test_samples.png", nrow=4, normalize=True, value_range=(-1, 1))
            print(f"Test sampling successful! Sampled images shape: {decoded_samples.shape}")
        exit(0)

    ################################################################################
    if args.cherry_pick:
        if rank == 0:
            cherry_pick_classes = [207, 250, 360, 387, 974, 88, 979, 417, 279, 475, 62, 388, 387, 281, 277, 971, 980, 812] #[33, 41, 37, 46, 55, 64, 76, 84] #[947, 933, 293, 291, 292, 113, 107] 
            for cls in cherry_pick_classes:
                print(f"Cherry-picking class {cls}")
                print("Cherry-pick sampling...")
                n_test = 27
                z = torch.randn(n_test, model.in_channels, args.latent_size, args.latent_size, device=device)
                y = torch.tensor([cls]*n_test, device=device)
                # Sample from the flow matching
                if args.sample_type == "ode":
                    samples = otflow.sample_euler(
                        model,
                        x=z,
                        class_idx=y,
                        device=device,
                        cfg=args.cfg_scale,
                        num_steps=args.num_sampling_steps,
                    )
                elif args.sample_type == "sde":
                    samples = otflow.sample_euler_maruyama(
                        model,
                        x=z,
                        class_idx=y,
                        device=device,
                        cfg=args.cfg_scale,
                        num_steps=args.num_sampling_steps,
                        diffusion_form=args.diff_form,
                        diffusion_norm=args.diff_norm
                    )
                samples = samples * std + mean
                decoded_samples = []
                for i in range(0, samples.shape[0], 1):
                    batch = samples[i:i+1]
                    decoded_batch = vae.decode(batch).sample
                    # save image png
                    decoded_samples.append(decoded_batch)
                    # Move to CPU and convert immediately to save GPU memory
                    decoded_batch = torch.clamp(127.5 * decoded_batch + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)[0].numpy()
                    img = Image.fromarray(decoded_batch)
                    img.save(f"{sample_folder_dir}/class_{cls}_sample_{i:02d}.png")
                decoded_samples = torch.cat(decoded_samples, dim=0)
                save_image(decoded_samples, f"{sample_folder_dir}/class_{cls}_samples.png", nrow=3, normalize=True, value_range=(-1, 1))
                # print(f"Cherry-pick sampling successful! Sampled images shape: {decoded_samples.shape}")

        exit(0)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    all_samples_numpy = []
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, args.latent_size, args.latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample from the flow matching
        if args.sample_type == "ode":
            samples = otflow.sample_euler(
                model,
                x=z,
                class_idx=y,
                device=device,
                cfg=args.cfg_scale,
                num_steps=args.num_sampling_steps,
            )
        elif args.sample_type == "sde":
            samples = otflow.sample_euler_maruyama(
                model,
                x=z,
                class_idx=y,
                device=device,
                cfg=args.cfg_scale,
                num_steps=args.num_sampling_steps,
                diffusion_form=args.diff_form,
                diffusion_norm=args.diff_norm
            )
        samples = samples * std + mean
        decoded_samples = []
        for i in range(0, samples.shape[0], 1):
            batch = samples[i:i+1]
            decoded_batch = vae.decode(batch).sample
            # Move to CPU and convert immediately to save GPU memory
            decoded_batch = torch.clamp(127.5 * decoded_batch + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)
            decoded_samples.append(decoded_batch)
            # Clear cache after each batch
            torch.cuda.empty_cache()

        # Concatenate and convert to numpy
        batch_numpy = torch.cat(decoded_samples, dim=0).numpy()
        all_samples_numpy.append(batch_numpy)
    all_sample_process = np.concatenate(all_samples_numpy, axis=0)
    np.savez(f"{sample_folder_dir}/temp_{rank}.npz", arr_0=all_sample_process)

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        all_data = []
        for rank in range(world_size):
            with np.load(f"{sample_folder_dir}/temp_{rank}.npz") as file:
                data = file["arr_0"]
                all_data.append(data)
            # delete temp file
            # os.remove(f"{sample_folder_dir}/temp_{rank}.npz")
        all_data = np.concatenate(all_data, 0)
        npz_path = f"{sample_folder_dir}/temp_final.npz"
        np.savez(npz_path, arr_0=all_data)
        print("Done Saving")

        if args.image_size == 256:
            metric_dict = calc_all_metric(npz_path, "VIRTUAL_imagenet256_labeled.npz")
        elif args.image_size == 512:
            metric_dict = calc_all_metric(npz_path, "VIRTUAL_imagenet512.npz")
        os.remove(f"{sample_folder_dir}/temp_final.npz")
        with open(os.path.join(sample_folder_dir, f"metrics_cfg{args.cfg_scale}.json"), "w") as f:
            json.dump(metric_dict, f, indent=4)
    dist.barrier()
    print("Sampling complete.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MPDiT_models.keys()), default="MPDiT-XL/4")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--per_proc_batch_size", type=int, default=32)
    parser.add_argument("--num_fid_samples", type=int, default=50_000)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale",  type=float, default=1.5)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a MPDiT checkpoint (default: auto-download a pre-trained MPDiT-XL/2 model).")
    parser.add_argument("--time_emb", type=str, default="mlp", choices=["mlp", "fno"], help="Type of time embedding to use in MPDiT models.")
    parser.add_argument("--use_rope", action="store_true", default=False, help="Use 2D RoPE in MPDiT models.")
    parser.add_argument("--num_condition_tokens", type=int, default=16, help="Number of condition tokens for MPDiT models.")
    parser.add_argument("--skip_connection", type=str, default=None, help="Enable skip connection in the model.")
    parser.add_argument("--demo", action="store_true", default=False, help="Run a quick demo to test sampling.")
    parser.add_argument("--sample_type", type=str, choices=["sde", "ode"], default="flow_matching", help="Type of sampling method.")
    parser.add_argument("--diff_norm", type=float, default=1)
    parser.add_argument("--diff_form", type=str, default="SBDM")
    parser.add_argument("--cherry_pick", action="store_true", default=False, help="Cherry pick some classes to sample.")
    
    args = parser.parse_args()
    main(args)