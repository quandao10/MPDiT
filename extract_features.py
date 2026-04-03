# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Enhanced to support RADIO feature extraction.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm
from diffusers.models import AutoencoderKL



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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Extract features using VAE encoder.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        feature_subdir = f'imagenet{args.image_size}_{args.encoder_type}_features'
        label_subdir = f'imagenet{args.image_size}_labels'
        os.makedirs(os.path.join(args.features_path, feature_subdir), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, label_subdir), exist_ok=True)

    # Create encoder model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    encoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    feature_subdir = f'imagenet{args.image_size}_{args.encoder_type}_features'
    label_subdir = f'imagenet{args.image_size}_labels'

    for train_steps, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        # check file existence to skip already extracted features
        # if os.path.exists(f'{args.features_path}/{label_subdir}/{train_steps}.npy'):
        #     continue
        with torch.no_grad():
            features = encoder.encode(x).latent_dist.sample()
        features = features.detach().cpu().numpy()
        np.save(f'{args.features_path}/{feature_subdir}/{train_steps}.npy', features)

        y = y.detach().cpu().numpy()
        np.save(f'{args.features_path}/{label_subdir}/{train_steps}.npy', y)

if __name__ == "__main__":
    # Extract features using VAE
    parser = argparse.ArgumentParser(description='Extract features from images using VAE')
    parser.add_argument("--data-path", type=str, required=True, help="Path to image dataset")
    parser.add_argument("--features-path", type=str, default="features", help="Path to save features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)

    # Encoder selection
    parser.add_argument("--encoder-type", type=str, choices=["vae"], default="vae",
                        help="Type of encoder to use (vae is currently the only option)")

    # VAE-specific arguments
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="VAE variant (only used if encoder-type=vae)")

    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
