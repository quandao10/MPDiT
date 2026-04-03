# extract feature of imagenet
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 \
                                                            --data-path /common/users/qd66/dataset/imagenet/imagenet \
                                                            --features-path /common/users/qd66/dataset/imagenet_feature \
                                                            --image-size 256 \


# download statistics of imagenet
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz