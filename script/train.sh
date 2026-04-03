



##################################################################### 256 model ###########################################################################




# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --main_process_port 29508 --gpu_ids 0,1,2,3,4,5,6,7 train.py \
#     --model MPDiT-B/4x \
#     --feature_path ./imagenet_feature \
#     --exp_name your_exp_name \
#     --results_dir /your/results/dir/ \
#     --ckpt_every 40 \
#     --p_std 1.0 \
#     --p_mean 0 \
#     --epochs 240 \
#     --global_batch_size 1024 \
#     --global_seed 0 \
#     --vae ema \
#     --latent_size 32 \
#     --in_channels 4 \
#     --num_classes 1000 \
#     --num_workers 4 \
#     --log_every 100 \
#     --time_sampler lognormal \
#     --reweight uniform \
#     --lr 2e-4 \
#     --loss_type l2 \
#     --time_emb fno \
#     --num_condition_tokens 16 \
#     --skip_connection \
#     --compile \
#     --model_ckpt /your/path/to/ckpt_to_resume.pt \


# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --main_process_port 29508 --gpu_ids 0,1,2,3,4,5,6,7 train.py \
#     --model MPDiT-XL/4x \
#     --feature_path ./imagenet_feature \
#     --exp_name your_exp_name \
#     --results_dir /your/results/dir/ \
#     --ckpt_every 20 \
#     --p_std 1.0 \
#     --p_mean 0 \
#     --epochs 400 \
#     --global_batch_size 1024 \
#     --global_seed 0 \
#     --vae ema \
#     --latent_size 32 \
#     --in_channels 4 \
#     --num_classes 1000 \
#     --num_workers 4 \
#     --log_every 100 \
#     --time_sampler lognormal \
#     --reweight uniform \
#     --lr 2e-4 \
#     --loss_type l2 \
#     --mask_ratio 0.75 \
#     --mask_type spatial \
#     --time_emb fno \
#     --num_condition_tokens 16 \
#     --skip_connection \
#     --compile \
#     --model_ckpt /your/path/to/ckpt_to_resume.pt \




##################################################################### 512 model ###########################################################################




# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --main_process_port 29508 --gpu_ids 0,1,2,3,4,5,6,7 train.py \
#     --model MPDiT-XL/4x \
#     --feature_path ./imagenet_feature \
#     --exp_name your_exp_name \
#     --results_dir /your/results/dir/ \
#     --ckpt_every 5 \
#     --p_std 1.0 \
#     --p_mean 0 \
#     --epochs 120 \
#     --global_batch_size 256 \
#     --global_seed 0 \
#     --vae ema \
#     --resolution 512 \
#     --vae_rate 8 \
#     --in_channels 4 \
#     --num_classes 1000 \
#     --num_workers 4 \
#     --log_every 100 \
#     --time_sampler lognormal \
#     --reweight uniform \
#     --lr 1e-4 \
#     --loss_type l2 \
#     --time_emb fno \
#     --num_condition_tokens 16 \
#     --skip_connection \
#     --compile \
#     --model_ckpt /your/path/to/ckpt_to_resume.pt


# accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --main_process_port 29508 --gpu_ids 0,1,2,3,5,6,7 train.py \
#     --model MPDiTv2-XL \
#     --feature_path ./imagenet_feature \
#     --exp_name your_exp_name \
#     --results_dir /your/results/dir/ \
#     --ckpt_every 5 \
#     --p_std 1.0 \
#     --p_mean 0 \
#     --epochs 120 \
#     --global_batch_size 256 \
#     --global_seed 0 \
#     --vae ema \
#     --resolution 512 \
#     --vae_rate 8 \
#     --in_channels 4 \
#     --num_classes 1000 \
#     --num_workers 4 \
#     --log_every 100 \
#     --time_sampler lognormal \
#     --reweight uniform \
#     --lr 1e-4 \
#     --loss_type l2 \
#     --time_emb fno \
#     --num_condition_tokens 16 \
#     --skip_connection \
#     --compile \
#     --model_ckpt /your/path/to/ckpt_to_resume.pt \