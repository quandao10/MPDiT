# diff_form [SBDM, constant, sigma]

# write for loop to run multiple experiments with different diff_norm values and cfg_scale values
# for cfg_scale in 1.35 1.4; do
#     torchrun --master_port=29501 --nnodes=1 --nproc_per_node=8 sample.py --model MPDiT-XL/4x \
#                                                                             --num_fid_samples 50000 \
#                                                                             --image_size 256 \
#                                                                             --cfg_scale $cfg_scale \
#                                                                             --num_sampling_steps 250 \
#                                                                             --ckpt /your/path/to/ckpt.pt \
#                                                                             --sample_dir samples/ \
#                                                                             --per_proc_batch_size 32 \
#                                                                             --global_seed 0 \
#                                                                             --num_classes 1000 \
#                                                                             --num_condition_tokens 16 \
#                                                                             --time_emb fno \
#                                                                             --skip_connection \
#                                                                             --sample_type sde \
#                                                                             --diff_norm 1 \ 
#                                                                             --diff_form SBDM \
# done




# for cfg_scale in 1.375; do
#     torchrun --master_port=29501 --nnodes=1 --nproc_per_node=8 sample.py --model MPDiT-XL/4x \ 
#                                                                             --num_fid_samples 50000 \
#                                                                             --image_size 512 \
#                                                                             --cfg_scale $cfg_scale \
#                                                                             --num_sampling_steps 250 \
#                                                                             --ckpt /your/path/to/ckpt.pt \
#                                                                             --sample_dir samples/ \
#                                                                             --per_proc_batch_size 32 \
#                                                                             --global_seed 0 \
#                                                                             --num_classes 1000 \
#                                                                             --num_condition_tokens 16 \
#                                                                             --time_emb fno \
#                                                                             --skip_connection \
#                                                                             --sample_type ode \
#                                                                             --diff_norm 1 \
# done
