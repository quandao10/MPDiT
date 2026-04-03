conda create -n mpdit python=3.10 -y
conda activate mpdit
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow[and-cuda]==2.17.0
pip install einops accelerate diffusers==0.30 wandb scikit-learn calflops timm transformers
