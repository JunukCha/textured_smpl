conda create -n smpl python=3.10 -y
conda activate smpl

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
pip install numpy==1.23.5