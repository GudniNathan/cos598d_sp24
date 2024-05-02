apt-get update
apt-get -y install htop dstat python3-pip
apt-get -y install libaio-dev

# Install PyTorch and other dependencies
pip3 uninstall --y torch torchvision torchaudio clean-fid xformers
pip3 install torch torchvision torchaudio clean-fid xformers

pip3 install torch torchvision
pip3 install tqdm boto3 requests regex sentencepiece sacremoses
pip3 install pytorch-transformers 
pip3 install scipy
pip3 install scikit-learn
pip3 install deepspeed

pip3 --y uninstall triton
pip install triton==1.0.0


cd $HOME
git clone https://github.com/NVIDIA/cutlass $HOME/cutlass
export CUTLASS_PATH=$HOME/cutlass

# Download the GLUE data
mkdir glue_data
cd cos598d_sp24
python3 download_glue_data.py --data_dir $HOME/glue_data