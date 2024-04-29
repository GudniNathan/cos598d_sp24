apt-get update
apt-get -y install htop dstat python3-pip

# Install PyTorch and other dependencies
pip3 install torch torchvision
pip3 install tqdm boto3 requests regex sentencepiece sacremoses
pip3 install pytorch-transformers 
pip3 install scipy
pip3 install scikit-learn

# Download the GLUE data
cd $HOME
mkdir glue_data
cd cos598d_sp24
python3 download_glue_data.py --data_dir $HOME/glue_data