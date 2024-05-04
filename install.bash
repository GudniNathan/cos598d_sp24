apt-get update
apt-get -y install htop dstat python3-pip
apt-get -y install libaio-dev


cd $HOME
git clone https://github.com/pytorch/PiPPy/ $HOME/PiPPy
cd $HOME/PiPPy
pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html
python setup.py install


pip3 install torch torchvision
pip3 install tqdm boto3 requests regex sentencepiece sacremoses
pip3 install pytorch-transformers 
pip3 install scipy
pip3 install scikit-learn
pip3 install deepspeed
pip3 install transformers

cd $HOME
git clone https://github.com/NVIDIA/cutlass $HOME/cutlass
export CUTLASS_PATH=$HOME/cutlass

# Download the GLUE data
mkdir glue_data
cd cos598d_sp24
python3 download_glue_data.py --data_dir $HOME/glue_data