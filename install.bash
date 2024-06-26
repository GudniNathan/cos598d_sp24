apt-get update
apt-get -y install htop dstat python3-pip
apt-get -y install libaio-dev


pip uninstall -y torch
pip uninstall -y torch

if [ ! -d $HOME/PiPPy ]; then
  cd $HOME
  git clone https://github.com/pytorch/PiPPy/ $HOME/PiPPy
fi 
cd $HOME/PiPPy
pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
python setup.py install


# pip3 install torchvision
pip3 install tqdm boto3 requests regex sentencepiece sacremoses
pip3 install pytorch-transformers 
pip3 install scipy
pip3 install scikit-learn
pip3 install deepspeed
pip3 install transformers
pip3 install memory-profiler
pip3 install tensorboard
pip install torch_tb_profiler

cd $HOME
if [ ! -d $HOME/cutlass ]; then
    git clone https://github.com/NVIDIA/cutlass $HOME/cutlass
fi 
export CUTLASS_PATH=$HOME/cutlass

# Download the GLUE data
if [ ! -d glue_data ]; then
  # Control will enter here if glue_data doesn't exist.
    mkdir glue_data
    cd cos598d_sp24
    python3 download_glue_data.py --data_dir $HOME/glue_data
fi