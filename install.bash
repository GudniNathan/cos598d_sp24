sudo apt-get update ; sudo apt-get install htop dstat python3-pip
pip3 install torch torchvision
pip3 install tqdm

cd $HOME
mkdir glue_data
cd cos598d_sp24
python3 download_glue_data.py --data_dir $HOME/glue_data