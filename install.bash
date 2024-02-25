sudo apt-get update ; sudo apt-get install htop dstat python3-pip
pip3 install torch torchvision

cd $HOME
mkdir glue_data
cd cos598d
python3 download_glue_data.py --data_dir $HOME/glue_data