sudo apt update
sudo apt install libmpich-dev -y
sudo apt install mpich libopenmpi-dev -y
#pip install --extra-index-url https://pypi.nvidia.com/ tensorrt==10.8.0.43
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm==0.16.0
#export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
 