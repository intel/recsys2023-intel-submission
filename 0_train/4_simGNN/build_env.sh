yes | conda create -n rec23_env python=3.8 -c anaconda

eval "$(conda shell.bash hook)"

conda activate rec23_env

yes | conda install pip cmake
yes | conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
yes | conda install -c dglteam/label/cu117 dgl
yes | conda install -c conda-forge psutil
yes | conda install -c conda-forge tqdm
yes | conda install -c conda-forge ogb
yes | conda install -c conda-forge numpy
yes | conda install -c conda-forge scikit-learn
yes | conda install -c conda-forge pydantic
yes | conda install -c conda-forge pyyaml
yes | conda install -c conda-forge chardet
yes | conda install -c conda-forge torchmetrics
yes | conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021
pip install lightgbm
