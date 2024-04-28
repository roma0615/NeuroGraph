#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
MINIFORGE_DIR=${CONDA_EXE%/*/*}
source "$MINIFORGE_DIR/etc/profile.d/mamba.sh"

# run this on a host w GPU
# need to install this and NeuroGraph deps

# OLD
# mamba create -n mace python=3.10 -y
# mamba activate mace

# already run
# mamba create -n mace_ng python=3.10 -y
mamba activate mace_ng
# mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# mamba install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y


mamba install pip numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn boto3 networkx
# mamba install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn -y # OLD

pip install torch_geometric
pip install nibabel nilearn
pip install .

pip install ./mace/mace-layer

# optional
mamba install ipykernel ipywidgets
