#!/bin/bash

eval "$(conda shell.bash hook)"
MINIFORGE_DIR=${CONDA_EXE%/*/*}
source "$MINIFORGE_DIR/etc/profile.d/mamba.sh"

mamba create -n NeuroGraph python=3.8 -y
mamba activate NeuroGraph

mamba install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch -y
mamba env update -f environment.yml
