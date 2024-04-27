#!/bin/bash

eval "$(conda shell.bash hook)"
MINIFORGE_DIR=${CONDA_EXE%/*/*}
source "$MINIFORGE_DIR/etc/profile.d/mamba.sh"

# mamba create -n NeuroGraph python=3.8 -y
mamba activate mace

mamba env update -f environment.yml
