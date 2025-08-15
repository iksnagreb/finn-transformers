#!/bin/bash

# Setup the python environment on the PC² cluster
module load lang/Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

# Setup the FPGA development environment
module load fpga
module load xilinx/xrt/2.14
module load xilinx/vitis/22.2

# Somehow these options are required to get FINN running on the cluster...
export LC_ALL="C"
export PYTHONUNBUFFERED=1
export XILINX_LOCAL_USER_DATA="no"

# Forward all arguments following the shell script to be executed as the
# command line within this script
eval "$@"
