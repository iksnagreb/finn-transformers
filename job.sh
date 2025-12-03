#!/bin/bash

# Setup the python environment on the PC² cluster
module load lang/Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate

# Forward all arguments following the shell script to be executed as the
# command line within this script
eval "$@"
