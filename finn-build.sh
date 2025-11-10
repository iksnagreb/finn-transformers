#!/bin/bash

# Check whether the argument selects one of the available options from the model
# output directory
# shellcheck disable=SC2010
ls outputs | grep -w -q "$1"

# If this is not one of the outputs, the status code $? will indicate failure
if [ $? -eq 1 ]; then
  # shellcheck disable=SC2012
  echo "Select one of the model options: $(ls outputs | tr '\n' ' ')"; exit 1;
fi;

# Base path to the model output to be used as FINN build inputs
path="outputs/$1"

# Run the FINN build command consuming the already streamlined model
finn build -d .finn build.yaml "$path/streamlined.onnx" "${@:2}" \
 --verify-input "$path/inp.npy" --verify-output "$path/out.npy" -o "$path/build"
