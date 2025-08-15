#!/bin/bash

# If specified via env-var, execute commands in a job scheduled via SLURM
if [ "$RUN_VIA_SLURM" = 1 ]; then
  # Specify a name for the SLURM job
  JOB_NAME="-J ${JOB_NAME:=DVC}"
  # Hard time limit of the job, will forcefully be terminated if not done within
  # this time
  TIME_LIMIT="-t ${TIME_LIMIT:=24:00:00}"
  # Number of CPUs to use per task
  NUM_CPUS="--cpus-per-task=${NUM_CPUS:=16}"
  # Amount of memory to allocate for the job
  MEM="--mem ${MEM:=64G}"
  # The partition to which the job is submitted
  PARTITION="-p ${PARTITION:=normal}"
  # Notify by mail on all events (queue, start, stop, fail, ...)
  MAIL="--mail-type FAIL --mail-user ${MAIL:=}"
  # If using GPUS, specify which type of GPU and how many
  #   Note: Hardcode this to 1, there is no need for more GPUs right now
  if [[ "$PARTITION" = "-p gpu" ]] || [[ "$PARTITION" = "-p dgx" ]]; then
    #   Note: Hardcode this to 1, there is no need for more GPUs right now
    GPUS="--gres=gpu:${GPUS:=a100:1}"
  fi;
  # Group all sbatch command line arguments into one string
  ARGS="$JOB_NAME $TIME_LIMIT $NUM_CPUS $MEM $PARTITION $MAIL $GPUS"
  # Forward all arguments following the shell script to be executed as the
  # command line of job scheduled via SLURM
  # shellcheck disable=SC2086
  srun $ARGS --verbose "$@"
# By default, execute the job locally
else
  # Forward all arguments following the shell script to be executed as the
  # command line within this script
  eval "$@";
fi;
