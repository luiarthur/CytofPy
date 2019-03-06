#!/bin/bash

# Source a utility function
source engine.sh

# Results directory
RESULTS_DIR=$1

# AWS Bucket to store results
AWS_BUCKET=$2

# MAX NUMBER OF CORES TO USE
MAX_CORES=32

# STAGGER TIME BETWEEN EXPERIMENTS
STAGGER=100

# RANDOM SEEDS TO USE IN RUNS
SEEDS=`seq -w 10`


### MAIN ###
for seed in $SEEDS; do
  # NAME OF EXPERIMENT
  EXP_NAME=$seed

  # DIRECTORY FOR EXPERIMENT RESULTS
  EXP_DIR=$RESULTS_DIR/$EXP_NAME/
  mkdir -p $EXP_DIR

  # MAIN COMMAND
  cmd="python3 cb.py $EXP_DIR $seed"

  # FOR DEBUGGING
  # cmd="sleep 10 && echo Hi, $seed!"

  engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$cmd" $MAX_CORES $STAGGER
done
