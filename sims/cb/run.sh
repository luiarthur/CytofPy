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
# STAGGER=100
STAGGER=0

# RANDOM SEEDS TO USE IN RUNS
SEEDS=`seq -w 10`

for seed in $SEEDS; do
  # MAIN
  EXP_NAME=$seed
  EXP_DIR=$RESULTS_DIR/$EXP_NAME
  mkdir -p $EXP_DIR
  # cmd="python3 cb.py $EXP_DIR $seed"
  cmd="echo Hi, $seed!"

  engine $RESULTS_DIR $AWS_BUCKET $EXP_NAME "$cmd" $MAX_CORES

  time_at_next_run=`date -d "+$STAGGER sec"`
  echo "Next run will start at $time_at_next_run"
  sleep $STAGGER
done
