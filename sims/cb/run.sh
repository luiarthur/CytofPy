#!/bin/bash

# Results directory
RESULTS_DIR=$1
# AWS Bucket to store results
AWS_BUCKET=$2

# MAX NUMBER OF CORES TO USE
MAX_CORES=10

# RANDOM SEEDS TO USE IN RUNS
SEEDS=`seq 10`

# COUNTER FOR PARALLEL JOBS
counter=0
for seed in $SEEDS; do
  # INCREMENT COUNTER
  ((counter++))

  # MAIN
  EXP_DIR=$RESULTS_DIR/$seed/
  mkdir -p $EXP_DIR
  python3 cb.py $EXP_DIR $seed > $EXP_DIR/log.txt &

  # Wait for jobs to finish every $MAX_CORES iterations.
  if (( $counter % $MAX_CORES == 0 )); then wait; fi 
done
