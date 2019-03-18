#!/bin/bash

# sem: requires gnu-paraller

# FUNCTION TO SCHEDULE RUNS
function engine() {
  local RESULTS_DIR=$1
  local AWS_BUCKET=$2
  local EXP_NAME=$3
  local cmd=$4
  local MAX_CORES=$5
  local STAGGER=$6

  # Output directory for experiment
  local EXP_DIR=$RESULTS_DIR/$EXP_NAME

  # Sync results to S3
  syncToS3="aws s3 sync $EXP_DIR $AWS_BUCKET/$EXP_NAME --exclude '*.nfs*'"

  # Remove output files to save space on cluster
  rmOutput="rm -rf ${EXP_DIR}"

  # FOR DEBUGGING
  # syncToS3="echo"
  # rmOutput="echo"

  # BUNDLE OF COMMANDS TO EXECUTE
  bundle_cmds="$cmd > $EXP_DIR/log.txt && sleep 2 && $syncToS3 && sleep 2 && $rmOutput"
  echo "Next job: $bundle_cmds"
  sem -j $MAX_CORES $bundle_cmds

  # STAGGER EXPERIMENTS TO NOT MEGA DUMP AT END OF EXPERIMENT
  time_at_next_run=`date -d "+$STAGGER sec"`
  echo "Next job will start after $time_at_next_run"
  sleep $STAGGER
}
