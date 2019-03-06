#!/bin/bash

# COUNTER FOR PARALLEL JOBS. Must be 0!
MY_COUNTER_112358=0

function engine() {
  local RESULTS_DIR=$1
  local AWS_BUCKET=$2
  local EXP_NAME=$3
  local cmd=$4
  local MAX_CORES=$5

  local OUTDIR=$RESULTS_DIR/$EXP_NAME

  ((MY_COUNTER_112358++))

  # Sync results to S3
  syncToS3="aws s3 sync $OUTDIR $AWS_BUCKET/$EXP_NAME --exclude '*.nfs*'"

  # Remove output files to save space on cluster
  rmOutput="rm -rf ${OUTDIR}"

  syncToS3="echo"
  rmOutput="echo"

  bundle_cmds="$cmd > $OUTDIR/log.txt && $syncToS3 && $rmOutput"
  echo "$bundle_cmds"
  # eval "$bundle_cmds &"
  sem -j $MAX_CORES $bundle_cmds

  # if (( $MY_COUNTER_112358 % $MAX_CORES == 0 )); then wait; fi 
}
