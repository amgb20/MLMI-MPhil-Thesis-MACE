#!/bin/bash

# Configuration
TEMP_START=250
TEMP_STEP=10
TEMP_END=330
RUNTIME=1000000  # Simulation runtime in steps
ROOT_DIR=$(pwd)  # Base directory for runs
RUN_NAME="simulation_run"  # Custom run name prefix

# Calculate the number of temperature steps
NUM_JOBS=$(( (TEMP_END - TEMP_START) / TEMP_STEP + 1 ))

# Create necessary directories
mkdir -p logs

# Create temperature directories with custom names
for ((i = 0; i < NUM_JOBS; i++)); do
    TEMP=$((TEMP_START + i * TEMP_STEP))
    FOLDER_NAME="${RUN_NAME}_${TEMP}K"
    mkdir -p "$ROOT_DIR/$FOLDER_NAME"
    echo "Created directory: $ROOT_DIR/$FOLDER_NAME"
done

# Submit the job array with exported variables
sbatch --array=0-$((NUM_JOBS - 1)) \
       --export=ALL,TEMP_START=$TEMP_START,TEMP_STEP=$TEMP_STEP,RUNTIME=$RUNTIME,ROOT_DIR=$ROOT_DIR,RUN_NAME=$RUN_NAME \
       slurm_job_array.sh

echo "Submitted job array with $NUM_JOBS tasks, runtime set to $RUNTIME steps."

