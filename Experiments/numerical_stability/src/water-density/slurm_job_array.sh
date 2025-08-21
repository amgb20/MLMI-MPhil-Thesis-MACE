#!/bin/bash
#SBATCH --job-name=mace-job
#SBATCH --account=gax@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6s
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mace_test_%A_%a.out

module purge
module load arch/h100
module load anaconda-py3/2024.06

conda activate mace_env_mh
export PYTHONPATH="/lustre/fswork/projects/rech/gax/usj67fz/water_sim/mace-tools-develop/:/lustre/fswork/projects/rech/gax/usj67fz/mace_field_scf/mace/"

# Configuration from environment variables
TEMP_START=${TEMP_START:?Error: TEMP_START not set}
TEMP_STEP=${TEMP_STEP:?Error: TEMP_STEP not set}
RUNTIME=${RUNTIME:?Error: RUNTIME not set}
ROOT_DIR=${ROOT_DIR:?Error: ROOT_DIR not set}
RUN_NAME=${RUN_NAME:?Error: RUN_NAME not set}

# Calculate the current temperature
TEMP=$((TEMP_START + SLURM_ARRAY_TASK_ID * TEMP_STEP))

# Ensure the temperature is within range
if [ $TEMP -gt 330 ]; then
    echo "Temperature $TEMP exceeds the defined range. Exiting."
    exit 1
fi

echo "Running job for temperature: ${TEMP}K with runtime: ${RUNTIME} steps"

START_STRUCTURE="$ROOT_DIR/128_water_box.xyz"
MODEL="$ROOT_DIR/seed1_swa.model"
PYTHON_SCRIPT="$ROOT_DIR/run_npt.py"

# Change to the temperature directory
FOLDER="$ROOT_DIR/${RUN_NAME}_${TEMP}K"
cd "$FOLDER" || { echo "Folder $FOLDER does not exist. Exiting."; exit 1; }

# Run the Python script
python "$PYTHON_SCRIPT" \
    --model="$MODEL" \
    --structure="$START_STRUCTURE" \
    --temp="$TEMP" \
    --runtime="$RUNTIME" \
    --label=seed1
