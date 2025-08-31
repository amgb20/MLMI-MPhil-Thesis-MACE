#!/bin/bash

export PYTHONPATH="./mace-tools-new_data_no_multihead:./mace"

model='/homes/ab3149/Documents/MLMI-MPhil-Thesis-MACE/Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model'
start_structure='./128_water_box.xyz'

# Create timestamped results directory
timestamp=$(date +"%Y%m%d_%H%M%S")
save_dir="./results/"
mkdir -p "$save_dir"

echo "Starting water density simulation..."
echo "Model: $model"
echo "Structure: $start_structure"
echo "Results will be saved to: $save_dir"

# Run the simulation
python3 run_npt.py \
    --model "$model" \
    --structure "$start_structure" \
    --temp 300 \
    --runtime 100 \
    --default_dtype float64 \
    --layer_default_dtype float64 \
    --label seed1 \
    --run_dir "$save_dir" \

# Check if simulation completed successfully
if [ $? -eq 0 ]; then
    echo "✓ Simulation completed successfully!"
    echo "Results saved to: $save_dir"
else
    echo "✗ Simulation failed with exit code $?"
    echo "Check the log files for errors."
fi
