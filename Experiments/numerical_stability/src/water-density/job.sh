#!/bin/bash
#SBATCH --job-name=mace-job
#SBATCH --account gax@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6s
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=mace_test_%A.out

module purge
module load arch/h100

module load anaconda-py3/2024.06
conda activate mace_env_mh
export PYTHONPATH="./mace-tools-new_data_no_multihead:./mace"


run_in_folder () {
        export CUDA_VISIBLE_DEVICES=$1
        folder=$2

        mkdir $folder
        cd $folder
        cp $python_script .
        python ./run_npt.py "${@:3}"
}

start_structure=$(pwd)"/128_water_box.xyz"
model=$(pwd)"/seed1_swa.model"
python_script=$(pwd)"/run_npt.py"


run_in_folder 0 "./250K" --model=$model --structure=$start_structure --temp=250 --runtime=500000 --label=seed1 &
run_in_folder 1 "./270K" --model=$model --structure=$start_structure --temp=270 --runtime=500000 --label=seed1 &
run_in_folder 2 "./290K" --model=$model --structure=$start_structure --temp=290 --runtime=500000 --label=seed1 &
run_in_folder 3 "./310K" --model=$model --structure=$start_structure --temp=310 --runtime=500000 --label=seed1 &
run_in_folder 4 "./330K" --model=$model --structure=$start_structure --temp=330 --runtime=500000 --label=seed1 &
wait
