#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --mem=10GB
#SBATCH --job-name=gpu_modified_test_model


base_dir="/scratch/s5397774/SSE-modified/"

cd $TMPDIR

tar -xvf "${base_dir}librispeech_selection.tar.gz" -C $TMPDIR
tar -xvf "${base_dir}urbansound16k.tar.gz" -C $TMPDIR

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# activate virtual environment
source "${base_dir}.env/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install --upgrade pip setuptools wheel
pip install -r "${base_dir}requirements.txt"

wandb login

# train the clean autoencoder (A)
python -u "${base_dir}test.py" --base_dir "$base_dir" --urban_noise False

deactivate

