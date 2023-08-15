#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --job-name=cpu_test_model

cd $TMPDIR

#rm /scratch/s5397774/SSE-gpu/output/*
#mkdir output
tar -xvf /scratch/s5397774/SSE-modified/librispeech_selection.tar.gz -C $TMPDIR
#tar -xvf /scratch/s5397774/SSE-gpu/urbansound16k.tar.gz -C $TMPDIR
tar -xvf /scratch/s5397774/SSE-modified/gymnoise_testing.tar.gz -C $TMPDIR

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


source /scratch/s5397774/.env/bin/activate


pip install --upgrade pip
pip install --upgrade pip setuptools wheel
pip install -r /scratch/s5397774/SSE-modified/requirements.txt
wandb login

python -u /scratch/s5397774/SSE-modified/test.py 

deactivate

