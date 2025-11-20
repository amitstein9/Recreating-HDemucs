#!/bin/bash
#SBATCH --job-name=DemucsTest
#SBATCH --output=slurm_out/DemucsTest.out
#SBATCH --error=slurm_out/DemucsTest.err
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1         # Only 1 GPU is needed for testing
#SBATCH --ntasks=1
#SBATCH --mem=46000
###SBATCH --constraint=geforce_rtx_3090

# Add the soundstretch binary to PATH (if needed – adjust as necessary).
export PATH="$(pwd)/soundtouch/build:$PATH"

# Get the model name parameter.
MODEL_ID=${1:-EndModel}

# Evaluate the model.
python3 -m tools.test_pretrained --repo ./release_models -n 759c9a58 #$MODEL_ID
