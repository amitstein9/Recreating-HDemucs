#!/bin/bash
#SBATCH --job-name=DemucsRelease
#SBATCH --output=./Demucs_Release.out   # redirect stdout
#SBATCH --error=./Demucs_Release.err    # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --nodes=1
#SBATCH --mem=64000

# Get the model name parameter from the command-line.
MODEL_ID=${1:-53288a8c}

# Export the model.
python3 -m tools.export $MODEL_ID
