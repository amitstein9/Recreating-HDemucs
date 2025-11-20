#!/bin/bash
#SBATCH --job-name=DemucsSeparate
#SBATCH --output=./DemucsSeparate.out   # redirect stdout
#SBATCH --error=./DemucsSeparate.err    # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --gpus=1

FILE_PATH="$1"
MODEL_ID="$2"

# Separate the input file using the provided model.
demucs --repo ./release_models -n $MODEL_ID $FILE_PATH
