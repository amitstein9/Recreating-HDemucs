#!/bin/bash
#SBATCH --job-name=DemucsTrain
#SBATCH --output=slurm_out/Demucs_Train.out
#SBATCH --error=slurm_out/Demucs_Train.err
#SBATCH --partition=killable         # In Real Train: killable     
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem=50000          # 50 GB of CPU memory
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --account=gpu-research

# Activate the conda environment.
cd /home/yandex/APDL2425a/group_9/demucs/ #TODO: Change this to your path

# Set NCCL debugging and memory options for multi-GPU training.
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1

# Add the soundstretch directory to PATH.
# Adjust this relative path to where your "soundstretch" binary is built.
export PATH="$(pwd)/soundtouch/build:$PATH"

# Get the number of epochs from the first parameter (default is 390).
MODEL_ID=$1

# Launch distributed training with 8 GPUs.
srun --ntasks=8 --gres=gpu:1 --gpu-bind=single:1 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_PROCID;
  export LOCAL_RANK=0;
  echo "Process rank: $SLURM_PROCID, using GPU: $CUDA_VISIBLE_DEVICES";
  export WORLD_SIZE=$SLURM_NTASKS;
  export RANK=$SLURM_PROCID;
  export MASTER_ADDR=$(hostname);
  export MASTER_PORT=29500;
  dora run -d -f $MODEL_ID
'