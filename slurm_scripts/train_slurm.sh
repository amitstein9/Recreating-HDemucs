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
EPOCHS=${1:-390}

echo "Training will run for $EPOCHS epochs, with tests every 100 epochs."

# Launch distributed training with 8 GPUs.
srun --ntasks=8 --gres=gpu:1 --gpu-bind=single:1 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_PROCID;
  export LOCAL_RANK=0;
  echo "Process rank: $SLURM_PROCID, using GPU: $CUDA_VISIBLE_DEVICES";
  export WORLD_SIZE=$SLURM_NTASKS;
  export RANK=$SLURM_PROCID;
  export MASTER_ADDR=$(hostname);
  export MASTER_PORT=29500;
  # Run training with the specified number of epochs and test frequency.
  # Here, we override "epochs" and set "test.every=100".
  dora run -d epochs='"${EPOCHS}"' test.every=100 batch_size=32 hdemucs.norm_starts=100 hdemucs.cac=False test.split=True valid_apply=True model=hdemucs hdemucs.dconv_lstm=4 ema.epoch=[0.9,0.95] ema.batch=[0.9995,0.9999] seed=42 hdemucs.hybrid_old=True svd=base svd.penalty=1e-05 svd.dim=80 svd.convtr=True optim.lr=0.0003
'

