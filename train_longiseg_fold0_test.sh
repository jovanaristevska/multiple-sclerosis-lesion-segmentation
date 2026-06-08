#!/bin/bash
##############################
# SLURM SETTINGS
##############################
#SBATCH --job-name=LongiSeg_fold0_test
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/train_%j.out
#SBATCH --error=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/train_%j.error
#SBATCH --nodelist=gpgpu02
#SBATCH --mail-type=FAIL,ARRAY_TASKS,ALL
#SBATCH --mail-user=jovana.ristevska@students.finki.ukim.mk

##############################
# PATHS AND ENVIRONMENT
##############################
CONTAINER="/home/hpc/users/ml_models/jovana.ristevska/FCDenseNet/images/pytorch-gpu.sif"

echo "Започнувам на нод $(hostname)"
echo "Достапни GPU: $(nvidia-smi -L)"

##############################
# TRAINING COMMAND
##############################
singularity exec --nv \
    --bind /home/hpc/users/ml_models:/mnt \
    --bind /home/hpc/users/jovana.ristevska/.local:/home/hpc/users/jovana.ristevska/.local \
    --bind /usr/bin/gcc:/usr/bin/gcc \
    --bind /usr/lib/gcc:/usr/lib/gcc \
    $CONTAINER \
    bash -c "
        export LongiSeg_raw=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw
        export LongiSeg_preprocessed=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_preprocessed
        export LongiSeg_results=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results
        export PATH=\$PATH:/home/hpc/users/jovana.ristevska/.local/bin
        export PYTHONPATH=\$PYTHONPATH:/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg
        export LongiSeg_compile=0
        export nnUNet_n_proc_DA=0
        export OMP_NUM_THREADS=1
        export PYTHONUNBUFFERED=1


        echo 'PyTorch version:'
        python3 -c 'import torch; print(torch.__version__); print(\"CUDA:\", torch.cuda.is_available())'

        echo 'Starting fold 0 training...'
        python3 -u /home/hpc/users/jovana.ristevska/.local/bin/LongiSeg_train 001 3d_fullres 0 --npz    "
