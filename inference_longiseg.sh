#!/bin/bash
##############################
# SLURM SETTINGS
##############################
#SBATCH --job-name=LongiSeg_inference
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/inference_%j.out
#SBATCH --error=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/inference_%j.error
#SBATCH --nodelist=gpgpu02
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jovana.ristevska@students.finki.ukim.mk

##############################
# PATHS
##############################
CONTAINER="/home/hpc/users/ml_models/jovana.ristevska/FCDenseNet/images/pytorch-gpu.sif"
PROJECT="/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project"

echo "Starting inference on node $(hostname)"
echo "Available GPU: $(nvidia-smi -L)"

##############################
# INFERENCE + EVALUATION
##############################
singularity exec --nv \
    --bind /home/hpc/users/ml_models:/mnt \
    --bind /home/hpc/users/jovana.ristevska/.local:/home/hpc/users/jovana.ristevska/.local \
    $CONTAINER \
    bash -c "
        export LongiSeg_raw=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw
        export LongiSeg_preprocessed=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_preprocessed
        export LongiSeg_results=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results
        export PATH=\$PATH:/home/hpc/users/jovana.ristevska/.local/bin
        export PYTHONPATH=\$PYTHONPATH:/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg
        export OMP_NUM_THREADS=1

        echo '========================================'
        echo 'STEP 1: Running inference on test patients'
        echo '========================================'
        LongiSeg_predict \
            -i /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset001_MSLesions/imagesTs \
            -o /mnt/jovana.ristevska/LongiSeg_Project/predictions_allfolds \
            -pat /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset001_MSLesions/patientsTs.json \
            -d 001 \
            -c 3d_fullres \
            -f 0 1 2 3 4 \
            -chk checkpoint_final.pth

        echo '========================================'
        echo 'STEP 2: Evaluating predictions'
        echo '========================================'
        LongiSeg_evaluate_folder \
            /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset001_MSLesions/labelsTs \
            /mnt/jovana.ristevska/LongiSeg_Project/predictions_allfolds  \
            -djfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results/Dataset001_MSLesions/LongiSegTrainer__nnUNetPlans__3d_fullres/dataset.json \
            -pfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results/Dataset001_MSLesions/LongiSegTrainer__nnUNetPlans__3d_fullres/plans.json \
            -patfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset001_MSLesions/patientsTs.json

        echo '========================================'
        echo 'DONE! Check predictions_allfolds/longi_summary.json for results'
        echo '========================================'
    "
