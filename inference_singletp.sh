#!/bin/bash
##############################
# SLURM SETTINGS
##############################
#SBATCH --job-name=SingleTP_inference
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/inference_singletp_%j.out
#SBATCH --error=/home/hpc/users/ml_models/jovana.ristevska/LongiSeg_Project/logs/inference_singletp_%j.error
#SBATCH --nodelist=gpgpu02
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jovana.ristevska@students.finki.ukim.mk

CONTAINER="/home/hpc/users/ml_models/jovana.ristevska/FCDenseNet/images/pytorch-gpu.sif"

echo "Starting inference on node $(hostname)"
echo "Available GPU: $(nvidia-smi -L)"

singularity exec --nv --bind /home/hpc/users/ml_models:/mnt --bind /home/hpc/users/jovana.ristevska:/home/hpc/users/jovana.ristevska $CONTAINER bash -c "export LongiSeg_raw=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw && export LongiSeg_preprocessed=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_preprocessed && export LongiSeg_results=/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results && export PATH=\$PATH:/home/hpc/users/jovana.ristevska/.local/bin && export PYTHONPATH=\$PYTHONPATH:/mnt/jovana.ristevska/LongiSeg_Project/LongiSeg && export OMP_NUM_THREADS=1 && LongiSeg_predict -i /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset002_MSLesionsSingleTP/imagesTs -o /mnt/jovana.ristevska/LongiSeg_Project/predictions_singletp_5folds -pat /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset002_MSLesionsSingleTP/patientsTs.json -d 002 -c 3d_fullres -tr nnUNetTrainerNoLongi -f 0 1 2 3 4 -chk checkpoint_final.pth && LongiSeg_evaluate_folder /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset002_MSLesionsSingleTP/labelsTs /mnt/jovana.ristevska/LongiSeg_Project/predictions_singletp_5folds -djfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results/Dataset002_MSLesionsSingleTP/nnUNetTrainerNoLongi__nnUNetPlans__3d_fullres/dataset.json -pfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_results/Dataset002_MSLesionsSingleTP/nnUNetTrainerNoLongi__nnUNetPlans__3d_fullres/plans.json -patfile /mnt/jovana.ristevska/LongiSeg_Project/LongiSeg_raw/Dataset002_MSLesionsSingleTP/patientsTs.json"