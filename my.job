#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/DepressionDetection-cv-2-balanced

eval "$(conda shell.bash hook)"

conda activate mm

cd /home1/tereeves/mm/DepressionDetection
python train.py