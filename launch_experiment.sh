#!/bin/bash
#SBATCH --job-name=experimentation-gat
#SBATCH --nodes=1
#SBATCH -o logs/experiments_gnn_gat.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=250:00:00
#SBATCH --gres=gpu:1

hostname
source /etc/profile
source /home/manjimnav1/TFM_EEG_SEIZURE_CLASSIFICATION/.venv/bin/activate
python loso_foundation.py --xai-enabled --xai-dir explainability --models chronos2 moirai2 tsmixer
