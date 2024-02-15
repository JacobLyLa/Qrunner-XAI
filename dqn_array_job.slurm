#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --nodes=1
#SBATCH --job-name=dqn_array_job
#SBATCH --array=1-10
#SBATCH --time=00:120:00
#SBATCH --mem=4G
#SBATCH --output=out/dqn_job_%A_%a.txt
#SBATCH --partition=CPUQ
#SBATCH --mail-user=jacob.LLarsen@hotmail.com
#SBATCH --mail-type=ALL

module purge
module load PyTorch/2.0.1-foss-2022a
module list

python -m src.DRL.train_qrunner