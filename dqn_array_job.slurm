#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --nodes=1
#SBATCH --job-name=dqn_array_job_8cores
#SBATCH --array=1-10
#SBATCH --time=00:60:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=out/dqn_job_%A_%a.txt
#SBATCH --partition=CPUQ
#SBATCH --mail-user=jacob.LLarsen@hotmail.com
#SBATCH --mail-type=ALL

module purge
module load PyTorch/2.0.1-foss-2022a
module list

python -m src.DRL.train_qrunner