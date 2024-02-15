#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --nodes=1
#SBATCH --job-name=dqn_job16
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=out/dqn_job_%A_%a.txt
#SBATCH --partition=CPUQ
#SBATCH --mail-user=jacob.LLarsen@hotmail.com
#SBATCH --mail-type=ALL

module purge
module load PyTorch/1.13.1-foss-2022b
module list

python -m cProfile -s cumtime -o program.prof -m src.DRL.train_qrunner