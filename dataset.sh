#!/bin/sh
#SBATCH -c 4 
#SBATCH --gres=gpu:1 
#SBATCH -p gpu
#SBATCH --time=00:30:00

source /home/hanfeld/.front-env/bin/activate
python src/attacks.py --file $1
