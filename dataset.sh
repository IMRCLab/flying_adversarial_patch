#!/bin/sh
#SBATCH -c 6 
#SBATCH --gres=gpu:1 
#SBATCH -p gpu
#SBATCH --time=02:00:00

source /home/hanfeld/.yolopatches/bin/activate
python src/attacks.py --file $1 --model $2
