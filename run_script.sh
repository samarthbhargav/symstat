#!bin/bash

MODELID=1

srun --gres=gpu:1 -t 10:00:00 python symstat.py train --model=sl --model-id=$MODELID --num-workers=4 --unlabeled=0.998 --epochs=100


