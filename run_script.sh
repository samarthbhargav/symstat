#!bin/bash

MODELID=1

python symstat.py train --model=sl --model-id=$MODELID --num-workers=4 --unlabeled=0.998 --batch-size=128  --epochs=100
