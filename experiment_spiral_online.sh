#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python train_spiral_online.py --dataset Spiral --noise gaussian --alpha 0.5 --optimizer adam --batch_size 5000 --num_steps 30 --meta_steps 1 --temperature 1 --temperature_factor 1.1 --lr 0.0001 --sigma 0.009


#python train_spiral_online.py --dataset Spiral --noise gaussian --alpha 0.5 --optimizer adam --batch_size 5000 --num_steps 10 --temperature 1 --temperature_factor 1.1 --lr 0.0001 --sigma 0.05

#python train_spiral_online.py --dataset Spiral --noise gaussian --alpha 0.5 --optimizer adam --batch_size 5000 --num_steps 30 --temperature 1 --temperature_factor 1.1 --lr 0.0001 --sigma 0.009
