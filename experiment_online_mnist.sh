#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python train_mnist_online.py --dataset MNIST --noise gaussian --alpha 0.5 --optimizer adam --batch_size 100 --num_steps 1 --meta_steps 5  --temperature 1 --temperature_factor 1.5 --lr 0.00001 --sigma 0.009 --use_conv False --dims 1200
