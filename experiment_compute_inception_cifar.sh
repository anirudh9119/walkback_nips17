#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

python compute_inception_cifar.py --dataset CIFAR10 --noise gaussian --alpha 0.5 --optimizer adam --batch_size 100 --dim 512 --num_steps 1 --meta_steps 30 --temperature 1 --temperature_factor 2.0 --lr 0.0001 --reload_ True  --saveto_filename /data/lisatmp4/anirudhg/cifar_walk_back/walkback_-170730T044514/params_6000.npz
#python compute_inception_cifar.py --dataset CIFAR10 --noise gaussian --alpha 0.5 --optimizer adam --batch_size 100 --dim 512 --num_steps 1 --meta_steps 30 --temperature 1 --temperature_factor 2.0 --lr 0.0001 --reload_ True  --saveto_filename /data/lisatmp4/anirudhg/cifar_walk_back/walkback_-170726T043234/params_20.npz
