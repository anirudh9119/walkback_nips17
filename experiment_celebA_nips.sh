#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
#python train_nips_celebA.py --dataset celeba --noise gaussian --alpha 0.5 --optimizer adam --batch_size 100 --num_steps 1 --meta_steps 30 --temperature 1.0 --temperature_factor 2 --lr 0.0001 --sigma 0.00001
python train_celebA.py --dataset celeba --noise gaussian --alpha 0.5 --optimizer adam --batch_size 100 --num_steps 1 --meta_steps 30 --temperature 1.0 --temperature_factor 2 --lr 0.0001 --sigma 0.00001 --reload_ True --saveto_filename /data/lisatmp3/anirudhg/celebA_walkback/walkback_-170506T155340/params_batch_index11500.npz --extra_steps 10000
