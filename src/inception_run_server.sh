#!/usr/bin/env bash

python3 InceptionV3_train.py --batch_size=64 --epochs=4 --output_dir='/data/TaskA_2018/src/check_point' --lr=0.1 --log_interval=10 --alpha=0.2

