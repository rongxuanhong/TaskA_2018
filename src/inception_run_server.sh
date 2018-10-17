#!/usr/bin/env bash

python3 InceptionV3_train.py --batch_size=64 --epochs=15 --output_dir='/data/TaskA_2018/src/check_point' --lr=0.1 --log_interval=20 --alpha=0.2

