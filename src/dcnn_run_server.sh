#!/usr/bin/env bash

python3 DCNN_train.py --batch_size=60 --epochs=50 --output_dir='/data/TaskA_2018/src/check_point' --lr=0.001 --log_interval=10 --alpha=0.2

