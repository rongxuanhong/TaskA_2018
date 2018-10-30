#!/usr/bin/env bash

python3 Xception_train.py --batch_size=100 --epochs=10 --output_dir='/data/TaskA_2018/src/check_point' --lr=0.01 --log_interval=10 --alpha=0.2

