#!/usr/bin/env bash

python3 DCNN_train_keras.py --batch_size=100 --epochs=5 --output_dir='/data/TaskA_2018/src/check_point' --lr=0.01 --log_interval=10 --alpha=0.2

