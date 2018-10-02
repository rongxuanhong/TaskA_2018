#!/usr/bin/env bash
python3 Estimator_DenseNet_train.py --batch_size=10 --epochs=1 --nb_layers=4 --n_db=2 --grow_rate=12 \
--data_format='channels_last' --output_dir='/data/TaskA_2018/src/check_point' --lr=0.1 --log_interval=10 --alpha=0.2
