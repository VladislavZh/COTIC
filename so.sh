#!/bin/bash
python train.py +datamodule.dataset_size_train=500 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=1000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=2000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=3000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=5000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=7000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=10000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=15000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=20000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=25000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
python train.py +datamodule.dataset_size_train=30000 +datamodule.dataset_size_val=1 +datamodule.dataset_size_test=1
