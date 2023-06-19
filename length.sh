#!/bin/bash
python train.py +datamodule.max_len=50
python train.py +datamodule.max_len=100
python train.py +datamodule.max_len=200
python train.py +datamodule.max_len=350
python train.py +datamodule.max_len=500
python train.py +datamodule.max_len=700
python train.py +datamodule.max_len=1000
python train.py +datamodule.max_len=1500
python train.py +datamodule.max_len=2000
python train.py +datamodule.max_len=3000