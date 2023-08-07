#!/bin/bash
python train.py +datamodule.num_event_types=60
python train.py +datamodule.num_event_types=50
python train.py +datamodule.num_event_types=40
python train.py +datamodule.num_event_types=30
python train.py +datamodule.num_event_types=20
python train.py +datamodule.num_event_types=10
python train.py +datamodule.num_event_types=5
python train.py +datamodule.num_event_types=1