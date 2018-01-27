#!/bin/bash
for config in {1..30}
do
if [ -f "/data/chercheurs/agarwals/DEX/gauss/checkpoints/$config" ]
then
    echo "Config number $config already exists...moving to next config"
else
    touch /data/chercheurs/agarwals/DEX/gauss/checkpoints/$config
    CUDA_VISIBLE_DEVICES=$1 python3 train.py --config_path=./configuration.ini --config_number="$config"
fi
done
