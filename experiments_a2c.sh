#!/bin/sh

# Default reward setting
python main.py --algo a2c --model_path ./model_checkpoints/a2c_1 --log_num 0

# Experiments on impact of opponent weighting
python main.py --algo a2c --model_path ./model_checkpoints/a2c_op_5 --reward_type custom --op_wgt 0.5 --log_num 1
python main.py --algo a2c --model_path ./model_checkpoints/a2c_op_1_5 --reward_type custom --op_wgt 1.5 --log_num 2

# Experiments on giving active pokemon more weight (own team and opponent team) 
python main.py --algo a2c --model_path ./model_checkpoints/a2c_act_1_5 --reward_type custom --act_wgt 0.5 --log_num 3

# HP Shift on active pokemon 
python main.py --algo a2c --model_path ./model_checkpoints/a2c_shift_5 --reward_type custom --hp_shift 0.5 --log_num 4
