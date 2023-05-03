#!/bin/sh

# Default reward setting
python main.py --algo ppo --lr 0.0003 --reward_type custom --model_path ./ppo_model_checkpoints_md/ppo_1 --log_num 0

# Experiments on impact of opponent weighting
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_op_5 --reward_type custom --op_wgt 0.5 --log_num 1
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_op_1_5 --reward_type custom --op_wgt 1.5 --log_num 2

# Experiments on giving active pokemon more weight (own team and opponent team) 
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_act_1_5 --reward_type custom --act_wgt 0.5 --log_num 3

# HP Shift on active pokemon 
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_shift_5 --reward_type custom --hp_shift 0.5 --log_num 4

# Best weight
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_best_5 --reward_type custom --op_wgt 1.1 \
--act_wgt 1.1 --status_val 0.5 --log_num 5

# Default reward setting
python main.py --algo ppo --lr 0.0003 --model_path ./ppo_model_checkpoints_md/ppo_6 --log_num 6