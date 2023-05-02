#!/bin/sh

# # Default Poke RL
# python main.py --algo dqn --learning_starts 1000 --lr 0.0001 \
# --gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_1 --reward_type custom --log_num 0 

# # # Experiments on impact of opponent weighting
# python main.py --algo dqn --learning_starts 1000 --lr 0.0001 --gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_op_5 --reward_type custom --op_wgt 0.5 --log_num 1
# python main.py --algo dqn --learning_starts 1000 --lr 0.0001 --gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_op_1_5 --reward_type custom --op_wgt 1.5 --log_num 2

# # Experiments on giving active pokemon more weight (own team and opponent team) 
python main.py --algo dqn --learning_starts 1000 --lr 0.0001 --gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_act_1_5 --reward_type custom --act_wgt 0.5 --log_num 3

# # HP Shift on active pokemon 
python main.py --algo dqn --learning_starts 1000 --lr 0.0001 --gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_shift_5 --reward_type custom --hp_shift 0.5 --log_num 4

# Best weight
python main.py --algo dqn --learning_starts 1000 --lr 0.0001 \
--gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_5 --reward_type custom --op_wgt 1.1 \
--act_wgt 1.1 --status_val 0.5 --log_num 5

# Default reward setting
python main.py --algo dqn --learning_starts 1000 --lr 0.0001 \
--gamma 0.99 --model_path ./dqn_model_checkpoints_sh/dqn_6 --log_num 6
