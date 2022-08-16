# RCMP
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 2 3 --n_heads 5 --sigma 0.08 --amsgrad True > run_logs/new_rcmp_sig.log & 
# no advice 
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --budget 0 --amsgrad True > run_logs/new_no_advice.log &
# random
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --rand_advice True --amsgrad True > run_logs/new_random.log &
# importance
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --amsgrad True > run_logs/new_importance.log &
# kill -9 `ps -ef |grep Agent|awk '{print $2}' `


# nohup python main.py --env SpaceInvaders-v0 --workers 32 --gpu-ids 0 1 2 3 --budget 0 --amsgrad True > run_logs/SpaceInvaders_noadvice.log &
# RCMP
# nohup python main.py --env SpaceInvaders-v0 --workers 32 --gpu-ids 0 1 2 3 --demo trained_models/SpaceInvaders-v0_demo.dat --n_heads 5 --amsgrad True > run_logs/SpaceInvaders_rcmp.log &