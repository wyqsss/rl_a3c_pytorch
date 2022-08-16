# RCMP
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 2 4 --n_heads 5 --sigma 0.03 --amsgrad True > run_logs/new_rcmp_sig.log & 
# no advice 
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --budget 0 --amsgrad True > run_logs/new_no_advice.log &
# random
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --rand_advice True --amsgrad True > run_logs/new_random.log &
# importance
# nohup python main.py --env Pong-v0 --workers 24 --gpu-ids 0 1 --amsgrad True > run_logs/new_importance.log &
# kill -9 `ps -ef |grep Agent|awk '{print $2}' `