# nohup python main.py --env Pong-v0 --workers 40 --gpu-ids 1 2 --n_heads 5 --amsgrad True > run_logs/rcmp_avg_ploss.log &
# kill -9 `ps -ef |grep Agent|awk '{print $2}' `