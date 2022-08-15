from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging


def test(args, shared_model, env_conf, epochs):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space, args.n_heads)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()

    max_score = -22
    while True:

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        ep = epochs.value
        # log['{}_log'.format(args.env)].info(f"ep is:  {epochs.value}")
        left_advice = args.budget.value
        player.model.eval()

        
        roll_rewards_sum = 0
        roll_eps_len = 0
        for i in range(20):
            while True:
                player.action_test()
                # log['{}_log'.format(args.env)].info(f"play action, info is {player.info}")
                reward_sum += player.reward

                if player.done and not player.info:
                    state = player.env.reset()
                    player.eps_len += 2
                    player.state = torch.from_numpy(state).float()
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.state = player.state.cuda()
                elif player.info:
                    # num_tests += 1
                    # reward_total_sum += reward_sum
                    roll_rewards_sum += reward_sum
                    roll_eps_len += player.eps_len
                    # reward_mean = reward_total_sum / num_tests
                    # log['{}_log'.format(args.env)].info(
                    #     "Time {0}, epoch {4}, episode reward {1}, episode length {2}, reward mean {3:.4f}, left advice {5}".
                    #     format(
                    #         time.strftime("%Hh %Mm %Ss",
                    #                     time.gmtime(time.time() - start_time)),
                    #         reward_sum, player.eps_len, reward_mean, ep, left_advice))

                    # if args.save_max and reward_sum >= max_score:
                    #     max_score = reward_sum
                    #     if gpu_id >= 0:
                    #         with torch.cuda.device(gpu_id):
                    #             state_to_save = player.model.state_dict()
                    #             torch.save(state_to_save, '{0}{1}.dat'.format(
                    #                 args.save_model_dir, args.env))
                    #     else:
                    #         state_to_save = player.model
                    #         torch.save(state_to_save, '{0}{1}.dat'.format(
                    #             args.save_model_dir, args.env))

                    reward_sum = 0
                    player.eps_len = 0
                    state = player.env.reset()
                    player.eps_len += 2
                    # time.sleep(10)
                    player.state = torch.from_numpy(state).float()
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.state = player.state.cuda()
                    break
        num_tests += 1  
        reward_total_sum += roll_rewards_sum / 20
        reward_mean = reward_total_sum / num_tests      
        log['{}_log'.format(args.env)].info(
                    "Time {0}, epoch {4}, episode avg_reward {1}, episode avg_length {2}, reward mean {3:.4f}, left advice {5}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                    time.gmtime(time.time() - start_time)),
                        roll_rewards_sum / 20, roll_eps_len / 20, reward_mean, ep, left_advice))
        if args.save_max and  (roll_rewards_sum / 20) >= max_score:
                    max_score =  roll_rewards_sum / 20
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}.dat'.format(
                                args.save_model_dir, args.env))
                    else:
                        state_to_save = player.model
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))
        # time.sleep(10)
