from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable


def train(rank, args, shared_model, optimizer, env_conf, epochs):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    demo = A3Clstm(env.observation_space.shape[0], env.action_space)
    demo.load_state_dict(torch.load("trained_models/demo.dat", map_location="cpu"))
    demo.eval()
    player = Agent(None, env, args, None, demo=demo)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space, args.n_heads)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
            player.demonstration = player.demonstration.cuda()
    player.model.train()
    player.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(args.n_heads, 1) if args.n_heads > 1 else torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            if args.n_heads > 1:
                for i in range(args.n_heads):
                    R[i] = value[i].data[0]
            else:
                R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        if args.n_heads > 1:
            value_loss = [0 for i in range(args.n_heads)]
        else:
            value_loss = 0
        policy_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        if args.n_heads > 1:
            for i in reversed(range(len(player.rewards))):
                avg_1 = 0
                avg_ = 0
                for k in range(args.n_heads):
                    R[k] = args.gamma * R[k] + player.rewards[i]
                    advantage = R[k] - player.values[i][k]
                    value_loss[k] = value_loss[k] + 0.5 * advantage.pow(2)
                    avg_1 += player.values[i + 1][k].data
                    avg_ += player.values[i][k].data

                avg_ = avg_ / args.n_heads
                avg_1 = avg_1 / args.n_heads
                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + args.gamma * avg_1 - avg_
                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    player.log_probs[i] * \
                    Variable(gae) - 0.01 * player.entropies[i]
            value_loss = sum(value_loss) / args.n_heads
            player.model.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()
            with epochs.get_lock(): epochs.value += 1
            player.clear_actions()
        else:
            for i in reversed(range(len(player.rewards))): # 返回的时逆序列
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]  # R(t) - V(t)
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + args.gamma * \
                    player.values[i + 1].data - player.values[i].data  # r + V(s') - V(s)  advantage V^

                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    player.log_probs[i] * \
                    Variable(gae) - 0.01 * player.entropies[i]

            player.model.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()
            with epochs.get_lock(): epochs.value += 1
            player.clear_actions()
