from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random

class Agent(object):
    def __init__(self, model, env, args, state, demo=None):
        self.model = model
        self.demonstration = demo
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.sigma = self.args.sigma
        self.budget = self.args.budget
        self.im_sigma = self.args.im_sigma
        self.gpu_id = -1

    def action_train(self):
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))   # 是否应该受demonstration的action 影响？
        # print(f"action is {action[0][0].cpu().numpy()}")
        if self.demonstration and self.budget.value > 0:
            if self.args.n_heads > 1:
                uncertainty = torch.var(torch.tensor(value))
                # print(f"uncertain is {uncertainty}")
                # print(f"budget is {self.budget.value}, device is {next(self.model.parameters()).device}")
                if uncertainty > self.sigma:
                    _, qs, _ = self.demonstration((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
                    qs = F.softmax(qs, dim=1)
                    action = qs.multinomial(1).data
                    with self.budget.get_lock(): self.budget.value -= 1
            elif self.args.rand_advice:
                if random.random() < 0.5:
                    _, qs, _ = self.demonstration((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
                    qs = F.softmax(qs, dim=1)
                    action = qs.multinomial(1).data
                    with self.budget.get_lock(): self.budget.value -= 1
            else:
                _, qs, _ = self.demonstration((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
                im = torch.max(qs) - torch.min(qs)
                if im > self.im_sigma:
                    qs = F.softmax(qs, dim=1)
                    action = qs.multinomial(1).data
                    with self.budget.get_lock(): self.budget.value -= 1

        # log_prob = log_prob.gather(1, Variable(action))   # 是否应该受demonstration的action 影响？
        state, self.reward, self.done, self.info = self.env.step(
            action[0][0].cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
