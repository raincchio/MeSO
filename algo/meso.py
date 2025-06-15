from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import np_to_pytorch_batch

import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from model.distribution import TanhNormal


class MeSOTrainer(object):
    def __init__(
            self,
            policy_producer,
            q_producer,

            action_space=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            beta=1,
            alpha=1,
            alphap=1,
            **kwargs,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = - \
                    np.prod(action_space.shape).item()

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale
        self.beta = beta
        self.alphap = alphap

        """
        The class mutable state
        """
        self.policy = policy_producer()
        self.log_alpha = ptu.tensor(np.log(alpha), requires_grad=True)
        # self.log_alpha = ptu.zeros(1, requires_grad=True)
        self.qf1 = q_producer()
        self.qf2 = q_producer()
        self.target_qf1 = q_producer()
        self.target_qf2 = q_producer()
        self.vf = q_producer()

        self.alpha_optimizer = optimizer_class(
            [self.log_alpha],
            lr=policy_lr,
        )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, np_batch, epoch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch, epoch)

    def train_from_torch(self, batch, epoch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        alpha and policy update
        """
        mean, std = self.policy(obs)
        tanh_normal = TanhNormal(mean, std)

        raction, pre_tanh_value = tanh_normal.rsample()
        # raction_is, pre_tanh_value_is = tanh_normal.sample() # for foward kl divergence esimation

        log_pi = tanh_normal.log_prob_sum(
            raction, pre_tanh_value
        )

        alpha_loss = -(self.log_alpha *(log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp().detach()
        beta = self.beta
        alphap = (1-epoch/1000)*self.alphap*alpha

        """
        Q update
        """

        # Make sure policy accounts for squashing
        # functions like tanh correctly!

        next_mean, next_std = self.policy(next_obs)
        next_tanh_normal = TanhNormal(next_mean, next_std)
        next_a, next_pre_tanh_value = next_tanh_normal.sample()
        next_log_prob = next_tanh_normal.log_prob_sum(
            next_a, next_pre_tanh_value
        )

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        with torch.no_grad():
            # next_state_value = self.vf(next_obs, torch.zeros_like(actions))
            # q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * next_state_value
            next_q_value = torch.min(self.target_qf1(next_obs, next_a), self.target_qf2(next_obs, next_a)) - alphap * next_log_prob
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * next_q_value

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        policy update
        """

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        with torch.no_grad():
            next_v_values = self.vf(next_obs, torch.zeros_like(actions))
            target_v_values = self.reward_scale * rewards + (1. - terminals) * self.discount * next_v_values

        state_value = self.vf(obs, torch.zeros_like(actions))
        vf_loss = self.vf_criterion(state_value, target_v_values)

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        minq = torch.min(self.qf1(obs, raction), self.qf2(obs, raction))

        # q_is = self.qf1(obs, raction_is)
        # log_pi_is = tanh_normal.log_prob_sum(
        #     raction_is, pre_tanh_value_is
        # )

        log_pi_is = tanh_normal.log_prob_sum(
            raction.detach(), pre_tanh_value.detach()
        )

        with torch.no_grad():
            # to avoid once more sampling, reuse the next state to predict the IS res
            eps = 1e-9
            state_value[state_value < eps] = eps
            is_ratio = (1/alpha*minq - torch.log(state_value) - log_pi).exp().clamp(0.8,1.2)
            # clipped_ratio = (beta*ratio).clamp(0,1)

        policy_loss = (alpha*log_pi - alpha*beta*is_ratio*log_pi_is - minq).mean()

        # fk_policy_loss = (alpha*clipped_ratio*log_pi_is).mean()

        # policy_loss = rk_policy_loss+fk_policy_loss


        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            # policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            # self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Q2 Predictions',
            #     ptu.get_numpy(q2_pred),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Q Targets',
            #     ptu.get_numpy(q_target),
            # ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Std',
                ptu.get_numpy(std),
            ))
            self.eval_statistics['Alpha'] = alpha.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf
        ]
