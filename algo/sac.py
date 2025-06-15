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


class SACTrainer(object):
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
            add_baseline=False,
            target_entropy=None,
            alpha=1,
            # beta=0,
            **kwargs
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.add_baseline = add_baseline
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

        """
        The class mutable state
        """
        self.policy = policy_producer()
        self.qf1 = q_producer()
        self.qf2 = q_producer()
        self.target_qf1 = q_producer()
        self.target_qf2 = q_producer()

        if self.use_automatic_entropy_tuning:
            self.log_alpha = ptu.tensor(np.log(alpha), requires_grad=True)
            # self.log_alpha = ptu.zeros(1, requires_grad=True)
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

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, np_batch, *args):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        alpha update
        """
        mean, std = self.policy(obs)
        tanh_normal = TanhNormal(mean, std)

        raction1, pre_tanh_value1 = tanh_normal.rsample()

        _log_prob = tanh_normal.log_prob(
            raction1, pre_tanh_value1
        )
        log_pi = _log_prob.sum(dim=-1, keepdim=True)

        alpha_loss = -(self.log_alpha *(log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        """
        Q update
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        next_mean, next_std = self.policy(next_obs)
        tanh_normal = TanhNormal(next_mean, next_std)
        next_a, pre_tanh_value = tanh_normal.sample()
        _log_prob = tanh_normal.log_prob(
            next_a, pre_tanh_value
        )
        next_log_prob = _log_prob.sum(dim=-1, keepdim=True)

        target_q_values = torch.min(
            self.target_qf1(next_obs, next_a),
            self.target_qf2(next_obs, next_a),
        ) - alpha * next_log_prob

        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())


        """
        policy update
        """

        q_new_actions = torch.min(self.qf1(obs, raction1), self.qf2(obs, raction1))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()


        """
        Soft Updates
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
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy mu',
            #     ptu.get_numpy(policy_mean),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Policy log std',
            #     ptu.get_numpy(policy_log_std),
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'kl estimate',
            #     ptu.get_numpy(kl_estimate),
            # ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
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
        ]