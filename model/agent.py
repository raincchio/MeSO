from model.networks import FlattenMlp
from model.policies import TanhGaussianPolicy

from algo.sac import SACTrainer
from algo.meso import MeSOTrainer


def get_policy_producer(obs_dim, action_dim, hidden_sizes):
    def policy_producer():
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

        return policy

    return policy_producer


def get_q_producer(obs_dim, action_dim, hidden_sizes):
    def q_producer():
        return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=1,
                          hidden_sizes=hidden_sizes, )

    return q_producer


def get_trainer(algo):
    return {'sac': SACTrainer, # original sac
            'meso': MeSOTrainer, # sac with policy update in advance
            }[algo]