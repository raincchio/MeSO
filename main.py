from utils.default import variant, get_cmd_args, get_log_dir

import utils.pytorch_util as ptu
from component.replay_buffer import ReplayBuffer
from utils.env_utils import domain_to_epoch, env_producer

from utils.launcher_util import set_up
from component.path_collector import MdpPathCollector, RemoteMdpPathCollector

from model.agent import get_policy_producer, get_q_producer, get_trainer
from component.framework import Framework
# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled=False

import ray
# ray.init()
# import logging
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    # log_to_driver=True,
    # The amount of memory (in bytes)
    num_cpus=1,
    object_store_memory=int(5e8), # 1g
    _redis_max_memory=int(5e8) # 1g
)


if __name__ == "__main__":

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)

    variant['seed'] = args.seed
    variant['domain'] = args.domain

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain)
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['batch_size'] = args.batch_size

    variant['trainer_kwargs']['use_automatic_entropy_tuning'] = args.no_aet
    variant['trainer_kwargs']['beta'] = args.beta
    variant['trainer_kwargs']['alpha'] = args.alpha
    variant['trainer_kwargs']['alphap'] = args.alphap

    set_up(variant, seed=args.seed, use_gpu=args.use_gpu)

    domain = variant['domain']
    seed = variant['seed']
    expl_env = env_producer(domain, seed)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    # Get producer function for policy and value functions

    M = variant['layer_size']

    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M, M])
    # print(f'{bcolors.FAIL}pass2{bcolors.ENDC}')
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=[M, M])
    # Finished getting producer

    expl_path_collector = MdpPathCollector(
        expl_env,
    )
    remote_eval_path_collector = RemoteMdpPathCollector.remote(
        domain, seed,
        policy_producer
    )

    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )

    trainer = get_trainer(args.algo)(
        policy_producer,
        q_producer,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )

    algorithm = Framework(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,
        replay_buffer=replay_buffer,
        seed=variant['seed'],
        **variant['algorithm_kwargs'],
    )

    algorithm.to(ptu.device)
    algorithm.train()

