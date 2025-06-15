import random
import numpy as np
import torch
import json
import os.path as osp
from utils.logging import logger
from utils.pytorch_util import set_gpu_mode

def set_up(
        variant,
        seed=None,
        use_gpu=True
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    set_gpu_mode(use_gpu and torch.cuda.is_available(), seed)

    log_dir = variant['log_dir']
    logger.reset()
    setup_logger(
        log_dir=log_dir,
    )

    # Log the variant
    logger.log("Variant:")
    logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
    variant_log_path = osp.join(log_dir, 'variant.json')
    logger.log_variant(variant_log_path, variant)
    logger.log(f'Seed: {seed}')

    # ensure reproduce
    torch.set_num_threads(1)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup_logger(
    log_dir,
    text_log_file="debug.log",
    tabular_log_file="progress.csv",
    log_tabular_only=False,
):

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)

    logger.set_log_tabular_only(log_tabular_only)

    logger.log(f'Logging to: {log_dir}')


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for
                   k, v in data.items())
    return False