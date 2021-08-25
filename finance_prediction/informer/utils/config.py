import os
import yaml
import argparse

CONFIG_FILE = 'params.yaml'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs'))

def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    file_name = os.path.join(ROOT_PATH, CONFIG_FILE)
    with open(file_name) as f:
        yaml_config = yaml.safe_load(f)

    for key, value in yaml_config.items():
        setattr(args, key, value)

    return args
