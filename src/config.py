"""Loads the YAML experiment configuration into a simple attribute-access object."""

import yaml


class Config:
    """Thin wrapper so config values can be accessed as `config.n_epochs`
    instead of `config['n_epochs']`, with the raw dict still available."""

    def __init__(self, config_dict):
        self._dict = config_dict
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return f"Config({self._dict})"


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)
