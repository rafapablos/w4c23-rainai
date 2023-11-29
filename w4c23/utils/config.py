"""
Custom Config Loader that allows to use !include in YAML files.
"""

import yaml
import os


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, "r") as f:
            return yaml.load(f, Loader)


Loader.add_constructor("!include", Loader.include)


def load_config(config_path):
    """Load confgiuration file

    Args:
        config_path (String): path to configuration file

    Returns:
        dict: configuration file
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader)
    return config
