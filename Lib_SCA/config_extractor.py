from typing import Dict
import yaml
import os.path
from collections import namedtuple

# use a namedtuple to make the cache entries a little more clear
CacheEntry = namedtuple('CacheEntry', ['entry', 'mtime'])


class TraceConfig:
    """
    this class manages configurations for all input files (.yaml)

    the structure of the config YAML files is complex. for a reference of all the required fields, just look at
    demo.yaml in configs.

    the cache will store serialized config data so we don't have to hit the disk every time we want a config field.
    they map a config name to a tuple containing the config entry and the time that config was last modified.
    the modified time is used so that if a config is changed while the program is running, the cache is cleared and
    the new version of the config is retrieved.
    """
    CONFIG_PATH = 'Lib_SCA/configs/'
    config_cache: Dict[str, CacheEntry] = {}

    def get_config(self, config_name):
        path = self.CONFIG_PATH + config_name
        # print(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError as e:
            raise ValueError(f'config "{config_name}" not found"') from e

        if config_name not in self.config_cache or self.config_cache[config_name].mtime < mtime:
            with open(path) as infile:
                entry = yaml.safe_load(infile)
            self.config_cache[config_name] = CacheEntry(entry=entry, mtime=mtime)
        return self.config_cache[config_name].entry
