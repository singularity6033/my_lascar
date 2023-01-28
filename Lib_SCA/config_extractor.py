from typing import Dict
import yaml
import json
import os.path
import jsonlines
from collections import namedtuple

# use a namedtuple to make the cache entries a little more clear
CacheEntry = namedtuple('CacheEntry', ['entry', 'mtime'])


class YAMLConfig:
    """
    this class manages configurations for input files (.yaml)

    the structure of the config YAML files is complex. for a reference of all the required fields, just look at
    demo.yaml in configs.

    the cache will store serialized config data so we don't have to hit the disk every time we want a config field.
    they map a config name to a tuple containing the config entry and the time that config was last modified.
    the modified time is used so that if a config is changed while the program is running, the cache is cleared and
    the new version of the config is retrieved.
    """
    CONFIG_PATH = './configs/yaml'
    config_cache: Dict[str, CacheEntry] = {}

    def get_config(self, config_name):
        path = self.CONFIG_PATH + config_name + '.yaml'
        try:
            mtime = os.path.getmtime(path)
        except OSError as e:
            raise ValueError(f'config "{config_name}" not found"') from e

        if config_name not in self.config_cache or self.config_cache[config_name].mtime < mtime:
            with open(path) as infile:
                entry = yaml.safe_load(infile)
            self.config_cache[config_name] = CacheEntry(entry=entry, mtime=mtime)
        return self.config_cache[config_name].entry


class JSONConfig:
    """
    this class manages configurations for input files (.json)

    .json file may contain multiple objects of config parameters, usually used in the scanning program
    write and read multiple records sequentially
    """
    CONFIG_PATH = './configs/json/'
    config_cache: Dict[str, CacheEntry] = {}

    def __init__(self, config_name):
        self.config_name = config_name
        self.path = self.CONFIG_PATH + self.config_name + '.json'
        if os.path.isfile(self.path):
            os.remove(self.path)

    def get_config(self):
        """
        output a list of dicts
        """
        json_list = list()
        try:
            mtime = os.path.getmtime(self.path)
        except OSError as e:
            raise ValueError(f'config "{self.config_name}" not found"') from e

        if self.config_name not in self.config_cache or self.config_cache[self.config_name].mtime < mtime:
            with open(self.path) as infile:
                for item in jsonlines.Reader(infile):
                    json_list.append(item)
                entry = json_list
            self.config_cache[self.config_name] = CacheEntry(entry=entry, mtime=mtime)
        return self.config_cache[self.config_name].entry

    def generate_config(self, dictionary):
        """
        input a list of dicts
        """
        with open(self.path, 'a') as f:
            json.dump(dictionary, f)
            f.write("\n")
