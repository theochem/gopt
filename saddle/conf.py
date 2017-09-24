import os
import json

from pkg_resources import Requirement, resource_filename

conf_path = resource_filename(__name__, "data/conf.json")

with open(conf_path) as json_data_f:
    json_data = json.load(json_data_f)

base_path = resource_filename(Requirement.parse('saddle'), '')

def get_path(given_path, base_path=base_path):
    if given_path.startswith('/' or '~'):  # abs path
        return given_path
    else:
        return os.path.join(base_path, given_path)

def set_work_dir(given):
    pass


def set_log_dir(given):
    pass


work_dir = get_path(json_data['work_dir'])

log_dir = get_path(json_data['log_dir'])
