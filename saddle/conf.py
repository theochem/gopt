import json
import os

cur_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(cur_path, 'conf.json')) as json_data_f:
    json_data = json.load(json_data_f)


def get_path(given_path, base_path=cur_path):
    if given_path.startswith('/' or '~'):  # abs path
        return given_path
    else:
        return os.path.join(base_path, given_path)


data_dir = get_path(json_data['data_dir'])

work_dir = get_path(json_data['work_dir'])

log_dir = get_path(json_data['log_dir'])
