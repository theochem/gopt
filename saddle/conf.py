import json
import os

cur_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(cur_path, 'conf.json')) as json_data_f:
    json_data = json.load(json_data_f)


def get_path(json_path):
    if json_path.startswith('/' or '~'):  # abs path
        return json_path
    else:
        return os.path.join(cur_path, json_path)


data_dir = get_path(json_data['data_dir'])

work_dir = get_path(json_data['work_dir'])

log_dir = get_path(json_data['log_dir'])
