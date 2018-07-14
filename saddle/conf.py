# -*- coding: utf-8 -*-
# PyGopt: Python Geometry Optimization.
# Copyright (C) 2011-2018 The HORTON/PyGopt Development Team
#
# This file is part of PyGopt.
#
# PyGopt is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# PyGopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"Config file to configure file directory"

import json

from importlib_resources import path
from pathlib import Path, PosixPath, WindowsPath


class Config:
    # load config contents from conf.json
    with path('saddle.data', 'conf.json') as json_path:
        with json_path.open(encoding='utf-8') as json_data_f:
            json_data = json.load(json_data_f)

    # set base path
    with path('saddle', '') as saddle_path:
        base_path = saddle_path

    @staticmethod
    def _find_path(given_path: str, system='posix'):
        if system == 'posix':
            given_path = PosixPath(given_path)
        elif system == 'windows':
            given_path = WindowsPath(given_path)
        else:
            raise ValueError(f'system {system} is not supported')
        return given_path

    @classmethod
    def get_path(cls, key: str):
        try:
            keyword_path = cls.json_data[key]
        except KeyError:
            print(f'Given key {key} is not in conf file')
        keyword_path = cls._find_path(keyword_path)
        if not keyword_path.is_absolute():
            keyword_path = cls.base_path / keyword_path
        return keyword_path

    @classmethod
    def set_path(cls, key: str, new_path):
        if key not in cls.json_data.keys():
            raise ValueError(f"Give key {key} is not in conf file")
        new_path = cls._find_path(new_path)
        if not new_path.is_absolute():
            new_path = (Path() / new_path).resolve()
        cls.json_data[key] = str(new_path)
        with path('saddle.data', 'conf.json') as json_path:
            with json_path.open(mode='w', encoding='utf-8') as json_data_f:
                json.dump(cls.json_data, json_data_f)

    @classmethod
    def reset_path(cls):
        cls.json_data['work_dir'] = 'work'
        cls.json_data['log_dir'] = 'work/log'
        with path('saddle.data', 'conf.json') as json_path:
            with json_path.open(mode='w', encoding='utf-8') as json_data_f:
                json.dump(cls.json_data, json_data_f)


WORK_DIR = Config.get_path('work_dir')

LOG_DIR = Config.get_path('log_dir')
