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

import os
import json

from pkg_resources import Requirement, resource_filename

conf_path = resource_filename(__name__, "data/conf.json")

with open(conf_path) as json_data_f:
    json_data = json.load(json_data_f)

base_path = resource_filename(Requirement.parse('saddle'), '')


def get_path(given_path: str, base: str = base_path) -> str:
    if given_path.startswith('/' or '~'):  # abs path
        return given_path
    return os.path.join(base, given_path)


def set_work_dir(given):
    pass


def set_log_dir(given):
    pass


work_dir = get_path(json_data['work_dir'])

log_dir = get_path(json_data['log_dir'])
