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
"""Config file to configure file directory.

Attributes
----------
LOG_DIR : Path
    Path obj to save/load log file
WORK_DIR : Path
    Path obj to save/load computation input/output file
"""

import json
from pathlib import Path, PosixPath, WindowsPath

from importlib_resources import path


class Config:
    """Config class for file directory."""

    # load config contents from conf.json
    with path("saddle.data", "conf.json") as json_path:
        with json_path.open(encoding="utf-8") as json_data_f:
            json_data = json.load(json_data_f)

    # set base path
    with path("saddle", "") as saddle_path:
        base_path = saddle_path

    @staticmethod
    def _find_path(given_path: str, system="posix"):
        """Turn given path into a proper path for given system.

        Parameters
        ----------
        given_path : str
            Path to be converted
        system : str, optional
            System type for getting the path, 'posix' or 'windows'

        Returns
        -------
        Path
            Generated path obj for locating certain directory

        Raises
        ------
        ValueError
            If system offered is not suppported
        """
        if system == "posix":
            given_path = PosixPath(given_path)
        elif system == "windows":
            given_path = WindowsPath(given_path)
        else:
            raise ValueError(f"system {system} is not supported")
        return given_path

    @classmethod
    def get_path(cls, key: str):
        """Get proper path for given key.

        Parameters
        ----------
        key : str
            key for certain type of directory path

        Returns
        -------
        Path
            proper path obj for given path key
        """
        try:
            keyword_path = cls.json_data[key]
        except KeyError:
            print(f"Given key {key} is not in conf file")
        keyword_path = cls._find_path(keyword_path)
        if not keyword_path.is_absolute():
            keyword_path = cls.base_path / keyword_path
        return keyword_path

    @classmethod
    def set_path(cls, key: str, new_path):
        """Set a new path for certain key path.

        Parameters
        ----------
        key : str
            key path to set
        new_path : str
            Preferred new path

        Raises
        ------
        ValueError
            Given key is not supported
        """
        if key not in cls.json_data.keys():
            raise ValueError(f"Give key {key} is not in conf file")
        new_path = cls._find_path(new_path)
        if not new_path.is_absolute():
            new_path = (Path() / new_path).resolve()
        cls.json_data[key] = str(new_path)
        with path("saddle.data", "conf.json") as json_path:
            with json_path.open(mode="w", encoding="utf-8") as json_data_f:
                json.dump(cls.json_data, json_data_f)

    @classmethod
    def reset_path(cls):
        """Reset all path to default."""
        cls.json_data["work_dir"] = "work"
        cls.json_data["log_dir"] = "work/log"
        with path("saddle.data", "conf.json") as json_path:
            with json_path.open(mode="w", encoding="utf-8") as json_data_f:
                json.dump(cls.json_data, json_data_f)


WORK_DIR = Config.get_path("work_dir")

LOG_DIR = Config.get_path("log_dir")
