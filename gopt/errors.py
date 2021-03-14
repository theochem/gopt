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
"""Custom Exception types."""


class NotSetError(Exception):
    """Value not set error."""

    pass


class AtomsNumberError(Exception):
    """Atoms number error."""

    pass


class ICNumberError(Exception):
    """Internal coordinates number error."""

    pass


class AtomsIndexError(Exception):
    """Atomic index error."""

    pass


class NotConvergeError(Exception):
    """Optimization not converge error."""

    pass


class InputTypeError(Exception):
    """Input type is not desired error."""

    pass


class PositiveProductError(Exception):
    """Positive product error."""

    pass


class InvalidArgumentError(Exception):
    """Invalid Argument error."""

    pass


class OverIterLimitError(Exception):
    """Over iteration limit error."""

    pass


class InvalidInputError(Exception):
    """Invalid input error."""

    pass


class OptError(Exception):
    """Opt result error."""

    pass
