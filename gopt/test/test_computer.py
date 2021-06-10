import re

# from unittest import TestCase
from importlib.resources import path, read_text

from gopt.conf import WORK_DIR
from gopt.computer import Gaussian

import iodata


def test_setup():
    gauss = Gaussian()
    ref_tmp = read_text("gopt.data", "gauss_template.com")
    contents = gauss.template.read_text()
    assert contents == ref_tmp
    gauss = Gaussian(template="", workdir=WORK_DIR)
    assert gauss.workdir == WORK_DIR


def test_create_input(tmp_path):
    gauss = Gaussian("", tmp_path)
    assert gauss.workdir == tmp_path

    with path("gopt.test.data", "ch4.xyz") as f:
        mol = iodata.load_one(f)
    gauss._generate_input(mol, gauss.workdir / "test.com")
    assert (tmp_path / "test.com").exists()

    contents = (tmp_path / "test.com").read_text()
    line_one = contents.split("\n")[0].strip()
    result = re.findall(r"%chk=(.*)", line_one)[0]
    assert result == str(tmp_path / "test.chk")


# def test_default_input():
#     gauss = Gaussian()
#     with path("gopt.test.data", "ch4.xyz") as f:
#         mol = iodata.load_one(f)
#     gauss._generate_input(mol, gauss.workdir / "test.com")
#     assert (WORK_DIR / "test.com").exists()
