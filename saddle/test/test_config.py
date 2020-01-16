from unittest import TestCase

from importlib_resources import path
from pathlib import PosixPath
from saddle.conf import Config


class TestUtils(TestCase):
    def setUp(self):
        Config.reset_path()

    def test_load_const(self):
        work_dir = Config.get_path(key="work_dir")
        with path("saddle", "") as ref_path:
            assert work_dir == (ref_path / "work")

    def test_set_work(self):
        Config.set_path("work_dir", "/usr/local")
        work_dir = Config.get_path("work_dir")
        assert work_dir == PosixPath("/usr/local")

        Config.reset_path()
        new_work_dir = Config.get_path("work_dir")
        with path("saddle", "") as ref_path:
            assert new_work_dir == (ref_path / "work")

    @classmethod
    def tearDownClass(cls):
        Config.reset_path()
