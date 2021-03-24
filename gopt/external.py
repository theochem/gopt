from string import Template
import iodata
from importlib.resources import path
import subprocess

from gopt.conf import WORK_DIR
from gopt.cartesian import Cartesian


class BaseCompute:
    def __init__(self):
        ...

    def compute_energy(self):
        ...


class Gaussian(BaseCompute):
    def __init__(self, template=None, workdir=None):
        if template:
            self.template = template
        else:
            with path("gopt.data", "gauss_template.com") as filepath:
                self.template = filepath
        if workdir:
            self.workdir = workdir
        else:
            self.workdir = WORK_DIR

    def compute_energy(self, molecule, filename="", suffix=".com"):
        if isinstance(molecule, Cartesian):
            molecule = molecule.as_iodata()
        if not filename:
            filename = molecule.title if molecule.title else "Generated_by_GOpt"
        if filename.endswith != suffix:
            filename += suffix
        file_path = self.workdir / filename

        self._generate_input(molecule, file_path)
        self._run_calculation(file_path)
        mol = iodata.load_one(file_path)
        return {
            "energy": mol.energy,
            "gradient": mol.atgradient,
            "hessian": mol.athessian,
        }

    def _generate_input(self, molecule, filepath):
        extra_fields = {}
        extra_fields["title"] = (
            molecule.title if molecule.title else "Generated_by_GOpt"
        )
        extra_fields["work_path"] = filepath.parent / filepath.stem
        extra_fields["run_type"] = "freq SCF(XQC) nosymmetry"
        extra_fields["lot"] = "uhf"
        extra_fields["obasis_name"] = "6-31+G"

        iodata.write_input(
            molecule,
            f"{filepath}",
            fmt="gaussian",
            template=self.template,
            **extra_fields,
        )

    def _run_calculation(self, filepath):
        subprocess.run(["g16", f"{filepath}"])
        subprocess.run(["formchk", f"{filepath.parent / filepath.stem}.chk"])
