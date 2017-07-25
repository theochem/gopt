from __future__ import absolute_import, print_function

import os
from string import Template

import numpy as np

from saddle.conf import data_dir, work_dir
from saddle.fchk import FCHKFile
from saddle.periodic import angstrom, periodic

__all__ = ('GaussianWrapper', )


class GaussianWrapper(object):

    counter = 0

    def __init__(self, molecule, title):
        self.molecule = molecule
        with open(
                os.path.join(data_dir, "single_hf_template.com"),
                "r") as f:
            self.template = Template(f.read())
        self.title = title

    def run_gaussian_and_get_result(self, charge, multi, **kwargs):
        coordinates = kwargs.pop('coordinates', True)
        energy = kwargs.pop('energy', True)
        gradient = kwargs.pop('gradient', False)
        hessian = kwargs.pop('hessian', False)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        if gradient or hessian:
            freq = "freq"
        else:
            freq = ""
        filename = self._create_input_file(charge, multi, freq=freq)
        # print "gausian is going to run \n{} \n{} \n{}".format(charge, multi,
        #   self.molecule.ic)
        fchk_file = self._run_gaussian(filename)
        assert isinstance(fchk_file,
                          FCHKFile), "Gaussian calculation didn't run properly"
        result = [None] * 4
        if coordinates:
            result[0] = fchk_file.get_coordinates()
        if energy:
            result[1] = fchk_file.get_energy()
        if gradient:
            result[2] = fchk_file.get_gradient()
        if hessian:
            result[3] = fchk_file.get_hessian()
        return result

    def create_gauss_input(self,
                           charge,
                           multi,
                           freq='freq',
                           spe_title='',
                           path='',
                           postfix='.com'):
        assert isinstance(path, str)
        assert isinstance(spe_title, str)
        atoms = ""
        for i in range(len(self.molecule.numbers)):
            x, y, z = self.molecule.coordinates[i] / angstrom
            atoms += ('%2s % 10.5f % 10.5f % 10.5f \n' %
                      (periodic[self.molecule.numbers[i]].symbol, x, y, z))
        if spe_title:
            filename = spe_title
        else:
            filename = "{0}_{1}".format(self.title, self.counter)
        if path:
            path = os.path.join(path, filename + postfix)
        else:
            path = os.path.join(work_dir, filename + postfix)
        with open(path, "w") as f:
            f.write(
                self.template.substitute(
                    charge=charge,
                    freq=freq,
                    multi=multi,
                    atoms=atoms,
                    title=filename))
        GaussianWrapper.counter += 1

    def _create_input_file(self, charge, multi, freq="freq"):
        filename = "{0}_{1}".format(self.title, self.counter)
        self.create_gauss_input(charge, multi, freq=freq, spe_title=filename)
        return filename

    def _run_gaussian(self, filename, fchk=True, command_bin="g09"):
        fchk_ob = None
        path = work_dir
        os.chdir(path)
        os.system("{0} {1}.com".format(command_bin, filename))
        if fchk:
            logname = "{0}.log".format(filename)
            if os.path.isfile(
                    os.path.join(path, logname)) and self._log_finish_test(
                        os.path.join(path, logname)):
                os.system("formchk {0}.chk {0}.fchk".format(
                    os.path.join(path, filename)))
                fchk_ob = FCHKFile(
                    "{0}.fchk".format(os.path.join(path, filename)))
        # os.chdir(os.path.join(self.pwd, '..'))
        # print("change_back", self.pwd)
        return fchk_ob

    def _log_finish_test(self, logname):
        flag = False
        with open(logname) as f:
            for line in f:
                if "Normal termination" in line:
                    flag = True
        return flag


if __name__ == '__main__':
    from collections import namedtuple
    molecule = namedtuple("molecule", "numbers, coordinates")
    aa = molecule([1, 3], np.array([[0., 0., 0.], [1., 1., 1.]]))
    a = GaussianWrapper(aa, "text_wrapper")
    print(a.template)
    a._create_input_file(0, 2)
    a._create_input_file(0, 2, "freq")
