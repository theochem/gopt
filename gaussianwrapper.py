from saddle.fchk import FCHKFile
from horton import angstrom
from horton import periodic
from string import Template
import numpy as np
import os


class GaussianWrapper(object):

    counter = 0

    def __init__(self, molecule, title):
        self.molecule = molecule
        self.pwd = os.path.dirname(os.path.realpath(__file__))
        with open(self.pwd + "/single_hf_template.com", "r") as f:
            self.template = Template(f.read())
        self.title = title

    def run_gaussian_and_get_result(self, charge, multi, **kwargs):
        energy = kwargs.pop('energy', True)
        gradient = kwargs.pop('gradient', False)
        hessian = kwargs.pop('hessian', False)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        if gradient or hessian:
            freq = "freq"
        else:
            freq = ""
        filename = self.create_input_file(charge, multi, freq=freq)
        print "gausian is going to run \n{} \n{} \n{}".format(charge, multi, self.molecule.ic)
        fchk_file = self._run_gaussian(filename)
        assert isinstance(fchk_file, FCHKFile), "Gaussian calculation didn't run properly"
        result = []
        if energy:
            result.append(fchk_file.get_energy())
        if gradient:
            result.append(fchk_file.get_gradient())
        if hessian:
            result.append(fchk_file.get_hessian())
        return result

    def create_input_file(self, charge, multi, freq="freq"):
        atoms = ""
        for i in range(len(self.molecule.numbers)):
            x, y, z = self.molecule.coordinates[i] / angstrom
            atoms += ('%2s % 10.5f % 10.5f % 10.5f \n' %
                      (periodic[self.molecule.numbers[i]].symbol, x, y, z))
        filename = "{0}_{1}".format(self.title, self.counter)
        postfix = ".com"
        file_path = "/test/gauss/" + filename + postfix
        with open(self.pwd + file_path, "w") as f:
            f.write(self.template.substitute(charge=charge, freq=freq, multi=multi,
                                             atoms=atoms, title="{}_{}".format(self.title, GaussianWrapper.counter)))
            GaussianWrapper.counter += 1
        return filename
        # if run_cal:
        #     filename = "{0}_{1}.com".format(self.title, GaussianWrapper.counter)
        #     self._run_gaussian(filename)

    def _run_gaussian(self, filename, fchk=True, command_bin="g09"):
        fchk_ob = None
        path = self.pwd + "/test/gauss/"
        os.chdir(path)
        os.system("{0} {1}.com".format(command_bin, filename))
        if fchk:
            logname = "{0}.log".format(filename)
            if os.path.isfile(path + logname) and self._log_finish_test(path + logname):
                os.system(
                    "formchk {0}{1}.chk {0}{1}.fchk".format(path, filename))
                fchk_ob = FCHKFile("{0}{1}.fchk".format(path, filename))
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
    print a.template
    a.create_input_file(0, 2)
    a.create_input_file(0, 2, "freq")
