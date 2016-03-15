from pele.potentials import LJ

class LJEnergy(object):

    def __init__(self, ictransformation):
        self.molecule = ictransformation

    def get_energy_gradient(self):
        method = LJ()
        energy, gredient = method.getEnergyGradient(self.molecule.coordinates.reshape(1, -1)[0])
        return energy, gredient

    def get_energy_gradient_hessian(self):
        method = LJ()
        energy, gredient, hessian = method.getEnergyGradientHessian(self.molecule.coordinates.reshape(1, -1)[0])
        return energy, gredient, hessian