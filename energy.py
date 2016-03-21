import numpy as np
import horton as ht

from saddle.pyscf_wrapper import gobasis

class EnergyCompute(object):

    def __init__(self, transformation_mol, default="3-21G", spin=0):
        self.coordinates = mol.coordinates
        self.numbers = mol.numbers
        self.default = default
        self.pseudo_numbers = mol.pseudo_numbers
        self.obasis = gobasis(self.coordinates, self.numbers, self.default)

    def get_energy(self, alpha_ele):
        lf = ht.DenseLinalgFactory(self.obasis.nbasis)
        olp = self.obasis.compute_overlap(lf)
        # print self.obasis.compute_overlap_gradient(lf)
        kin = self.obasis.compute_kinetic(lf)
        # kin_g = self.obasis.compute_kinetic_gradient(lf)
        na = self.obasis.compute_nuclear_attraction(self.coordinates, self.pseudo_numbers, lf)
        # na_g = self.obasis.compute_nuclear_attraction_gradient(self.coordinates, self.pseudo_numbers, lf)
        er = self.obasis.compute_electron_repulsion(lf)
        # er_g = self.obasis.compute_electron_repulsion_gradient(lf)
        exp_alpha = lf.create_expansion()
        ht.guess_core_hamiltonian(olp, kin, na, exp_alpha)
        terms = [
            ht.RTwoIndexTerm(kin, 'kin'),
            ht.RDirectTerm(er, 'hartree'),
            ht.RExchangeTerm(er, 'x_hf'),
            ht.RTwoIndexTerm(na, 'ne'),
        ]
        ham = ht.REffHam(terms)
        occ_model = ht.AufbauOccModel(alpha_ele)
        scf_solver = ht.PlainSCFSolver(1e-6)
        scf_solver(ham, lf, olp, occ_model, exp_alpha)
        # print ham.cache["energy"]

if __name__ == '__main__':
    fn_xyz = ht.context.get_fn('test/water.xyz')
    print fn_xyz
    mol = ht.IOData.from_file(fn_xyz)
    ec = EnergyCompute(mol)
    result = ec.get_energy(5)
