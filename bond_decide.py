import ic
import horton as ht
import numpy as np

# Use saddle.ic to generate proper internal coordinates for a molecule.
# Five kinds of bond are defined depens on the specifit situation.

class Bond(object):
    """bond class is used to store the information for bonds between two atoms.
    """

    def __init__(self, length, atom1, atom2, type):
        self.value = length
        self.atoms = set([atom1, atom2])
        self.type = type


class Angle(object):
    """angle class is used to store the information for angles between two side atoms and one center atoms.
    """

    def __init__(self, angle, atom1, atom2, atom3):
        self.value = angle
        self.center_atoms = atom2
        self.side_atoms = set([atom1, atom3])
        self.atoms = (self.center_atoms, self.side_atoms)


class Dihed(object):
    """Dihed class is used to store the information for dihed for two planes. two subclass is list below.
    """

    def __init__(self, dihed, atom1, atom2, atom3, atom4):
        self.value = dihed
        self.center_atoms = set([atom2, atom3])
        self.side_atoms = set([atom1, atom4])
        self.atoms = (self.center_atoms, self.side_atoms)


class Dihed_Dot(Dihed):
    def __init__(self, dihed, atom1, atom2, atom3, atom4, type = "robust_dot"):
        super(Dihed_Dot, self).__init__(dihed, atom1, atom2, atom3, atom4)
        self.type = type


class Dihed_Cross(Dihed):
    def __init__(self, dihed, atom1, atom2, atom3, atom4, type = "robust_cross"):
        super(Dihed_Cross, self).__init__(dihed, atom1, atom2, atom3, atom4)
        self.type = type


class IC(object):
    """Internal coordinate class, which is like x,y,z but store some information for internal coordiantes

    Arguments:
     | ``coordiantes`` Coorrdinates of each atom in cartecian coordinates
     | ``numbers`` Atomic number of each atom in a list form
     | ``len`` Number of atoms in this molecule
     | ``molecule`` An instance of Coordinate_Transform object from ic.py
     | ``ic_types`` Contain instances of different kinds of internal_coordinates class
     | ``B_matrix`` Matrix to do transformation from cartecian to internal
     | ``atoms`` Atom pairs or set pairs for any internal coordinates that have already been added
     | ``connect`` Contain connectivity information for each atom in a dict form
    """

    def __init__(self, mol):
        self.numbers = mol.numbers 
        self.len = len(mol.numbers)
        self.molecule = ic.Coordinate_Transform(mol)
        self.coordinates = (mol.coordinates/ht.angstrom)
        self.ic_types = []
        self.B_matrix = np.zeros((0,3*self.len),float)
        self.atoms = []
        self.connect = dict([(i, set()) for i in range(self.len)])
        self.internal_coordinates = []


    def add_bond(self, atom1, atom2, type):
        bond_length, deriv1, deriv2 = self.molecule.bond_length(atom1, atom2, deriv = 2)
        bond_object = Bond(bond_length, atom1, atom2, type)
        if bond_object.atoms in self.atoms:
            print "bond have already existed"
            return
        self.ic_types.append(bond_object) 
        self.connect[atom1].add(atom2)
        self.connect[atom2].add(atom1)
        self._fill_B_matrix(deriv1, atom1, atom2)
        self.atoms.append(bond_object.atoms)
        self.internal_coordinates.append(bond_length)


    def add_angle(self, atom1, atom2, atom3):
        bond_angle, deriv1, deriv2 = self.molecule.bend_angle(atom1, atom2, atom3, deriv = 2)
        angle_object = Angle(bond_angle, atom1, atom2, atom3)
        if angle_object.atoms in self.atoms:
            print "angle have already existed"
            return
        self.ic_types.append(angle_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3)
        self.atoms.append(angle_object.atoms)
        self.internal_coordinates.append(bond_angle)


    def add_robust_dihed(self, atom1, atom2, atom3, atom4):
        dihed_atoms = (set([atom2, atom3]),set([atom1, atom4]))
        print dihed_atoms
        if dihed_atoms in self.atoms:
            print "dihed have already existed"
            return
        self._add_dihed_dot(atom1, atom2, atom3, atom4)
        self._add_dihed_cross(atom1, atom2, atom3, atom4)
        self.atoms.append(dihed_atoms)


    def _add_dihed_dot(self, atom1, atom2, atom3, atom4):
        dihed_angle_newdot, deriv1, deriv2 = self.molecule.dihed_angle_new_dot(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Dot(dihed_angle_newdot, atom1, atom2, atom3, atom4)
        self.ic_types.append(dihed_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newdot)


    def _add_dihed_cross(self, atom1, atom2, atom3, atom4):
        dihed_angle_newcross, deriv1, deriv2 = self.molecule.dihed_angle_new_cross(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Cross(dihed_angle_newcross, atom1, atom2, atom3, atom4)
        self.ic_types.append(dihed_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newcross)


    def _fill_B_matrix(self, deriv1, *atoms):
        tmp_B_matrix = np.zeros((len(self.ic_types),3*self.len),float)
        tmp_B_matrix[:-1,:] = self.B_matrix
        for i in range(len(atoms)):
            tmp_B_matrix[-1, 3*atoms[i]:3*atoms[i]+3] += deriv1[i]
        self.B_matrix = np.zeros((len(self.ic_types),3*self.len),float)
        self.B_matrix[:,:] = tmp_B_matrix


    def transform_i_to_c(self, difference):
        B_pseudo = np.linalg.pinv(self.B_matrix)
        cartecian_change = np.dot(B_pseudo, difference)
        return (self.coordinates.reshape(1,-1) + cartecian_change).reshape(3,-1)



if __name__ == '__main__':
    # mol = ht.IOData.from_file('../data/test/2h-azirine.xyz')
    # ic_object = IC(mol) 
    # print ic_object.connect
    # print ic_object.molecule
    # ic_object.add_bond(0,1,'regular')
    # print ic_object.B_matrix
    # print ic_object.B_matrix.shape
    # ic_object.add_angle(0,1,2)
    # print ic_object.B_matrix
    # print ic_object.B_matrix.shape
    # ic_object.add_bond(1,0,'regular') #duplicate bond add test
    # ic_object.add_angle(2,1,0)  #duplicate angle add test          
    # ic_object._add_dihed_dot(1,0,2,3)
    # print ic_object.B_matrix
    # print ic_object.ic_types
    # ic_object._add_dihed_cross(1,0,2,3)
    # print ic_object.B_matrix.shape
    # ic_object.add_robust_dihed(1,0,2,3) #duplicate dihed add test
    # print ic_object.ic_types
    # print ic_object.B_matrix.shape
    # print ic_object.B_matrix
    # ic_object.add_robust_dihed(3,2,0,1) #another duplicate dihed add test
    # print ic_object.B_matrix.shape
    # print ic_object.ic_types[4].internal_coordinates
    # ic_object.add_bond(1,3,'regular')
    # print ic_object.ic_types
    # print ic_object.internal_coordinates
    mol = ht.IOData.from_file('../data/test/water.xyz')
    water = IC(mol)
    water.add_bond(0,1, 'regular')
    water.add_bond(1,2, 'regular')
    water.add_angle(0,1,2)
    diff = [-0.1,-0.1,0]
    print water.coordinates
    print  water.internal_coordinates
    print water.transform_i_to_c(diff)