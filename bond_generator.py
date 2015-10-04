import ic
import horton as ht
import numpy as np

# Use saddle.ic to generate proper internal coordinates for a molecule.
# Five kinds of bond are defined depens on the specifit situation.

class Bond(object):
    """bond class is used to store the information for bonds between two atoms.
    """

    def __init__(self, length, atom1, atom2, type = 1):
        self.value = length
        self.atoms = set([atom1, atom2])
        self.type = type


class Aux_Bond(object):
    """subclass of bond class. Used to contain auxiliary bond object.
    """

    def __init__(self, length, atom1, atom2, type = 5):
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
     | ``numbers`` Atomic number of each atom in a list form
     | ``len`` Number of atoms in this molecule
     | ``coordiantes`` Coorrdinates of each atom in cartecian coordinates
     | ``molecule`` An instance of Coordinate_Transform object from ic.py
     | ``ic_types`` Contain instances of different kinds of internal_coordinates class
     | ``B_matrix`` Matrix to do transformation from cartecian to internal
     | ``atoms`` Atom pairs or set pairs for any internal coordinates that have already been added
     | ``connect`` Contain connectivity information for each atom in a dict form
     | ``internal_coordinates`` Internal coordinates of molecule.


     bond_type explanation:
     ``1`` regular bond
     ``2`` hydrogen bond
     ``3`` inter-fragment bond
     ``4`` linear-chain bond
     ``5`` auxiliary bond
    """

    def __init__(self, mol):
        self.numbers = mol.numbers 
        self.len = len(mol.numbers)
        self.coordinates = (mol.coordinates/ht.angstrom)
        self.molecule = ic.Coordinate_Transform(self.coordinates)
        self.ic_types = []
        self.B_matrix = np.zeros((0, 3*self.len),float)
        self.atoms = []
        self.connect = dict([(i, set()) for i in range(self.len)])
        self.internal_coordinates = []
        self.steps = []
        self.auxiliary = []
        self.H_matrix = np.zeros((0, 3*self.len, 3*self.len), float)


    def add_bond(self, atom1, atom2, type=1):
        bond_length, deriv1, deriv2 = self.molecule.bond_length(atom1, atom2, deriv = 2)
        bond_object = Bond(bond_length, atom1, atom2, type)
        if bond_object.atoms in self.atoms:
            print "bond have already existed"
            return
        self.ic_types.append(bond_object) 
        self.connect[atom1].add(atom2)
        self.connect[atom2].add(atom1)
        self._fill_B_matrix(deriv1, atom1, atom2)
        self._fill_H_matrix(deriv2, atom1, atom2)
        self.atoms.append(bond_object.atoms)
        self.internal_coordinates.append(bond_length)
        self.steps.append((atom1, atom2))



    def add_aux_bond(self, atom1, atom2, type=5):
        bond_length, deriv1, deriv2 = self.molecule.bond_length(atom1, atom2, deriv =2)
        bond_object = Aux_Bond(bond_length, atom1, atom2, type)
        if bond_object.atoms in self.atoms:
            print "bond have already existed"
            return        
        self.auxiliary.append(bond_object)



    def add_angle(self, atom1, atom2, atom3):
        bond_angle, deriv1, deriv2 = self.molecule.bend_angle(atom1, atom2, atom3, deriv = 2)
        angle_object = Angle(bond_angle, atom1, atom2, atom3)
        if angle_object.atoms in self.atoms:
            print "angle have already existed"
            return
        self.ic_types.append(angle_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3)
        self._fill_H_matrix(deriv2, atom1, atom2, atom3)
        self.atoms.append(angle_object.atoms)
        self.internal_coordinates.append(bond_angle)
        self.steps.append((atom1, atom2, atom3))


    def add_robust_dihed(self, atom1, atom2, atom3, atom4):
        dihed_atoms = (set([atom2, atom3]),set([atom1, atom4]))
        print dihed_atoms
        if dihed_atoms in self.atoms:
            print "dihed have already existed"
            return
        self._add_dihed_dot(atom1, atom2, atom3, atom4)
        self._add_dihed_cross(atom1, atom2, atom3, atom4)
        self.atoms.append(dihed_atoms)
        self.steps.append((atom1, atom2, atom3, atom4))


    def _add_dihed_dot(self, atom1, atom2, atom3, atom4):
        dihed_angle_newdot, deriv1, deriv2 = self.molecule.dihed_angle_new_dot(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Dot(dihed_angle_newdot, atom1, atom2, atom3, atom4)
        self.ic_types.append(dihed_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)
        self._fill_H_matrix(deriv2, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newdot)


    def _add_dihed_cross(self, atom1, atom2, atom3, atom4):
        dihed_angle_newcross, deriv1, deriv2 = self.molecule.dihed_angle_new_cross(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Cross(dihed_angle_newcross, atom1, atom2, atom3, atom4)
        self.ic_types.append(dihed_object)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)
        self._fill_H_matrix(deriv2, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newcross)


    # def add_dihed_angle(self, atom1, atom2, atom3, atom4, type = )


    def _fill_B_matrix(self, deriv1, *atoms):
        tmp_B_matrix = np.zeros((len(self.ic_types),3*self.len),float)
        tmp_B_matrix[:-1,:] = self.B_matrix
        for i in range(len(atoms)):
            tmp_B_matrix[-1, 3*atoms[i]:3*atoms[i]+3] += deriv1[i]
        self.B_matrix = np.zeros((len(self.ic_types),3*self.len),float)
        self.B_matrix[:,:] = tmp_B_matrix


    def _fill_H_matrix(self, deriv2, *atoms):
        tmp_H_matrix = np.zeros((len(self.ic_types), 3*self.len, 3*self.len), float)
        tmp_H_matrix[:-1, : , : ] = self.H_matrix
        for i in range(len(atoms)):
            for j in range(3):
                tmp_H_matrix[-1, 3*atoms[i]+j, 3*atoms[i]: 3*atoms[i]+3] += deriv2[i][j][i]
        self.H_matrix = np.zeros((len(self.ic_types), 3*self.len, 3*self.len), float)
        self.H_matrix = tmp_H_matrix



    def set_target_internal(self, target_ic):
        self.target_ic = target_ic
        self.difference = np.array(self.target_ic) - np.array(self.internal_coordinates)

    def transform_i_to_c(self):
        B_pseudo = np.linalg.pinv(self.B_matrix)
        cartecian_change = np.dot(B_pseudo, self.difference)
        self.target_cc = (self.coordinates.reshape(1,-1) + cartecian_change).reshape(3,-1)

    # def iteration(self):
        # Iteration( )



class IC_Iter(IC):
    '''class used to do IC transform iteration. Most attributes and function are similar to IC class.
    '''

    def __init__(self, coordinates, target_ic, procedures):
        self.coordinates = coordinates
        self.len = len(coordinates)
        self.molecule = ic.Coordinate_Transform(self.coordinates)
        self.B_matrix = np.zeros((0,3*self.len),float)
        self.internal_coordinates = []
        self.target_ic = target_ic
        self.procedures = procedures


    def add_bond(self, atom1, atom2, type = 1):
        bond_length, deriv1, deriv2 = self.molecule.bond_length(atom1, atom2, deriv = 2)
        bond_object = Bond(bond_length, atom1, atom2)
        self.internal_coordinates.append(bond_length)
        self._fill_B_matrix(deriv1, atom1, atom2)


    def add_angle(self, atom1, atom2, atom3):
        bond_angle, deriv1, deriv2 = self.molecule.bend_angle(atom1, atom2, atom3, deriv = 2)
        angle_object = Angle(bond_angle, atom1, atom2, atom3)
        self.internal_coordinates.append(bond_angle)
        self._fill_B_matrix(deriv1, atom1, atom2)
 

    def add_robust_dihed(self, atom1, atom2, atom3, atom4):
        self._add_dihed_dot(atom1, atom2, atom3, atom4)
        self._add_dihed_cross(atom1, atom2, atom3, atom4)


    def _add_dihed_dot(self, atom1, atom2, atom3, atom4):
        dihed_angle_newdot, deriv1, deriv2 = self.molecule.dihed_angle_new_dot(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Dot(dihed_angle_newdot, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newcross)
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)

    def _add_dihed_cross(self, atom1, atom2, atom3, atom4):
        dihed_angle_newcross, deriv1, deriv2 = self.molecule.dihed_angle_new_cross(atom1, atom2, atom3, atom4, deriv = 2)
        dihed_object = Dihed_Cross(dihed_angle_newcross, atom1, atom2, atom3, atom4)
        self.internal_coordinates.append(dihed_angle_newcross) 
        self._fill_B_matrix(deriv1, atom1, atom2, atom3, atom4)


    def _fill_B_matrix(self, deriv1, *atoms):
        tmp_B_matrix = np.zeros((len(self.internal_coordinates),3*self.len),float)
        tmp_B_matrix[:-1,:] = self.B_matrix
        for i in range(len(atoms)):
            tmp_B_matrix[-1, 3*atoms[i]:3*atoms[i]+3] += deriv1[i]
        self.B_matrix = np.zeros((len(self.internal_coordinates),3*self.len),float)
        self.B_matrix[:,:] = tmp_B_matrix


    def ic_auto_add(self):
        for i in self.procedures:
            if len(i) == 2:
                self.add_bond(i[0], i[1])
            elif len(i) == 3:
                self.add_angle(i[0], i[1], i[2])
            elif len(i) == 4:
                self.add_robust_dihed(i[0], i[1], i[2], 1[3])






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
    target_ic = ([0.5])
    water.add_bond(0,1, 1)
    water.add_bond(1,2, 1)
    water.add_angle(0,1,2)
    water.set_target_internal(target_ic)
    print water.B_matrix
    print water.H_matrix
    print water.H_matrix.shape
    water.transform_i_to_c()
    print water.target_cc
    print water.difference
    # water.add_bond(1,2, 1)
    # water.add_angle(0,1,2)
    # water.set_target_internal(target_ic)
    # water.transform_i_to_c()
    # print water.target_cc
    # test = IC_Iter(water.target_cc, target_ic, water.steps)
    # test.ic_auto_add()
    # print test.target_ic
    # print test.internal_coordinates