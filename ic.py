import numpy as np
import molmod as mm
import horton as ht
import molmod.ic as mmi 

# vector object mm.Vector3
# scalar object mm.Scalar
# doc, cross function mm.dot, mm.cross
# transform basic functions: mm.bond_length, mm.bend_cos, mm.bend_angle, mm.dihed_cos, mm.dihed_ang
# transform implement functions (private method): mm._bond_transform, mm.bend_transform, mm.dihed_transform

def dihed_new_dot(rs, deriv=0):
    """Compute the new dihedral between the planes rs[0], rs[1], rs[2] and rs[1], rs[2], rs[3]

   Arguments:
    | ``rs`` -- four numpy array with three elements
    | ``deriv`` -- the derivatives to be computed: 0, 1 or 2 [default = 0]
    """
    return mmi._dihed_transform(rs, _dihed_new_dot, deriv)


def dihed_new_cross(rs, deriv=0):
    """Compute the new dihedral between the planes rs[0], rs[1], rs[2] and rs[1], rs[2], rs[3]

    Arguments:
     | ``rs`` -- four numpy array with three elements
     | ``deriv`` -- the derivatives to be computed: 0, 1 or 2 [default = 0]
    """
    return mmi._dihed_transform(rs, _dihed_new_cross, deriv)


def _dihed_new_dot(av, bv, cv, deriv):
    """Similar to dihed_ang and dihed_cos, but a novel method to calculate dihed between to planes"""
    a = mm.Vector3(9, deriv, av, (0, 1, 2))
    b = mm.Vector3(9, deriv, bv, (3, 4, 5))
    c = mm.Vector3(9, deriv, cv, (6, 7, 8))
    a /= a.norm()
    b /= b.norm()
    c /= c.norm()
    return mm.dot(a, c).results()


def _dihed_new_cross(av, bv, cv, deriv):
    """Novel method for calculating dihed between two planes

    """
    a = mm.Vector3(9, deriv, av, (0, 1, 2))
    b = mm.Vector3(9, deriv, bv, (3, 4, 5))
    c = mm.Vector3(9, deriv, cv, (6, 7, 8))
    a /= a.norm()
    b /= b.norm()
    c /= c.norm()
    return mm.dot(mm.cross(a, c), b).results()


class Coordinate_Transform(object):
    """A class used to transform cartesian coordinate into internal coordinate 
    
    Arguments:
     | ``mol`` object of horton IODate class
     | ``atomX`` index of atoms to transform from cartesian coordinate to internal coordiantes.
     | ``deriv`` derivatives need to be calculated in the transformation.
    """

    def __init__(self, mol):
        self.coordinates = mol.coordinates/ht.angstrom
        self.numbers = mol.numbers


    def bond_length(self, atom1, atom2, deriv = 0):
        rs = np.vstack((self.coordinates[atom1],self.coordinates[atom2]))
        return mm.bond_length(rs, deriv)
    

    def bend_angle(self, atom1, atom2, atom3, deriv = 0):
        rs = np.vstack((self.coordinates[atom1],self.coordinates[atom2],self.coordinates[atom3]))
        return mm.bend_angle(rs, deriv)


    def dihed_angle(self, atom1, atom2, atom3, atom4, deriv = 0):
        rs = np.vstack((self.coordinates[atom1],self.coordinates[atom2],self.coordinates[atom3],self.coordinates[atom4]))
        return mm.dihed_angle(rs, deriv)


    def dihed_angle_new_dot(self, atom1, atom2, atom3, atom4, deriv = 0):
        rs = np.vstack((self.coordinates[atom1],self.coordinates[atom2],self.coordinates[atom3],self.coordinates[atom4]))    
        return dihed_new_dot(rs, deriv) 
    

    def dihed_angle_new_cross(self, atom1, atom2, atom3, atom4,deriv = 0):
        rs = np.vstack((self.coordinates[atom1],self.coordinates[atom2],self.coordinates[atom3],self.coordinates[atom4]))    
        return dihed_new_cross(rs,deriv)




if __name__ == '__main__':
    fn_xyz = ht.context.get_fn('test/2h-azirine.xyz')
    mol = ht.IOData.from_file(fn_xyz)
    structure = Coordinate_Transform(mol)
    print structure.numbers
    print len(structure.numbers)
    print structure.bond_length(1,2,1)
    print structure.dihed_angle_new_dot(1,0,2,3,2)[1]
    print structure.dihed_angle_new_cross(1,0,2,3,2)[1]
