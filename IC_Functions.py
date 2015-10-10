import numpy as np
import molmod as mm
import horton as ht



class IC_Functions(object):
    """IC_Functions is a set for store different coordinate transforming method.

    """

    @staticmethod
    def dihed_new_dot(rs, deriv=0):
        """Compute the new dihedral between the planes rs[0], rs[1], rs[2] and rs[1], rs[2], rs[3]

       Arguments:
        | ``rs`` -- four numpy array with three elements
        | ``deriv`` -- the derivatives to be computed: 0, 1 or 2 [default = 0]
        """
        return mm._dihed_transform(rs, IC_Functions._dihed_new_dot, deriv)


    @staticmethod
    def dihed_new_cross(rs, deriv=0):
        """Compute the new dihedral between the planes rs[0], rs[1], rs[2] and rs[1], rs[2], rs[3]

        Arguments:
         | ``rs`` -- four numpy array with three elements
         | ``deriv`` -- the derivatives to be computed: 0, 1 or 2 [default = 0]
        """
        return mm._dihed_transform(rs, IC_Functions._dihed_new_cross, deriv)


    @staticmethod 
    def _dihed_new_dot(av, bv, cv, deriv):
        """Similar to dihed_ang and dihed_cos, but a novel method to calculate dihed between to planes"""
        a = mm.Vector3(9, deriv, av, (0, 1, 2))
        b = mm.Vector3(9, deriv, bv, (3, 4, 5))
        c = mm.Vector3(9, deriv, cv, (6, 7, 8))
        a /= a.norm()
        b /= b.norm()
        c /= c.norm()
        return mm.dot(a, c).results()


    @staticmethod
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


    @staticmethod
    def bond_length(atoms, deriv = 0):
        if len(atoms) != 2:
            raise AtomsNumberError
        rs = np.array(atoms)
        return mm.bond_length(rs, deriv)

    
    @staticmethod
    def bend_angle(atoms, deriv = 0):
        if len(atoms) != 3:
            raise AtomsNumberError
        rs = np.array(atoms)
        return mm.bend_angle(rs, deriv)


    @staticmethod
    def dihed_angle(atoms, deriv = 0):
        if len(atoms) != 4:
            raise AtomsNumberError
        rs = np.array(atoms)
        return mm.dihed_angle(rs, deriv)


    @staticmethod
    def dihed_angle_new_dot(atoms, deriv = 0):
        if len(atoms) != 4:
            raise AtomsNumberError
        rs = np.array(atoms)    
        return IC_Functions.dihed_new_dot(rs, deriv) 
    

    @staticmethod
    def dihed_angle_new_cross(atoms, deriv = 0):
        if len(atoms) != 4:
            raise AtomsNumberError
        rs = np.array(atoms)    
        return IC_Functions.dihed_new_cross(rs,deriv)



class AtomsNumberError(Exception):
    pass



# if __name__ == '__main__':
#     fn_xyz = ht.context.get_fn('test/2h-azirine.xyz')
#     mol = ht.IOData.from_file(fn_xyz).coordinates
#     print IC_Functions.bond_length([mol[0],mol[1]])
#     # print IC_Functions.bend_angle([mol[0], mol[1], mol[2], mol[3]])
#     print IC_Functions.bend_angle([mol[0], mol[1], mol[2]])
#     print IC_Functions.dihed_angle_new_dot([mol[0], mol[1], mol[2], mol[3]])
#     print IC_Functions.dihed_angle_new_cross([mol[0], mol[1], mol[2], mol[3]])