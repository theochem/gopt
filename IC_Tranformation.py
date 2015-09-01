from IC_Functions import *
import numpy as np
import horton as ht


class IC_Transformation(object):
    """IC_Transformation is a class for coordinates transformation from cartesian coordinates to internal 
    coordinates and vise versa.

    Arguments:
     | ``molecule`` is a object of Horton IODate class

    Attributes:
     | ``coordinates`` -- cartesian coordinates of molecule in N*3 numpy array.
     | ``numbers`` -- a list of atomic number of each atoms of targit molecule.
     | ``ic`` -- a list of internal coordinates calculated through transformation.
     | ``iteration_falg`` -- a Bool flag to mark is it doing iteration or generate new ic. T for iteration, F for new ic.
     | ``procedures`` -- a list of steps used to generate ic which will be used when doing iteration
     | ``bond`` -- a list of added ic bond set
     | ``angle`` -- a list of added ic angle set
     | ``dihed`` -- a list of added ic dihed set 

    """

    def __init__(self, molecule):
        self.coordinates = molecule.coordinates
        self.numbers = molecule.numbers
        self.ic = []
        self.iteration_flag = False
        self.procedures = []
        self.bond = []
        self.angle = []
        self.dihed = []


    def add_bond_length(self, atom1, atom2, type):
        """To add a bond between atom1 and atom2

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
        """
        atoms = (atom1, atom2)
        info = "add_bond_length"
        if self._repetition_check(atoms):
            self._add_ic(info, atoms)


    def add_bend_angle(self, atom1, atom2, atom3):
        """To add an angle between atom1, atom2 and atom3.
        1-2-3, atom2 is the central atom of the angle

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
        """        
        atoms = (atom1, atom2, atom3)
        info = "add_bend_angle"
        self._add_ic(info, atoms)


    def add_dihed_angle(self, atom1, atom2, atom3, atom4):
        """To add an conventional angle between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
         | ``atom4`` the index of atom4 in self.numbers
        """ 
        atoms = (atom1, atom2, atom3, atom4)
        info = "add_dihed_angle"
        self._add_ic(info, atoms)


    def add_dihed_new(self, atom1, atom2, atom3, atom4):
        """To add an new dihedral angle indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
         | ``atom4`` the index of atom4 in self.numbers
        """ 
        atoms = (atom1, atom2, atom3, atom4)
        self._add_dihed_angle_new_dot(atoms)
        self._add_dihed_angle_new_cross(atoms)


    def _add_dihed_angle_new_dot(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 
        """ 
        info = "add_dihed_agnle_new_dot"
        self._add_ic(info, atoms)


    def _add_dihed_angle_new_cross(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 
        """ 
        info = "add_dihed_agnle_new_cross"
        self._add_ic(info, atoms)


    def _repetition_check(self, atoms):
        """private method to check whether the newly add ic has already existed or not
        """
        atoms_len = len(atoms)
        if atoms_len == 2:
            content = set(atoms)
            if content not in self.bond:
                self.bond.append(content)
                return True
            else: return False

        if atoms_len == 3:
            content = (atom[1], set(atom[0], atom[2]))
            if content not in self.angle:
                self.angle.append(content)
                return True
            else: return False
    
        if atoms_len == 4:
            content = (set(atoms[1], atoms[2]), set(atoms[0], atoms[3]))
            if content not in self.dihed:
                self.dihed.append(content)
                return True
            else: return False 


    def _add_ic(self, info, atoms):
        """ ic was added through this private method.
        """
        procedures = (info, atoms)
        if not self.iteration_flag:
            self.procedures.append(procedures)
        ic_function = IC_Transformation._IC_types[info]
        coordinates = self._get_coordinates(atoms)
        self.ic.append(ic_function(coordinates, deriv = 2)[0])


    def _get_coordinates(self, atoms):
        """private method to retrive atoms' cartesian coordinates
        """
        atom_length = len(atoms)
        coordinates = np.array([])
        for i in atoms:
            coordinates = np.append(coordinates, self.coordinates[i])
        return coordinates.reshape(-1, 3)


    _IC_types = {
        "add_bond_length":IC_Functions.bond_length,
        "add_bend_angle":IC_Functions.bend_angle,
        "add_dihed_angle":IC_Functions.dihed_angle,
        "add_dihed_agnle_new_dot":IC_Functions.dihed_angle_new_dot,
        "add_dihed_agnle_new_cross":IC_Functions.dihed_angle_new_cross,
    }
        
        

if __name__ == '__main__':
    fn_xyz = ht.context.get_fn('test/2h-azirine.xyz')
    mol = ht.IOData.from_file(fn_xyz)
    cc_object = IC_Transformation(mol)
    print cc_object.coordinates
    print cc_object.numbers
    cc_object.add_bond_length(0,1,"type")
    print cc_object.ic
    print cc_object.procedures[0][1]
    cc_object.add_bond_length(0,1,"type")
    print cc_object.ic
    
    
