from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np
from saddle.errors import (AtomsNumberError, InputTypeError,
                           InvalidArgumentError, NotSetError)
from saddle.internal import Internal
from saddle.path_ri import PathRI
from saddle.utils import Utils
from saddle.reduced_internal import ReducedInternal

__all__ = ('TSConstruct', )


class TSConstruct(object):
    """Transitian State Constructor

    Properties
    ----------
    reactant : Internal
        internal coordinates structure of reactant of certain reaction
    product : Internal
        internal coordinates structure of product of certain reaction
    ts : Internal
        internal coordinates structure of initial transition state guess of
        certain chemical reaction
    numbers : np.ndarray(N,)
        A numpy array of atomic number for input coordinates
    key_ic_number : int
        Number of key internal coordinates which correspond to important
        chemical property

    Classmethod
    -----------
    from_file(rct_file, prd_file, charge=0, multi=1):
        Create an instance from given file of reactant and product

    Methods
    -------
    __init__(reactant_ic, product_ic)
        Initializes constructor with the input of two Internal instance,
        each represent the structure of reactant and product respectively
    add_bond(atom1, atom2)
        Add bond connection for both reactant and product structures
        between atoms1 and atoms2
    add_angle_cos(atom1, atom2, atom3)
        Add angle cos to both reactant and product between angle atom1,
        atom2, and atom3
    add_dihedral(self, atom1, atom2, atom3, atom4):
        Add normal dihedral to both reactant and product between plane
        (atom1, atom2, and atom3) and plane(atom2, atom3, and atom4)
    delete_ic(self, *indices)
        Delete undesired internal coordinates with indices
    auto_select_ic(self, reset_ic=False, auto_select=True, mode="mix")
        Auto select internal coordinates of reactant and product with
        specific choices
    create_ts_state(start_with, ratio=0.5)
        Create transition state based on reactant and product internal
        coordinates
    select_key_ic(ic_indices)
        Select certain internal coordinates to be the key ic which is
        important to chemical reaction
    auto_generate_ts(ratio=0.5,
                     start_with="reactant",
                     reset_ic=False,
                     auto_select=True,
                     mode="mix"):
        Generate transition state structure automatically based on the reactant
        and product structure and corresponding setting parameters
    """

    def __init__(self, reactant_ic, product_ic):
        if isinstance(reactant_ic, Internal) and isinstance(
                product_ic, Internal):
            if np.allclose(reactant_ic.numbers, product_ic.numbers):
                self._numbers = reactant_ic.numbers
                self._reactant = deepcopy(reactant_ic)
                self._product = deepcopy(product_ic)
            else:
                raise AtomsNumberError("The number of atoms is not the same")
        else:
            raise InputTypeError("The type of input data is invalid.")
        self._key_ic_counter = 0
        self._ts = None

    @property
    def reactant(self):
        """Internal coordinates instance of reactant structure

        Returns
        -------
        reactant : Internal
        """
        return self._reactant

    rct = reactant

    @property
    def product(self):
        """Internal coordinates instance of product structure

        Returns
        -------
        product : Internal
        """
        return self._product

    prd = product

    @property
    def ts(self):
        """Internal cooridnates instance of transition structure if
        it has been generated already. Otherwise, raise NotSetError.

        Returns
        -------
        ts : Internal
        """
        if self._ts is None:
            raise NotSetError("TS state hasn't been set")
        return self._ts

    @property
    def numbers(self):
        """A numpy array of atomic number for input coordinates

        Returns
        -------
        numbers : np.ndarray(N,)
        """
        return self._numbers

    @property
    def key_ic_counter(self):
        """Number of key internal coordinates in this reaction

        Returns
        -------
        key_ic_counter : int
        """
        return self._key_ic_counter

    @classmethod
    def from_file(cls, rct_file, prd_file, charge=0, multi=1):
        """Create a TSConstruct instance from files contains info for reactant
        and product

        Arguments
        ---------
        rct_file : str
            path to reactant file
        prd_file : str
            path to product file
        charge : int
            charge of the given system or molecules as a whole
        multi : int
            multiplicity of the given system or molecules as a whole

        Return
        ------
        new TSConstruct instance : TSConstruct
            new instance create from the structure of given reactant and
            product
        """
        rct_mol = Internal.from_file(rct_file, charge, multi)
        prd_mol = Internal.from_file(prd_file, charge, multi)
        return cls(rct_mol, prd_mol)

    def add_bond(self, atom1, atom2):
        """Add bond connection between atom1 and atom2 for both reactant
        and product structure

        Arguments
        ---------
        atom1 : int
            The index of the first atom of a bond
        atom2 : int
            The index of the second atom of a bond
        """
        self._reactant.add_bond(atom1, atom2)
        self._product.add_bond(atom1, atom2)

    def add_angle_cos(self, atom1, atom2, atom3):
        """Add cos angle connection between atom1, atom2, and atom3 for
        both reactant and product structure

        Arguments
        ---------
        atom1 : int
            The index of the first atom of the angle
        atom2 : int
            The index of the second atom of the angle
        atom3 : int
            The index of the third atom of the angle
        """
        self._reactant.add_angle_cos(atom1, atom2, atom3)
        self._product.add_angle_cos(atom1, atom2, atom3)

    def add_angle(self, atom1, atom2, atom3):
        """Add cos angle connection between atom1, atom2, and atom3 for
        both reactant and product structure

        Arguments
        ---------
        atom1 : int
            The index of the first atom of the angle
        atom2 : int
            The index of the second atom of the angle
        atom3 : int
            The index of the third atom of the angle
        """
        self._reactant.add_angle(atom1, atom2, atom3)
        self._product.add_angle(atom1, atom2, atom3)

    def add_dihedral(self, atom1, atom2, atom3, atom4):
        """Add dihedral angle between plane1(atom1, atom2, and atom3)
        and plane2(atom2, atom3, and atom4) for both reactant and
        product structures

        Arguments
        ---------
        atom1 : int
            The index of atom1 in plane1
        atom2 : int
            The index of atom2 in plane1 and plane2
        atom3 : int
            The index of atom3 in plane1 and plane2
        atom4 : int
            The index of atom4 in plane2
        """
        self._reactant.add_dihedral(atom1, atom2, atom3, atom4)
        self._product.add_dihedral(atom1, atom2, atom3, atom4)

    def delete_ic(self, *indices):
        """Delete a exsiting internal cooridnates with given indices(index)

        Arguments
        ---------
        *indices : int
            The index of undesired internal coordinates
        """
        self._reactant.delete_ic(*indices)
        self._product.delete_ic(*indices)

    def auto_select_ic(self,
                       *_,
                       reset_ic=False,
                       auto_select=True,
                       dihed_special=False,
                       mode="mix"):
        """Select internal coordinates for both reactant and product based on
        given structure and choices

        Arguments
        ---------
        reset_ic : bool, default is False
            Remove the existing internal coordinates for reactant and product
        auto_select_ic : bool, default is True
            Use buildin algorithm to auto select proper internal coordinates
        mode : str, default is "mix"
            The way to unify reactant and product internal coordinates
            choices:
                "mix" : choose the union of reactant and product ic
                "reactant" : choose the reactant ic as the template
                "product" : choose the product ic as the template
        """
        if auto_select:
            self._reactant.auto_select_ic(
                reset_ic=reset_ic, dihed_special=dihed_special)
            self._product.auto_select_ic(
                reset_ic=reset_ic, dihed_special=dihed_special)
        target_ic_list = self._get_union_of_ics(mode=mode)
        self._reactant.set_new_ics(target_ic_list)
        self._product.set_new_ics(target_ic_list)

    def create_ts_state(self, start_with, ratio=0.5, task='ts'):
        """Create transition state structure based on the linear combination of
        internal structure of both reactant and product.

        Arguments
        ---------
        start_with : string
            The initial structure of transition state to optimize from.
            choices:
                "reactant" : optimize ts structure from reactant
                "product" : optimize ts structure from product
        ratio : float, default is 0.5
            The ratio of linear combination of ic for reactant and product.
            ts = ratio * reactant + (1 - ratio) * product
        """
        if start_with == "reactant":
            model = self.reactant
        elif start_with == "product":
            model = self.product
        else:
            raise InvalidArgumentError(
                "The input of start_with is not supported")
        if ratio > 1. or ratio < 0:
            raise InvalidArgumentError("The input of ratio is not supported")
        ts_internal = deepcopy(model)
        target_ic = ratio * self.reactant.ic_values + (
            1. - ratio) * self.product.ic_values
        ts_internal.set_target_ic(target_ic)
        ts_internal.converge_to_target_ic()
        if task == 'ts':
            ts_internal = ReducedInternal.update_to_reduced_internal(
                ts_internal)
        elif task == 'path':
            path_vector = self.product.ic_values - self.reactant.ic_values
            ts_internal = PathRI.update_to_path_ri(ts_internal, path_vector)
        # change the ts_internal to Class ReducedInternal
        self._ts = ts_internal  # set _ts attribute

    def select_key_ic(self, *ic_indices):
        """Set one or multiply internal coordinate(s) as the the key internal
        coordinates

        Arguments
        ---------
        *ic_indices : *int
            the index(indices) of internal coordinates
        """
        for index in ic_indices:
            if index >= self._key_ic_counter:
                # if the index is smaller then ic counter
                # it is pointless to swap
                self.ts.swap_internal_coordinates(self._key_ic_counter, index)
                self._key_ic_counter += 1
                self.ts.set_key_ic_number(self._key_ic_counter)

    def auto_generate_ts(self,
                         ratio=0.5,
                         *_,
                         start_with="reactant",
                         reset_ic=False,
                         auto_select=True,
                         dihed_special=False,
                         mode="mix",
                         task='ts'):
        """Complete auto generate transition state structure based on some
        default parameters

        Arguments
        ---------
        ratio : float, default is 0.5
            The ratio of linear combination of ic for reactant and product.
            ts = ratio * reactant + (1 - ratio) * product
        start_with : string, default is "reactant"
            The initial structure of transition state to optimize from.
            choices:
                "reactant" : optimize ts structure from reactant
                "product" : optimize ts structure from product
        reset_ic : bool, default is False
            Remove the existing internal coordinates for reactant and product
        auto_select_ic : bool, default is True
            Use buildin algorithm to auto select proper internal coordinates
        mode : str, default is "mix"
            The way to unify reactant and product internal coordinates
            choices:
                "mix" : choose the union of reactant and product ic
                "reactant" : choose the reactant ic as the template
                "product" : choose the product ic as the template
        """
        self.auto_select_ic(
            reset_ic=False,
            dihed_special=dihed_special,
            auto_select=auto_select,
            mode=mode)
        self.create_ts_state(start_with, ratio, task=task)

    def ts_to_file(self, filename=''):
        if filename:
            Utils.save_file(filename, self.ts)
        else:
            raise InvalidArgumentError('Invalid empty filename')

    def update_rct_and_prd_with_ts(self):
        target_ic = self.ts.ic
        self._reactant.set_new_ics(target_ic)
        self._product.set_new_ics(target_ic)

    def _get_union_of_ics(self, mode='mix'):  # need tests
        """Get the combined internal coordinates based on the ic structure of
        both reactant and product
        """
        if mode == 'mix':
            basic_ic = deepcopy(self._reactant.ic)
            for new_ic in self._product.ic:
                for ic in self._reactant.ic:
                    if new_ic.atoms == ic.atoms and type(new_ic) == type(ic):
                        break
                else:
                    basic_ic.append(new_ic)
            return basic_ic
        elif mode == 'reactant':
            return deepcopy(self._reactant.ic)
        elif mode == 'product':
            return deepcopy(self._product.ic)
        else:
            raise InvalidArgumentError(
                'The argument {} is invalid'.format(mode))
