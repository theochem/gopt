from pyscf import gto

class gobasis(object):
    def __init__(self, coordinates, numbers, default, spin=0,
                 basis2=None, element_map=None,
                 index_map=None, pure=True):
        atoms = [(i, j) for i, j in zip(numbers, coordinates)]

        if index_map and element_map:
            print "Can't specify both element map and index map."
        elif index_map:
            basis = {e: index_map[i] if i in index_map
            else default
                     for i, e in enumerate(numbers)}
        elif element_map:
            basis = {e: element_map[e] if e in index_map
            else default
                     for e in numbers}
        else:
            basis = default

        if (element_map or index_map) and basis2:
            print "Basis set concatenation is not supported with element_map " \
                  "or index_map."
            raise NotImplementedError
        elif basis2:
            self.mol2 = gto.M(atom=atoms, basis=basis2, spin=spin)

        if not pure:
            print "Basis functions are always cartesian"
            # TODO: test to see what horton functionality is
            raise NotImplementedError

        self.mol = gto.M(atom=atoms, basis=basis, spin=spin)

    @property
    def nbasis(self):
        return self.mol.nao_nr()

    @property
    def naux(self):
        return self.mol2.nao_nr()

    def compute_overlap(self, output):
        """Compute the overlap integrals in a Gaussian orbital basis

           **Arguments:**

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_ovlp_sph", output._array)
        return output

    def compute_overlap_gradient(self, output):
        """Compute the energy gradient of the overlap integrals in a Gaussian
        orbital basis

           **Arguments:**

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_ipovlp_sph", output._array)
        return output

    def compute_kinetic(self, output):
        """Compute the kinetic energy integrals in a Gaussian orbital basis

           **Arguments:**

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_kin_sph", output._array)
        return output

    def compute_kinetic_gradient(self, output):
        """Compute the energy gradient of the kinetic energy integrals in a
        Gaussian orbital basis

           **Arguments:**

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_ipkin_sph", output._array)
        return output

    def compute_nuclear_attraction(self, dummy1, dummy2, output):
        """Compute the nuclear attraction integral in a Gaussian orbital basis

           **Arguments:**

           dummy1
                Kept to maintain compatibility with other class. No
                functionality. Set the external field coordinates and charge
                in the constructor.

           dummy2
                Kept to maintain signature compatibility with other class. No
                functionality. Set charges in the constructor for gobasis.

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_nuc_sph", output._array)
        return output

    def compute_nuclear_attraction_gradient(self, dummy1, dummy2, output):
        """Compute the energy gradient of the nuclear attraction integral in a
        Gaussian orbital basis

           **Arguments:**

           dummy1
                Kept to maintain compatibility with other class. No
                functionality. Set the external field coordinates and charge
                in the constructor.

           dummy2
                Kept to maintain signature compatibility with other class. No
                functionality. Set charges in the constructor for gobasis.

           output
                When a ``TwoIndex`` instance is given, it is used as output
                argument and its contents are overwritten. When ``LinalgFactory``
                is given, it is used to construct the output ``TwoIndex``
                object. In both cases, the output two-index object is returned.

           **Returns:** ``TwoIndex`` object
        """
        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_two_index(self.nbasis)
        self._get_cint1e("cint1e_ipnuc_sph", output._array)
        return output

    def compute_electron_repulsion(self, output):
        '''Compute electron-electron repulsion integrals

           **Argument:**

           output
                When a ``DenseFourIndex`` object is given, it is used as output
                argument and its contents are overwritten. When a
                ``DenseLinalgFactory`` or ``CholeskyLinalgFactory`` is given, it
                is used to construct the four-index object in which the
                integrals are stored.

           **Returns:** The four-index object with the electron repulsion
           integrals.

           Keywords: :index:`ERI`, :index:`four-center integrals`
        '''
        # FIXME: restore log behaviour
        # log.cite('valeev2014', 'the efficient implementation of four-center electron repulsion integrals')

        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_four_index(self.nbasis)
        self._get_cint2e("cint2e_sph", output._array)
        return output

    def compute_electron_repulsion_gradient(self, output):
        '''Compute electron-electron repulsion integrals

           **Argument:**

           output
                When a ``DenseFourIndex`` object is given, it is used as output
                argument and its contents are overwritten. When a
                ``DenseLinalgFactory`` or ``CholeskyLinalgFactory`` is given, it
                is used to construct the four-index object in which the
                integrals are stored.

           **Returns:** The four-index object with the electron repulsion
           integrals.

           Keywords: :index:`ERI`, :index:`four-center integrals`
        '''
        # FIXME: restore log behaviour
        # log.cite('valeev2014', 'the efficient implementation of four-center electron repulsion integrals')

        if isinstance(output, LinalgFactory):
            lf = output
            output = lf.create_four_index(self.nbasis)
        self._get_cint2e("cint2e_ip1_sph", output._array)
        return output

    def compute_electron_repulsion_df(self, output1, output2):
        '''Compute electron-electron repulsion integrals

           **Argument:**

           output1, output2
                When a ``DenseFourIndex`` object is given, it is used as output
                argument and its contents are overwritten. When a
                ``DenseLinalgFactory`` or ``CholeskyLinalgFactory`` is given, it
                is used to construct the four-index object in which the
                integrals are stored.

           **Returns:** The four-index object with the electron repulsion
           integrals.

           Keywords: :index:`ERI`, :index:`four-center integrals`
        '''
        if self.mol2 is None:
            print "Auxiliary basis set wasn't provided for gobasis."
            raise AttributeError

        if isinstance(output1, LinalgFactory):
            lf = output1
            output1 = lf.create_three_index(self.nbasis, self.nbasis,
                                            self.naux)
        self._get_cint3c2e("cint3c2e_sph", output1._array)

        if isinstance(output2, LinalgFactory):
            lf = output2
            output2 = lf.create_two_index(self.naux, self.naux)
        self._get_cint2c2e("cint2c2e_sph", output2._array)

        return output1, output2

    def _get_cint2e(self, int_name, output):
        nao = self.mol.nao_nr()
        assert output.shape == (nao, nao, nao, nao)
        pi = 0
        for i in xrange(self.mol.nbas):
            pj = 0
            for j in xrange(self.mol.nbas):
                pk = 0
                for k in xrange(self.mol.nbas):
                    pl = 0
                    for l in xrange(self.mol.nbas):
                        shells = (i, j, k, l)  # FIXME: chemist's notation
                        buf = gto.getints_by_shell(int_name, shells,
                                                   self.mol._atm,
                                                   self.mol._bas, self.mol._env)
                        di, dj, dk, dl = buf.shape
                        output[pi:pi + di,
                        pj:pj + dj,
                        pk:pk + dk,
                        pl:pl + dl] = buf
                        pl += dl
                    pk += dk
                pj += dj
            pi += di

        return output

    def _get_cint1e(self, int_name, output):
        nao = self.mol.nao_nr()
        assert output.shape == (nao, nao)
        pk = 0
        for k in xrange(self.mol.nbas):
            pl = 0
            for l in xrange(self.mol.nbas):
                shells = (k, l)
                buf = gto.getints_by_shell(int_name, shells, self.mol._atm,
                                           self.mol._bas, self.mol._env)
                dk, dl = buf.shape
                output[pk:pk + dk, pl:pl + dl] = buf
                pl += dl
            pk += dk

        return output

    def _get_cint3c2e(self, int_name, output):
        atm, bas, env = gto.conc_env(self.mol._atm, self.mol._bas,
                                     self.mol._env, self.mol2._atm,
                                     self.mol2._bas, self.mol2._env)

        nao = self.mol.nao_nr()
        naux = self.mol2.nao_nr()
        assert output.shape == (nao, nao, naux)
        pi = 0
        for i in range(self.mol.nbas):
            pj = 0
            for j in range(self.mol.nbas):
                pk = 0
                for k in range(self.mol.nbas,
                               self.mol.nbas + self.mol2.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell(int_name, shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    output[pi:pi + di, pj:pj + dj, pk:pk + dk] = buf
                    pk += dk
                pj += dj
            pi += di
        return output

    def _get_cint2c2e(self, int_name, output):
        atm, bas, env = gto.conc_env(self.mol._atm, self.mol._bas,
                                     self.mol._env, self.mol2._atm,
                                     self.mol2._bas, self.mol2._env)

        naux = self.mol2.nao_nr()
        assert output.shape == (naux, naux)
        pk = 0
        for k in range(self.mol.nbas, self.mol.nbas + self.mol2.nbas):
            pl = 0
            for l in range(self.mol.nbas, self.mol.nbas + self.mol2.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell(int_name, shls, atm, bas, env)
                dk, dl = buf.shape
                output[pk:pk + dk, pl:pl + dl] = buf
                pl += dl
            pk += dk
        return output