from saddle.errors import NotSetError

class PathPoint:

    def __init__(self, red_int):
        self._instance = red_int
        self._step = None
        self._stepsize = None
        self._mod_hessian = None

    @property
    def energy(self):
        return self._instance.energy

    @property
    def x_gradient(self):
        return self._instance.energy_gradient

    @property
    def x_hessian(self):
        return self._instance.energy_hessian

    @property
    def b_matrix(self):
        return self._instance.b_matrix

    @property
    def q_gradient(self):
        return self._instance.q_gradient

    # @property
    # def q_hessian(self):
    #     return self._instance.q_hessian

    @property
    def vspace(self):
        return self._instance.vspace

    @property
    def v_gradient(self):
        return self._instance.v_gradient

    @property
    def v_hessian(self):
        if self._mod_hessian:
            return self._mod_hessian
        return self.raw_hessian

    @property
    def df(self):
        return self.df

    @property
    def raw_hessian(self):
        return self._instance.v_hessian

    @property
    def step(self):
        if self._step:
            return self._step
        raise NotSetError

    @step.setter
    def step(self, value):
        assert value > 0
        self._step = value

    @property
    def stepsize(self):
        if self._stepsize:
            return self._stepsize
        raise NotSetError

    def __repr__(self):
        return f"PathPoint object"
