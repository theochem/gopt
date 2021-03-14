"""A module to implement different kind of y-vectors."""


class Point:
    """Data class to storing data for each points."""

    def __init__(
        self, coords, value=None, gradient=None, hessian=None, step=None, y_vector=None
    ):
        self.coords = coords
        self.value = value
        self.gradient = gradient
        self.hessian = hessian
        self.step = step
        self.y_vector = y_vector


class YVector:
    """Compute different y vector formula."""

    _method_dict = {"name": YVector._compute_y_vector_method_name}

    def compute_y_vector(self, points, method=None, **kwargs):
        # points is a list of points []
        return YVector._method_dict[method](points, old_point, **kwargs)

    @staticmethod
    def _compute_y_vector_method_name(points, **kwargs):
        ...

    @staticmethod
    def _y_vector_method_name_low(args):
        ...
