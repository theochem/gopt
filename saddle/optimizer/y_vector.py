"""A module to implement different kind of y-vectors."""


class Point:
    """Data class to storing data for each points."""
    def __init__(self, coords, value=None, gradient=None, hessian=None, step=None):
        self.coords = coords
        self.value = value
        self.gradient = gradient
        self.hessian = hessian
        self.step = step


class YVector:
    """Compute different y vector formula."""

    _method_dict = {}

    def compute_y_vector(new_point, old_point, method=None, **kwargs):
        return YVector._method_dict[method](new_point, old_point, **kwargs)

    def _compute_y_vector_method_name(*args, **kwargs):
        ...
