def direct_square(origin, target):
    value = (origin - target) ** 2
    deriv = 2 * (origin - target)
    deriv2 = 2
    return value, deriv, deriv2
