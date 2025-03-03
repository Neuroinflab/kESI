import numpy as np

from kesi.models.foursphere_monopolar_model import PointMonopole


class NShereModel(object):

    def __init__(self, conductivity, radius, n=100, precision="float64"):
        """Creates a four sphere analytical model of point dipole or monopole in conductive medium.

        Parameters:
            conductivity - array of N floats, conductivity values of the concentric spherical shells, in S/m
            radius  - array of N floats, radii of the shells
            n - int, order of the mode, how many therms of the expansion used for computations
            precision - string, "float64" or "float128" - precision of the computation, 128 needed for big spheres
                in orders of hundred meters, slows computation time significantly.
        """
        assert precision in ["float64", "float128"]
        self.precision = precision
        self.n = np.arange(1, n)
        self.radius = radius
        self.conductivity = conductivity
        assert len(self.radius ) == len(self.conductivity)
        assert len(self.radius) > 2

    def get_monopolar_model(self, loc, A):
        """
        Get the point source model, which you can then query by calling.

        Parameters:
            loc - array of shape (3, ) - x,y,z of the point source, must be inside the first shell. In meters.
            A - float - amplitude of the point source, in Ampers.
        """
        return PointMonopole(self, np.reshape(loc,
                                              (1, 3)), A)