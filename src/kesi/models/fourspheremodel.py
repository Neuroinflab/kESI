import collections
import configparser

import numpy as np

from kesi.models.foursphere_dipolar_model import PointDipole
from kesi.models.foursphere_monopolar_model import PointMonopole


class FourSphereModel(object):
    """
    Based on https://github.com/Neuroinflab/fourspheremodel
    by Chaitanya Chintaluri
    """
    _LAYERS = ['brain',
               'csf',
               'skull',
               'scalp',
               ]

    class Properties(collections.namedtuple('FourSpheres',
                                            _LAYERS)):
        @classmethod
        def from_config(cls, path, field):
            config = configparser.ConfigParser()
            config.read(path)
            return cls(*[config.getfloat(section, field)
                         for section in FourSphereModel._LAYERS])

    def __init__(self, conductivity, radius, n=100, precision="float64"):
        assert precision in ["float64", "float128"]
        self.precision = precision
        self.n = np.arange(1, n)
        self._set_radii(radius)
        self._set_conductivities(conductivity)

    @classmethod
    def from_config(cls, path, n=100):
        return cls(cls.Properties.from_config(path, 'conductivity'),
                   cls.Properties.from_config(path, 'radius'),
                   n)

    def _set_radii(self, radius):
        self.radius = radius
        self.r12 = radius.brain / radius.csf
        self.r23 = radius.csf / radius.skull
        self.r34 = radius.skull / radius.scalp
        self.r21 = 1. / self.r12
        self.r32 = 1. / self.r23
        self.r43 = 1. / self.r34

    def _set_conductivities(self, conductivity):
        self.conductivity = conductivity
        self.s12 = conductivity.brain / conductivity.csf
        self.s23 = conductivity.csf / conductivity.skull
        self.s34 = conductivity.skull / conductivity.scalp

    def V(self, n):
        try:
            return self._V
        except AttributeError:
            n = self.n

        k = (n + 1.) / n
        Factor = ((self.r34 ** n - self.r43 ** (n + 1))
                  / (k * self.r34 ** n + self.r43 ** (n + 1)))
        num = self.s34 / k - Factor
        den = self.s34 + Factor
        self._V = num / den
        return self._V

    def Y(self, n):
        try:
            return self._Y
        except AttributeError:
            n = self.n

        k = n / (n + 1.)
        V_n = self.V(n)
        r23n = self.r23 ** n
        r32n1 = self.r32 ** (n + 1)
        Factor = ((r23n * k - V_n * r32n1)
                  / (r23n + V_n * r32n1))
        self._Y = (self.s23 * k - Factor) / (self.s23 + Factor)
        return self._Y

    def Z(self, n):
        try:
            return self._Z
        except AttributeError:
            n = self.n

        k = (n + 1.) / n
        Y_n = self.Y(n)
        r12n = self.r12 ** n
        r21n1 = self.r21 ** (n + 1)
        self._Z = (r12n - k * Y_n * r21n1) / (r12n + Y_n * r21n1)
        return self._Z

    def __call__(self, loc, P):
        return PointDipole(self,
                                 np.reshape(loc,
                                            (1, 3)),
                                 np.reshape(P,
                                            (1, 3)))

    def get_dipole_model(self, loc, P):
        return self(loc, P)

    def get_monopolar_model(self, loc, A):
        return PointMonopole(loc, np.reshape(loc,
                                            (1, 3)), A)
