import numpy as np


class NSphereModel:
    """Class to calculate point current sources in N conductive spheres, with outer shell having a boundary condition of zero current going through it"""
    def __init__(self, radii=(0.079, 0.082, 0.086, 0.09, 100.0),
                 conductivities=(0.33, 1.65, 0.0165, 0.33, 1e-10)):
        """
        Params:
        :param radii: - iterable of sphere radii, in meters, must be sorted
        :param conductivities: - iterable of shpere conductivities from most inner to outer in S/m
        """
        assert len(radii) == len(conductivities)
        assert list(sorted(radii)) == list(radii)

        self.radii = np.array(radii)
        self.conductivities = np.array(conductivities)