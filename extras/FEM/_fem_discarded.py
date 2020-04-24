import numpy as np

from FEM import _fem_common as fc
from FEM.fem_sphere_gaussian import _SomeSphereGaussianPotentialLoaderBase, _SourceBase, _SomeSphereControllerBase


class _FixedElectrodesGaussianLoaderBase(_SomeSphereGaussianPotentialLoaderBase):
    ATTRIBUTES = _SomeSphereGaussianPotentialLoaderBase.ATTRIBUTES + ['ELECTRODES']
    _REGISTRATION_RADIUS = 0.079
    _RADIUS = 73.21604532
    _CENTER = np.array([[82.40997559, 118.14496578, 104.73314426]])
    _COORDS_NW_LATER = np.array([[139.667, 154.115, 82.4576],
                                 [142.692, 154.864, 89.9479],
                                 [144.815, 154.772, 99.6451],
                                 [146.533, 154.306, 109.924],
                                 [146.793, 152.317, 119.917],
                                 [145.519, 149.51, 129.734],
                                 [142.5, 145.33, 139.028],
                                 [138.167, 139.983, 146.762],
                                 [134.107, 162.626, 81.968],
                                 [137.267, 163.09, 91.0938],
                                 [139.0, 162.752, 101.274],
                                 [140.269, 161.607, 111.542],
                                 [140.621, 159.943, 121.695],
                                 [139.28, 157.254, 131.404],
                                 [136.56, 152.946, 140.479],
                                 [132.636, 148.073, 148.438],
                                 [128.5, 170.806, 82.8056],
                                 [130.92, 170.867, 92.3083],
                                 [132.583, 170.061, 102.457],
                                 [133.44, 168.896, 112.796],
                                 [133.667, 167.244, 123.129],
                                 [132.519, 164.448, 133.341],
                                 [130.0, 160.417, 142.392],
                                 [125.76, 155.421, 150.013],
                                 [122.769, 178.133, 83.3814],
                                 [124.414, 177.565, 93.4052],
                                 [125.429, 176.458, 103.698],
                                 [126.433, 175.274, 114.01],
                                 [126.435, 173.27, 124.348],
                                 [125.696, 170.729, 134.461],
                                 [123.5, 167.063, 143.401],
                                 [119.111, 161.944, 150.995],
                                 [115.286, 183.415, 84.5052],
                                 [116.385, 182.716, 94.5753],
                                 [117.778, 182.103, 104.61],
                                 [118.5, 180.673, 115.402],
                                 [118.125, 178.511, 125.278],
                                 [117.25, 175.265, 135.161],
                                 [115.778, 170.284, 144.184],
                                 [113.409, 163.49, 151.354],
                                 [106.5, 186.847, 85.5174],
                                 [107.769, 186.418, 95.8093],
                                 [109.304, 186.073, 105.915],
                                 [109.667, 184.267, 116.196],
                                 [109.143, 181.696, 126.124],
                                 [108.346, 178.001, 135.869],
                                 [106.455, 173.021, 144.583],
                                 [104.522, 166.893, 152.495],
                                 [100.471, 149.902, 112.561],
                                 [103.9, 153.427, 117.047],
                                 [107.062, 156.549, 121.387],
                                 [109.941, 159.473, 125.435],
                                 [113.077, 162.179, 129.744],
                                 [115.929, 164.509, 133.653],
                                 [118.2, 166.681, 137.424],
                                 [120.077, 168.966, 141.202],
                                 [106.7, 140.594, 112.292],
                                 [107.8, 146.743, 114.049],
                                 [108.588, 152.598, 115.729],
                                 [109.385, 158.389, 117.228],
                                 [110.0, 163.663, 118.498],
                                 [110.4, 168.667, 119.319],
                                 [110.0, 173.462, 120.841],
                                 [109.0, 177.5, 123.229],
                                 [99.4412, 139.926, 103.226],
                                 [95.9286, 148.44, 106.917],
                                 [92.4615, 156.715, 110.617],
                                 [90.4359, 164.794, 114.143],
                                 [89.8, 172.235, 117.497],
                                 [91.5625, 178.643, 120.85],
                                 [102.893, 152.314, 93.7946],
                                 [100.125, 159.939, 88.5634],
                                 [96.5769, 166.587, 83.4696],
                                 [94.9565, 174.017, 78.8632],
                                 [97.25, 181.778, 77.0573],
                                 [102.5, 187.076, 78.8333],
                                 [89.0, 173.479, 99.9167],
                                 [89.3333, 172.512, 90.0116],
                                 [93.8333, 172.352, 83.1684],
                                 [102.125, 172.591, 75.3385],
                                 [109.0, 174.658, 71.3691],
                                 [118.8, 176.917, 70.4688]])
    ELECTRODES = (_COORDS_NW_LATER - _CENTER) / _RADIUS * _REGISTRATION_RADIUS

    def _load(self):
        super(_FixedElectrodesGaussianLoaderBase, self)._load()
        sd = self.standard_deviation

        self.ALTITUDE = []
        self.AZIMUTH = []
        for i, altitude in enumerate(
                             np.linspace(
                                  0,
                                  np.pi / 2,
                                  int(np.ceil(self.source_resolution * self.cortex_radius_external * np.pi / 2 / sd)) + 1)):
            for azimuth in np.linspace(0 if i % 2 else 2 * np.pi,
                                       2 * np.pi if i % 2 else 0,
                                       int(np.ceil(self.source_resolution * self.cortex_radius_external * np.cos(altitude) * np.pi * 2 / sd)) + 1)[:-1]:
                self.ALTITUDE.append(altitude)
                self.AZIMUTH.append(azimuth)


class SomeSphereFixedElectrodesGaussianSourceFactory(_FixedElectrodesGaussianLoaderBase):
    def __init__(self, filename):
        self.path = filename
        self._load()
        self._r_index = {r: i for i, r in enumerate(self.R)}
        self._altitude_azimuth_index = {coords: i
                                        for i, coords
                                        in enumerate(zip(self.ALTITUDE,
                                                         self.AZIMUTH))}

    def _provide_attributes(self):
        self._load_attributes()

    def __call__(self, r, altitude, azimuth):
        i_r = self._r_index[r]
        i_aa = self._altitude_azimuth_index[altitude, azimuth]
        POTENTIAL = self.POTENTIAL[i_r, i_aa]
        a = self.A[i_r]
        return self._Source(r, altitude, azimuth, a, POTENTIAL, self)

    class _Source(_SourceBase):
        def __init__(self, r, altitude, azimuth, a, POTENTIAL, parent):
            self._r = r
            self._altitude = altitude
            self._azimuth = azimuth
            r2 = r * np.cos(altitude)
            self._x = r2 * np.cos(azimuth)
            self._y = r * np.sin(altitude)
            self._z = -r2 * np.sin(azimuth)
            self._a = a
            self.parent = parent
            self._POTENTIAL = POTENTIAL

        def potential(self):
            return self._POTENTIAL


class _SomeSphereFixedElectrodesGaussianController(
    _FixedElectrodesGaussianLoaderBase,
          _SomeSphereControllerBase):
    @property
    def path(self):
        fn = '{0._fem.mesh_name}_gaussian_{1:04d}_deg_{0.degree}.npz'.format(
                   self,
                   int(round(1000 / 2 ** self.k)))

        return fc._SourceFactory_Base.solution_path(fn, False)

    def _empty_solutions(self):
        super(_SomeSphereFixedElectrodesGaussianController,
              self)._empty_solutions()
        n = 2 ** self.k
        self.STATS = []
        self.POTENTIAL = fc.empty_array((n * self.source_resolution,
                                        len(self.AZIMUTH),
                                        len(self.ELECTRODES)))
