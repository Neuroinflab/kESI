from _common_new import FourSphereModel, cv, altitude_azimuth_mesh


class DipolarSourcesFactory(object):
    def __init__(self, CONDUCTIVITY, RADIUS):
        self.FSM = FourSphereModel(CONDUCTIVITY,
                                   RADIUS)

    class DipolarSource(object):
        def __init__(self, PS, potential):
            self._PS = PS
            self.potential = potential

        def dipole_moments(self):
            return self._PS

    def __call__(self, R, step):
        DIPOLES = np.array(list(self._dipoles(R, step)))
        sources = [self.DipolarSource(INDEX,
                                      self.FSM(D[3:6], D[6:]))
                   for D, INDEX
                   in zip(DIPOLES, np.eye(DIPOLES.shape[0]))]
        return (pd.DataFrame(DIPOLES,
                             columns=['R', 'ALTITUDE', 'AZIMUTH',
                                      'X', 'Y', 'Z',
                                      'PX', 'PY', 'PZ']),
                sources)


    def _dipoles(self, R, step):
        for altitude, azimuth in altitude_azimuth_mesh(0, step / R):
            y = R * np.sin(altitude)
            r = R * np.cos(altitude)
            x = r * np.sin(azimuth)
            z = r * np.cos(azimuth)
            for P in np.eye(3):
                yield [R, altitude, azimuth, x, y, z] + list(P)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    BRAIN_CONDUCTIVITY = 1. / 300.  # S / cm
    CONDUCTIVITY = FourSphereModel.Properies(1.00 * BRAIN_CONDUCTIVITY,
                                             5.00 * BRAIN_CONDUCTIVITY,
                                             0.05 * BRAIN_CONDUCTIVITY,
                                             1.00 * BRAIN_CONDUCTIVITY)
    RADIUS = FourSphereModel.Properies(7.9, 8.0, 8.5, 9.0)

    import common

    BRAIN_R = RADIUS.brain
    SCALP_R = RADIUS.scalp
    WHITE_R = 7.5
    RAD_TOL = 0.01
    NECK_ANGLE = np.pi / 6 # -np.pi / 3
    RADIANS_PER_ELECTRODE = 7 * np.pi / 180

    ele_coords = []

    for altitude, azimuth in altitude_azimuth_mesh(NECK_ANGLE,
                                                   RADIANS_PER_ELECTRODE):
        r = SCALP_R - RAD_TOL
        ele_coords.append([r * np.cos(altitude) * np.sin(azimuth),
                           r * np.sin(altitude),
                           r * np.cos(altitude) * np.cos(azimuth),
                           r,
                           altitude,
                           azimuth])

    ele_coords = np.array(ele_coords)

    DF = pd.DataFrame(ele_coords, columns=['X', 'Y', 'Z',
                                           'R', 'ALTITUDE', 'AZIMUTH'])
    ELECTRODES = DF[['X', 'Y', 'Z']].copy()
    field = common.FourSphereModel(CONDUCTIVITY,
                                   RADIUS,
                                   ELECTRODES)

    dipolar_source_factory = DipolarSourcesFactory(CONDUCTIVITY, RADIUS)
    DIPOLES, sources = dipolar_source_factory(0.5 * (BRAIN_R + WHITE_R),
                                              2.)
    SRC_DIPOLE = DIPOLES.iloc[324].copy()
    SRC_DIPOLE['PX'] = -1
    SRC_DIPOLE['PY'] = 1
    SRC_DIPOLE['PZ'] = 0.05
    SRC = [SRC_DIPOLE.X + 0.05 * SRC_DIPOLE.PX,
           SRC_DIPOLE.Y + 0.05 * SRC_DIPOLE.PY,
           SRC_DIPOLE.Z + 0.05 * SRC_DIPOLE.PZ,
           ]
    SNK = [SRC_DIPOLE.X - 0.05 * SRC_DIPOLE.PX,
           SRC_DIPOLE.Y - 0.05 * SRC_DIPOLE.PY,
           SRC_DIPOLE.Z - 0.05 * SRC_DIPOLE.PZ,
           ]

    DF['V'] = field.compute_phi(SRC, SNK)

    import matplotlib.pyplot as plt
    import cbf

    def plot_altitude_lines(ax, altitude_max,
                            step=10 * np.pi / 180.,
                            r=1):
        altitude = step
        while altitude <= altitude_max:
            ax.add_artist(plt.Circle((0, 0),
                          radius=r * altitude,
                          ls=':',
                          edgecolor=cbf.BLACK,
                          facecolor='none'))
            altitude += step

    plt.title('Potential')
    plot_altitude_lines(plt.gca(),
                        np.pi / 2 - DF.ALTITUDE.min())
    plt.scatter((np.pi / 2 - DF.ALTITUDE) * np.cos(DF.AZIMUTH),
                (np.pi / 2 - DF.ALTITUDE) * np.sin(DF.AZIMUTH),
                c=DF.V,
                cmap=cbf.PRGn,
                vmin=-abs(DF.V).max(),
                vmax=abs(DF.V).max()
                )
    plt.colorbar()


    import kesi

    class MeasurementManager(kesi.MeasurementManagerBase):
        def __init__(self, ELECTRODES):
            self._ELECTRODES = ELECTRODES

        @property
        def number_of_measurements(self):
            return len(self._ELECTRODES)

        def probe(self, field):
            return field.potential(self._ELECTRODES.X,
                                   self._ELECTRODES.Y,
                                   self._ELECTRODES.Z)

        def load(self, V):
            assert len(self._ELECTRODES) == len(V)
            return V.values


    reconstructor = kesi.FunctionalFieldReconstructor(sources,
                                                      MeasurementManager(ELECTRODES))
    IDX = ~np.isnan(reconstructor._pre_kernel).any(axis=1)
    DIPOLES_VALID = DIPOLES.iloc[IDX]

    plt.figure()
    plt.title('Dipoles')
    plot_altitude_lines(plt.gca(),
                        np.pi / 2 - DIPOLES.ALTITUDE.min())
    if len(DIPOLES_VALID) > 0:
        plt.scatter((np.pi / 2 - DIPOLES_VALID.ALTITUDE) * np.cos(DIPOLES_VALID.AZIMUTH),
                    (np.pi / 2 - DIPOLES_VALID.ALTITUDE) * np.sin(DIPOLES_VALID.AZIMUTH),
                    color=cbf.SKY_BLUE)

    if not IDX.all():
        for (altitude, azimuth), TMP in DIPOLES.iloc[~IDX].groupby(['ALTITUDE',
                                                                    'AZIMUTH']):
            plt.text((np.pi / 2 - altitude) * np.cos(azimuth),
                     (np.pi / 2 - altitude) * np.sin(azimuth),
                     str(len(TMP)),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='bold',
                     color=cbf.VERMILION)

    if not IDX.all():
        plt.figure()
        plt.title('Dipoles vs. electrodes')
        plot_altitude_lines(plt.gca(),
                            np.pi / 2 - min(DIPOLES.ALTITUDE.min(),
                                            DF.ALTITUDE.min()))
        plt.scatter((np.pi / 2 - DIPOLES.ALTITUDE) * np.cos(DIPOLES.AZIMUTH),
                    (np.pi / 2 - DIPOLES.ALTITUDE) * np.sin(DIPOLES.AZIMUTH),
                    color=cbf.BLUE,
                    marker='+')
        plt.scatter((np.pi / 2 - DF.ALTITUDE) * np.cos(DF.AZIMUTH),
                    (np.pi / 2 - DF.ALTITUDE) * np.sin(DF.AZIMUTH),
                    color=cbf.VERMILION,
                    marker='x')

    reconstructor2 = kesi.FunctionalFieldReconstructor([s
                                                        for s, i in zip(sources,
                                                                        IDX)
                                                        if i],
                                                       MeasurementManager(ELECTRODES))

    K = reconstructor2._kernel
    EIGENVALUES = abs(np.linalg.eigvalsh(K))
    LOG_START = np.floor(np.log10(EIGENVALUES.min())) - 2
    LOG_END = np.ceil(np.log10(EIGENVALUES.max())) + 4
    REGULARIZATION_PARAMETERS = [0] + list(np.logspace(LOG_START, LOG_END, 100))
    CV_ERROR = cv(reconstructor2, DF.V, REGULARIZATION_PARAMETERS)
    idx = np.argmin(CV_ERROR)
    regularization_parameter = REGULARIZATION_PARAMETERS[idx]
    plt.figure()
    plt.title('CV')
    plt.xscale('symlog', linthreshx=10**LOG_START)
    plt.axvspan(EIGENVALUES.min(), EIGENVALUES.max(), color=cbf.YELLOW)
    for x in EIGENVALUES:
        plt.axvline(x, ls='--', color=cbf.ORANGE)
    plt.plot(REGULARIZATION_PARAMETERS, CV_ERROR, color=cbf.BLUE)
    plt.axvline(regularization_parameter, ls=':', color=cbf.BLACK)

    for rp in [0, regularization_parameter]:
        approximator = reconstructor2(DF.V,
                                      regularization_parameter=rp)
        DIPOLES['W'] = approximator.dipole_moments()
        DIPOLES['WPX'] = DIPOLES.PX * DIPOLES.W
        DIPOLES['WPY'] = DIPOLES.PY * DIPOLES.W
        DIPOLES['WPZ'] = DIPOLES.PZ * DIPOLES.W


        fig = plt.figure()
        fig.suptitle('$\\lambda = {:g}$'.format(rp))
        ax = plt.subplot(2, 2, 1)
        ZOOM = 2
        ax.set_title('Dipole XY (dipole zoom = {})'.format(ZOOM))
        ax.set_aspect('equal')

        for (R, AL, AZ), ROW in DIPOLES.groupby(['R', 'ALTITUDE', 'AZIMUTH']):
            ax.arrow(R * (np.pi / 2 - AL) * np.cos(AZ),
                     R * (np.pi/2 - AL) * np.sin(AZ),
                     ZOOM * ROW.WPX.sum(),
                     ZOOM * ROW.WPY.sum(),
                     color=cbf.SKY_BLUE)
        ax.arrow(SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.cos(SRC_DIPOLE.AZIMUTH),
                 SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.sin(SRC_DIPOLE.AZIMUTH),
                 ZOOM * SRC_DIPOLE.PX,
                 ZOOM * SRC_DIPOLE.PY,
                 color=cbf.VERMILION,
                 ls=':')

        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)

        ax = plt.subplot(2, 2, 2)
        ZOOM = 2
        ax.set_title('Dipole ZY (dipole zoom = {})'.format(ZOOM))
        ax.set_aspect('equal')
        for (R, AL, AZ), ROW in DIPOLES.groupby(['R', 'ALTITUDE', 'AZIMUTH']):
            ax.arrow(R * (np.pi / 2 - AL) * np.cos(AZ),
                     R * (np.pi / 2 - AL) * np.sin(AZ),
                     ZOOM * ROW.WPZ.sum(),
                     ZOOM * ROW.WPY.sum(),
                     color=cbf.SKY_BLUE)

        ax.arrow(SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.cos(
            SRC_DIPOLE.AZIMUTH),
                 SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.sin(
                     SRC_DIPOLE.AZIMUTH),
                 ZOOM * SRC_DIPOLE.PZ,
                 ZOOM * SRC_DIPOLE.PY,
                 color=cbf.VERMILION,
                 ls=':')

        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)

        ax = plt.subplot(2, 2, 3)
        ZOOM = 2
        ax.set_title('Dipole XZ (dipole zoom = {})'.format(ZOOM))
        ax.set_aspect('equal')

        for (R, AL, AZ), ROW in DIPOLES.groupby(['R', 'ALTITUDE', 'AZIMUTH']):
            ax.arrow(R * (np.pi / 2 - AL) * np.cos(AZ),
                     R * (np.pi / 2 - AL) * np.sin(AZ),
                     ZOOM * ROW.WPX.sum(),
                     ZOOM * ROW.WPZ.sum(),
                     color=cbf.SKY_BLUE)

        ax.arrow(SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.cos(
            SRC_DIPOLE.AZIMUTH),
                 SRC_DIPOLE.R * (np.pi / 2 - SRC_DIPOLE.ALTITUDE) * np.sin(
                     SRC_DIPOLE.AZIMUTH),
                 ZOOM * SRC_DIPOLE.PX,
                 ZOOM * SRC_DIPOLE.PZ,
                 color=cbf.VERMILION,
                 ls=':')

        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)

    plt.show()
