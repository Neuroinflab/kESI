import os.path
import warnings

import numpy as np
from scipy.integrate import romb

from kesi import Reconstructor
from kesi.common import SphericalSplineSourceKCSD, GaussianSourceKCSD3D, cv
from kesi.kernel.constructor import Convolver, ConvolverInterfaceIndexed, KernelConstructor, CrossKernelConstructor
from kesi.kernel.electrode import Conductivity
from kesi.kernel import potential_basis_functions as pbf
from kesi.kernel.electrode import LinearlyInterpolatedLeadfieldCorrection, NearestNeighbourInterpolatedLeadfieldCorrection


class KcsdKesi3d:
    def __init__(self, estimation_points_grid, positions, conductivity=1.0, R_init=1.0, mask=None, source_type='spherical'):
        """
        estimation_points_grid: list of fleshed out meshgrids X, Y, Z, which span the CSD estimation, even grid spacing along all axii
        positions: electrode positions same units as grid, 2D numpy array (N, 3)
        conductivity: tissue conductivity, isotropic and even along the whole universe, defaults to 1 S/m
        R_init: radius of the sources
        mask: None for no mask, 3D tensor of bools to put sources along estimation_points_grids nodes
        source_type: Type of the CSD sources, spherical or gaussian
        """
        assert source_type in ['spherical', 'gaussian']
        self.positions = positions
        if mask is None:
            mask = np.ones_like(estimation_points_grid[0], dtype=bool)

        sim_space_step = np.abs(estimation_points_grid[0][0, 0, 0] - estimation_points_grid[0][1, 0, 0])

        electrodes = [Conductivity(i[0], i[1], i[2], conductivity) for i in positions]

        if source_type == 'spherical':
            spline_nodes = [R_init / 3, R_init]
            spline_polynomials = [[1],
                                  [0,
                                   6.75 / R_init,
                                   -13.5 / R_init ** 2,
                                   6.75 / R_init ** 3]]
            model_src = SphericalSplineSourceKCSD(0, 0, 0,
                                                  spline_nodes,
                                                  spline_polynomials)
        elif source_type == 'gaussian':
            model_src = GaussianSourceKCSD3D(0, 0, 0, R_init, conductivity=conductivity)
        else:
            NotImplemented("Unsupported source type {}".format(source_type))

        # only works with non rotated affines!!!!!!
        x = estimation_points_grid[0][:, 0, 0]
        y = estimation_points_grid[1][0, :, 0]
        z = estimation_points_grid[2][0, 0, :]

        pot_grid = [x, y, z]
        csd_grid = [x, y, z]

        convolver = Convolver(pot_grid, csd_grid)
        self.convolver = convolver

        # romberg weights define a kernel created around source placed at electrode
        # for analytical kernels, we just need to have a ROMBRRG_N * dx to span significant portion of the kernel...
        # I guess for numerically corrected it needs to be as big as possible to include numerical corrections across the brain?
        # most likely it just decays differently

        x = np.linspace(-R_init * 100, R_init * 100, 100000)
        csd = model_src.csd(x, np.zeros_like(x), np.zeros_like(x))

        # we assume that source is symmetrical...
        cutoff = csd.max() * 1.0e-4
        effective_source_radius = np.abs(x[np.argmax(csd >= cutoff)])

        source_size_in_grid = int(effective_source_radius / sim_space_step * 2)
        if source_size_in_grid < 2:
            warnings.warn("Source size is smaller than step, are you sure it's intentional?")
            minimum_romberg_k = 2
        else:
            minimum_romberg_k = int(np.ceil(np.log(source_size_in_grid - 1) / np.log(2)))

        romberg_n = 2 ** minimum_romberg_k + 1
        # ROMBERG_WEIGHTS = romb(np.identity(ROMBERG_N),
        #                        dx=2 ** -ROMBERG_K)

        ROMBERG_WEIGHTS = romb(np.identity(romberg_n),
                               sim_space_step)

        convolver_interface = ConvolverInterfaceIndexed(convolver,
                                                        model_src.csd,
                                                        ROMBERG_WEIGHTS,
                                                        mask)

        pbf_kcsd = pbf.Analytical(convolver_interface,
                                  potential=model_src.potential)

        kernel_constructor = KernelConstructor()

        CSD_MASK = np.ones(convolver.shape('CSD'),
                           dtype=bool)

        kernel_constructor.crosskernel = CrossKernelConstructor(convolver_interface,
                                                                CSD_MASK)

        B_KCSD = kernel_constructor.potential_basis_functions_at_electrodes(electrodes,
                                                                            pbf_kcsd)
        KERNEL_KCSD = kernel_constructor.kernel(B_KCSD)
        CROSSKERNEL_KCSD = kernel_constructor.crosskernel(B_KCSD)
        del B_KCSD  # the array is large and no longer needed

        reconstructor_kcsd = Reconstructor(KERNEL_KCSD,
                                           CROSSKERNEL_KCSD)
        self.reconstructor = reconstructor_kcsd

    def reconstruct_csd(self, potential, regularization_parameter=0):
        """
        potential: 2D array of potential at electrode positions given in constructor [electrodes x samples]
        regularisation_parameter: lambda for CSD regularisation

        returns 3D CSD reconstruction across the grid passed in the constructor
        """
        assert len(potential.shape) == 2
        assert potential.shape[0] == len(self.positions)

        csd = self.reconstructor(potential, regularization_parameter=regularization_parameter)
        csd_3d = csd.reshape(list(self.convolver.shape('CSD')) + [csd.shape[1]])
        return csd_3d

    def cv_lambda(self, potential, lambd=None):
        """
        potential: 2D array of potential at electrode positions given in constructor [electrodes x samples]
        lambd: regularisation lambdas to check

        returns: best lambda, checked lambdas, error values per lambda
        """

        assert len(potential.shape) == 2
        assert potential.shape[0] == len(self.positions)

        if lambd is None:
            lambd = np.logspace(-20, 20, 1000)

        errors = np.array(cv(self.reconstructor, potential, lambd)).flatten()
        best_lambda = np.array(lambd[np.argmin(errors)]).flatten()
        return best_lambda, lambd, errors

    def eigh(self, lambd=0):
        """
        count eigenvalues an eigenvectors for KcsdKesi3d
        """
        kernel = self.reconstructor._solve_kernel._kernel
        eigenvalues, eigenvectors = np.linalg.eigh(kernel + lambd * np.identity(kernel.shape[0]))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        cross_kernel = self.reconstructor._cross_kernel
        eigensources = np.dot(cross_kernel, eigenvectors)
        eigensources_grid = eigensources.reshape(list(self.convolver.csd_shape) + [-1, ])
        eigensources = eigensources_grid

        return eigenvalues, eigensources


class Kesi3d(KcsdKesi3d):
    def __init__(self, estimation_points_grid, electrode_names, electrode_folder, conductivity=1.0, R_init=1.0,
                 mask=None, source_type='spherical', interpolation='linear'):
        """
        estimation_points_grid: list of fleshed out meshgrids X, Y, Z, which span the CSD estimation, even grid spacing along all axii
        electrode_names - list of electrode names to use from electrode folder
        electrode_folder - str, path to folder with sampled electrode corrections
        conductivity: tissue conductivity, isotropic and even along the whole universe, defaults to 1 S/m
        R_init: radius of the sources
        mask: None for no mask, 3D tensor of bools to put sources along estimation_points_grids nodes
        source_type: Type of the CSD sources, spherical or gaussian
        interpolation: str, "linear" or "nearest" - interpolator to map correction potentials to estimation points grid
            nearest is around 2 times faster
        """
        assert source_type in ['spherical', 'gaussian']
        if mask is None:
            mask = np.ones_like(estimation_points_grid[0], dtype=bool)

        sim_space_step = np.abs(estimation_points_grid[0][0, 0, 0] - estimation_points_grid[0][1, 0, 0])

        electrodes = []
        positions = []
        for el_name in electrode_names:
            el_path = os.path.join(electrode_folder, el_name + '.npz')
            if interpolation == 'linear':
                electrode = LinearlyInterpolatedLeadfieldCorrection(el_path)
            elif interpolation == 'nearest':
                electrode = NearestNeighbourInterpolatedLeadfieldCorrection(el_path)
            else:
                raise NotImplementedError("Interpolator not implemented: {}".format(interpolation))
            electrodes.append(electrode)
            positions.append([electrode.x, electrode.y, electrode.z])

        self.positions = np.array(positions)

        if source_type == 'spherical':
            spline_nodes = [R_init / 3, R_init]
            spline_polynomials = [[1],
                                  [0,
                                   6.75 / R_init,
                                   -13.5 / R_init ** 2,
                                   6.75 / R_init ** 3]]
            model_src = SphericalSplineSourceKCSD(0, 0, 0,
                                                  spline_nodes,
                                                  spline_polynomials)
        elif source_type == 'gaussian':
            model_src = GaussianSourceKCSD3D(0, 0, 0, R_init, conductivity=conductivity)
        else:
            NotImplemented("Unsupported source type {}".format(source_type))

        # only works with non rotated affines!!!!!!
        x = estimation_points_grid[0][:, 0, 0]
        y = estimation_points_grid[1][0, :, 0]
        z = estimation_points_grid[2][0, 0, :]

        pot_grid = [x, y, z]
        csd_grid = [x, y, z]

        convolver = Convolver(pot_grid, csd_grid)
        self.convolver = convolver

        # romberg weights define a kernel created around source placed at electrode
        # for analytical kernels, we just need to have a ROMBRRG_N * dx to span significant portion of the kernel...
        # I guess for numerically corrected it needs to be as big as possible to include numerical corrections across the brain?
        # most likely it just decays differently

        x = np.linspace(-R_init * 100, R_init * 100, 100000)
        csd = model_src.csd(x, np.zeros_like(x), np.zeros_like(x))

        # we assume that source is symmetrical...
        cutoff = csd.max() * 1.0e-4
        effective_source_radius = np.abs(x[np.argmax(csd >= cutoff)])

        source_size_in_grid = int(effective_source_radius / sim_space_step * 2)
        if source_size_in_grid < 2:
            warnings.warn("Source size is smaller than step, are you sure it's intentional?")
            minimum_romberg_k = 2
        else:
            minimum_romberg_k = int(np.ceil(np.log(source_size_in_grid - 1) / np.log(2)))

        romberg_n = 2 ** minimum_romberg_k + 1
        # ROMBERG_WEIGHTS = romb(np.identity(ROMBERG_N),
        #                        dx=2 ** -ROMBERG_K)

        ROMBERG_WEIGHTS = romb(np.identity(romberg_n),
                               sim_space_step)

        convolver_interface = ConvolverInterfaceIndexed(convolver,
                                                        model_src.csd,
                                                        ROMBERG_WEIGHTS,
                                                        mask)

        pbf_instance = pbf.AnalyticalCorrectedNumerically(convolver_interface,
                                                          potential=model_src.potential)

        kernel_constructor = KernelConstructor()

        CSD_MASK = np.ones(convolver.shape('CSD'),
                           dtype=bool)

        kernel_constructor.crosskernel = CrossKernelConstructor(convolver_interface,
                                                                CSD_MASK)

        B_KESI = kernel_constructor.potential_basis_functions_at_electrodes(electrodes,
                                                                            pbf_instance)
        KERNEL_KESI = kernel_constructor.kernel(B_KESI)
        CROSSKERNEL_KESI = kernel_constructor.crosskernel(B_KESI)
        del B_KESI  # the array is large and no longer needed

        reconstructor = Reconstructor(KERNEL_KESI,
                                      CROSSKERNEL_KESI)
        self.reconstructor = reconstructor