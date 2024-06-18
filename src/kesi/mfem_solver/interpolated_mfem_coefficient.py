import mfem.ser as mfem
import numpy as np
import scipy.interpolate as si


class CSDCoefficient(mfem.PyCoefficient):

    def __init__(self, x, y, z, values):
        """x, y, z grid definition, values - numpy, 3D grid of CSD values"""
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.values = values
        interpolator = si.RegularGridInterpolator(
            (x, y, z),
            self.values,
            bounds_error=False,
            fill_value=0,
            method="linear")
        self.interpolator = interpolator

    def get_nearest_neighbor_compiled_coeff(self):
        """Returns a compiled MFEM coefficient, which interpolates the CSD grid using nearest neighbor."""
        x_ = self.x
        y_ = self.y
        z_ = self.z
        values_ = self.values

        @mfem.jit.scalar
        def EvalValue(x):
            # nearest neighbor, but extremely fast thanks to to jit compilation
            x, y, z = x
            x_ind = np.argmin(np.abs(x_ - x))
            y_ind = np.argmin(np.abs(y_ - y))
            z_ind = np.argmin(np.abs(z_ - z))
            return values_[x_ind, y_ind, z_ind]
        return EvalValue

    def EvalValue(self, x):
        """MFEM code is not vectorized, so this will be EXTREMELY slow"""
        return self.interpolator(x)[0]