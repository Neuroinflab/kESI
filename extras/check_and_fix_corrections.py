import argparse
import glob
import os.path

from tqdm import tqdm
import numpy as np


def correct_correction(correction_potential):
    """interpolates nans in 3D array using Gauss Seidel method"""
    bad_pixels = np.isnan(correction_potential).nonzero()
    if bad_pixels:
        interpolatedData = correction_potential.copy()
        interpolatedData[bad_pixels] = np.nanmean(correction_potential)

        def sign(x, size=4):
            """returns the sign of the neighbor to be averaged for boundary elements"""
            if x == 0:
                return [1, 1]
            elif x == size - 1:
                return [-1, -1]
            else:
                return [-1, 1]

        # calculate kernels for the averages on boundaries/non boundary elements
        for i in range(len(bad_pixels)):
            bad_pixels = *bad_pixels, np.array([sign(x) for x in bad_pixels[i]])

        # gauss seidel iteration to interpolate Nan values with neighbors
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
        for _ in range(100):
            for x, y, z, dx, dy, dz in zip(*bad_pixels):
                interpolatedData[x, y, z] = (
                        (interpolatedData[x + dx[0], y, z] + interpolatedData[x + dx[1], y, z] +
                         interpolatedData[x, y + dy[0], z] + interpolatedData[x, y + dy[1], z] +
                         interpolatedData[x, y, z + dz[0]] + interpolatedData[x, y, z + dz[1]]) / 6)
        return interpolatedData
    else:
        return correction_potential


def main():
    parser = argparse.ArgumentParser(description="Interpolates NANs in corrections")
    parser.add_argument("sampled_corrections_folder", type=str)
    parser.add_argument("sampled_corrections_folder_corrected", type=str)
    namespace = parser.parse_args()

    os.makedirs(namespace.sampled_corrections_folder_corrected, exist_ok=True)

    files = glob.glob("*.npz", root_dir=namespace.sampled_corrections_folder)

    for file in tqdm(files):
        a = np.load(os.path.join(namespace.sampled_corrections_folder, file))

        outfile = os.path.join(namespace.sampled_corrections_folder_corrected, file)

        correction_potential = a['CORRECTION_POTENTIAL']

        corrected = correction_potential(correction_potential)
        np.savez(outfile, CORRECTION_POTENTIAL=corrected,
                 X=a['X'],
                 Y=a['Y'],
                 Z=a['Z'],
                 LOCATION=a['LOCATION'],
                 BASE_CONDUCTIVITY=a['BASE_CONDUCTIVITY']
                 )