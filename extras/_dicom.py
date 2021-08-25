#!/usr/bin/env python
# coding: utf-8

# # Literature
# 
# - http://dicomiseasy.blogspot.com/2011/10/introduction-to-dicom-chapter-1.html
# - https://pydicom.github.io/
# - http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.25.html
# - https://dicom.innolitics.com/ciods/mr-image/image-plane/00201041


import os

import numpy as np
import uuid
import logging
import scipy.interpolate as si


class permissive_DICOM_interpolator(object):
    _PIXEL_SPACING_ROW = 0
    _PIXEL_SPACING_COL = 1

    def __init__(self, dicoms):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}.{id(self)}')

        self.DICOM_ORIGIN = np.array([ds.ImagePositionPatient for ds in dicoms]) * 1e-3
        DICOM_FIRST_ROW_DIR = np.array([ds.ImageOrientationPatient[:3] for ds in dicoms])
        DICOM_FIRST_COL_DIR = np.array([ds.ImageOrientationPatient[3:] for ds in dicoms])
        DICOM_PS = np.array([ds.PixelSpacing for ds in dicoms]) * 1e-3

        self.PROJ = np.stack([DICOM_FIRST_COL_DIR, DICOM_FIRST_ROW_DIR],
                             axis=1) / DICOM_PS[:, [self._PIXEL_SPACING_ROW, self._PIXEL_SPACING_COL]].reshape(-1, 2, 1)

        self.DICOM_NORMALS = np.cross(DICOM_FIRST_ROW_DIR, DICOM_FIRST_COL_DIR)
        self.DICOM_ARRAYS = [ds.pixel_array for ds in dicoms]
        self._DOT_NORMAL_BASE = -(self.DICOM_ORIGIN * self.DICOM_NORMALS).sum(axis=1)

    def __call__(self, X, Y, Z, max_dst=None):
        if max_dst is None:
            max_dst = np.diff(Z).max()

        DATA = np.zeros((len(X), len(Y), len(Z)),
                        dtype=self.DICOM_ARRAYS[0].dtype)

        # [Projecting points to plane](https://stackoverflow.com/questions/23472048/projecting-3d-points-to-2d-plane)

        NORMALS_X, NORMALS_Y, NORMALS_Z = self.DICOM_NORMALS.T
        for i_z, z in enumerate(Z):
            self.logger.info(f'{i_z} {100.0 * i_z / len(Z):.1f}')

            _DOT_NORMAL_Z = NORMALS_Z * z + self._DOT_NORMAL_BASE
            for i_x, x in enumerate(X):
                _DOT_NORMAL_XZ = _DOT_NORMAL_Z + NORMALS_X * x
                for i_y, y in enumerate(Y):
                    _S = abs(_DOT_NORMAL_XZ + NORMALS_Y * y)
                    i_ds = np.argmin(_S)
                    if _S[i_ds] > max_dst:
                        continue

                    try:
                        DATA[i_x, i_y, i_z] = self.nearest(i_ds, x, y, z)
                    except IndexError:
                        pass

        return DATA

    def nearest(self, slice_n, x, y, z):
        return self.DICOM_ARRAYS[slice_n][self._slice_indices(slice_n, x, y, z)]

    def _slice_indices(self, slice_n, x, y, z):
        return tuple(np.round(self._slice_coords(slice_n, x, y, z)).astype(int))

    def _slice_coords(self, slice_n, x, y, z):
        return np.matmul(self.PROJ[slice_n],
                         np.array([x, y, z]) - self.DICOM_ORIGIN[slice_n])


class strict_DICOM_interpolator(object):
    _PIXEL_SPACING_ROW = 0
    _PIXEL_SPACING_COL = 1

    def __init__(self, dicoms, tolerance=1e-3, method='nearest'):
        self.tolerance = tolerance

        self._extract_orientation_data(dicoms)
        DICOMS = sorted(dicoms, key=self._slice_dot_normal)

        self._assert_orientation_coherent(DICOMS)

        DZ = np.diff(list(map(self._slice_dot_normal, DICOMS)))
        assert abs(DZ - DZ.mean()).max() < self.tolerance

        self._x0, self._y0, self._z0 = np.array(DICOMS[0].ImagePositionPatient) * 1e-3
        self._CONV = np.array([self._NORMAL / (DZ.mean() * 1e-3),
                               self._FIRST_COL_ORIENTATION / (
                                           self._PIXEL_SPACING[self._PIXEL_SPACING_ROW] * 1e-3),
                               self._FIRST_ROW_ORIENTATION / (
                                           self._PIXEL_SPACING[self._PIXEL_SPACING_COL] * 1e-3),
                               ])
        _DATA = np.stack([ds.pixel_array for ds in DICOMS],
                         axis=0)
        self._interpolator = si.RegularGridInterpolator([range(_DATA.shape[0]),
                                                         range(_DATA.shape[1]),
                                                         range(_DATA.shape[2]),
                                                         ],
                                                        _DATA,
                                                        method=method,
                                                        bounds_error=False,
                                                        fill_value=0)

    def _slice_dot_normal(self, ds):
        return np.dot(self._NORMAL, ds.ImagePositionPatient)

    def _assert_top_left_pixel_aligned(self, dicoms):
        PROJ = np.array([self._FIRST_COL_ORIENTATION,
                         self._FIRST_ROW_ORIENTATION])
        LOCS = np.array([np.matmul(PROJ,
                                   ds.ImagePositionPatient)
                         for ds in dicoms])
        assert abs(LOCS - LOCS.mean(axis=0).reshape(-1, 2)) < self.tolerance

    def _assert_orientation_coherent(self, dicoms):
        for ds in dicoms:
            assert abs(2 - np.dot(self._ORIENTATION,
                                  ds.ImageOrientationPatient)) < self.tolerance
            assert abs(
                self._ORIENTATION - ds.ImageOrientationPatient).max() < self.tolerance
            assert abs(
                self._PIXEL_SPACING - ds.PixelSpacing).max() < self.tolerance

    def _extract_orientation_data(self, dicoms):
        self._ORIENTATION = np.array(dicoms[0].ImageOrientationPatient)
        self._FIRST_ROW_ORIENTATION = self._ORIENTATION[:3]
        self._FIRST_COL_ORIENTATION = self._ORIENTATION[3:]
        self._NORMAL = np.cross(self._FIRST_ROW_ORIENTATION,
                                self._FIRST_COL_ORIENTATION)

        self._PIXEL_SPACING = np.mean([ds.PixelSpacing for ds in dicoms],
                                      axis=0)

    def __call__(self, X, Y, Z):
        return self._interpolate(X - self._x0,
                                 Y - self._y0,
                                 Z - self._z0)

    def _interpolate(self, X, Y, Z):
        # no matrix multiplication to allow for broadcasting
        SLICE = (self._CONV[0, 0] * X
                 + self._CONV[0, 1] * Y
                 + self._CONV[0, 2] * Z)
        ROW = (self._CONV[1, 0] * X
               + self._CONV[1, 1] * Y
               + self._CONV[1, 2] * Z)
        COL = (self._CONV[2, 0] * X
               + self._CONV[2, 1] * Y
               + self._CONV[2, 2] * Z)
        return self._interpolator(np.stack([SLICE, ROW, COL], axis=-1))


try:
    from pydicom.dataset import Dataset, FileMetaDataset

except (ImportError, SystemError, ValueError):
    pass

else:
    class StoreDICOM(object):
        def __init__(self, StudyInstanceUID, FrameOfReferenceUID,
                     dicom_dir,
                     filename_pattern='{0.SeriesNumber}_{0.InstanceNumber}.dcm'):

            self.StudyInstanceUID = StudyInstanceUID
            self.FrameOfReferenceUID = FrameOfReferenceUID

            self.dicom_dir = dicom_dir
            if not os.path.exists(dicom_dir):
                os.makedirs(dicom_dir, mode=0o755)

            self.filename_pattern = filename_pattern

        def store(self, X, Y, Z, DATA,
                  WindowCenter, WindowWidth,
                  SeriesNumber=501,
                  SeriesInstanceUID=None,
                  SeriesDescription=None,
                  **kwargs):
            if SeriesInstanceUID is None:
                SeriesInstanceUID = f'2.5.{uuid.uuid4().int:d}'


            for i, z in enumerate(Z):
                ds = self.DICOM_MR_backbone(SeriesInstanceUID)
                self.add_universal_metadata(ds, X, Y, Z, i)

                ds.SeriesNumber = SeriesNumber

                SLICE_INT = DATA[:, :, i]

                ds.WindowCenter = WindowCenter
                ds.WindowWidth = WindowWidth

                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1
                ds.PixelData = SLICE_INT.tobytes()

                ds.file_meta = self.DICOM_File_Meta_Information(ds)
                if SeriesDescription is not None:
                    ds.SeriesDescription = SeriesDescription

                for k, v in kwargs.items():
                    setattr(ds, k, v)

                ds.save_as(os.path.join(self.dicom_dir,
                                        self.filename_pattern.format(ds)),
                           write_like_original=False)

        def DICOM_MR_backbone(self, SeriesInstanceUID):
            # Main data elements
            ds = Dataset()
            ds.SpecificCharacterSet = 'ISO_IR 101'  # latin-2
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR
            ds.SOPInstanceUID = f'2.5.{uuid.uuid4().int:d}'

            ds.StudyDate = ''  # YYYYMMDD
            ds.StudyTime = ''  # HHMMSS

            ds.AccessionNumber = ''
            ds.Modality = 'MR'
            ds.Manufacturer = 'Jakub Dzik'

            ds.ReferringPhysicianName = ''

            ds.PatientName = 'Anonymized'
            ds.PatientID = '1234'
            ds.PatientBirthDate = ''
            ds.PatientSex = ''

            ds.ContrastBolusAgent = ''

            ds.ScanningSequence = 'EP'
            ds.SequenceVariant = 'NONE'
            ds.ScanOptions = ''
            ds.MRAcquisitionType = '3D'

            ds.EchoTime = ''

            ds.EchoTrainLength = ''

            ds.PatientPosition = ''

            ds.StudyInstanceUID = self.StudyInstanceUID
            ds.SeriesInstanceUID = SeriesInstanceUID
            ds.StudyID = ''

            ds.AcquisitionNumber = ''

            ds.FrameOfReferenceUID = self.FrameOfReferenceUID
            ds.Laterality = ''

            ds.PositionReferenceIndicator = ''

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'

            ds.CommentsOnThePerformedProcedureStep = ''
            ds.RequestedProcedureID = 'MOZGOWIE'
            ds.ReasonForTheRequestedProcedure = ''
            ds.RequestedProcedurePriority = ''
            ds.PatientTransportArrangements = ''
            ds.RequestedProcedureLocation = ''
            ds.RequestedProcedureComments = ''
            ds.ReasonForTheImagingServiceRequest = ''
            ds.IssueDateOfImagingServiceRequest = '20210222'
            ds.IssueTimeOfImagingServiceRequest = '084604.947'
            ds.OrderEntererLocation = ''
            ds.OrderCallbackPhoneNumber = ''
            ds.ImagingServiceRequestComments = ''
            ds.PresentationLUTShape = 'IDENTITY'

            ds.is_implicit_VR = False
            ds.is_little_endian = True
            return ds

        @staticmethod
        def DICOM_File_Meta_Information(ds):
            # File meta info data elements
            file_meta = FileMetaDataset()
        #     file_meta.FileMetaInformationGroupLength = 210
        #     file_meta.FileMetaInformationVersion = b'\x00\x01'
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
            file_meta.ImplementationClassUID = '1.2.276.0.20.1.1.33.6.1.0'
            file_meta.ImplementationVersionName = 'PatientSelect6.1'
            file_meta.SourceApplicationEntityTitle = 'kESI'
            return file_meta

        @staticmethod
        def d(X):
            return (X[-1] - X[0]) / (len(X) - 1)

        def add_universal_metadata(self, ds, X, Y, Z, i):
            ds.InstanceNumber = i
            ds.SliceLocation = 1000 * Z[i]

            ds.ImagePositionPatient = [1000 * X[0],
                                       1000 * Y[0],
                                       1000 * Z[i]]
            ds.ImageOrientationPatient = [0.0, 1.0, 0.0,
                                          1.0, 0.0, 0.0]
        #     ds.ImageOrientationPatient = [1.0, 0.0, 0.0,
        #                                   0.0, 1.0, 0.0]

            ds.PixelSpacing = [1000 * self.d(X),
                               1000 * self.d(Y),
                               ]
            ds.SliceThickness = 1000 * self.d(Z)
            ds.SpacingBetweenSlices = 1000 * self.d(Z)
            ds.Rows = len(X)
            ds.Columns = len(Y)


try:
    import matplotlib.pyplot as plt
    import cbf


except (ImportError, SystemError, ValueError):
    pass

else:
    def crude_plot_data(DATA,
                        x=None,
                        y=None,
                        z=None,
                        dpi=30,
                        cmap=cbf.bwr,
                        title=None,
                        amp=None):
        wx, wy, wz = DATA.shape
        x, y, z = [w // 2 if a is None else a
                   for a, w in zip([x, y, z], [wx, wy, wz])]


        fig = plt.figure(figsize=((wx + wy) / dpi,
                                  (wz + wy) / dpi))
        if title is not None:
            fig.suptitle(title)
        gs = plt.GridSpec(2, 2,
                          figure=fig,
                          width_ratios=[wx, wy],
                          height_ratios=[wz, wy])

        ax_xz = fig.add_subplot(gs[0, 0])
        ax_xz.set_aspect('equal')
        ax_xz.set_ylabel('Z')

        ax_xy = fig.add_subplot(gs[1, 0],
                                sharex=ax_xz)
        ax_xy.set_aspect('equal')
        ax_xy.set_ylabel('Y')
        ax_xy.set_xlabel('X')

        ax_yz = fig.add_subplot(gs[0, 1],
                                sharey=ax_xz)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlabel('Y')

        cax = fig.add_subplot(gs[1,1])
        cax.set_visible(False)
    #     cax.get_xaxis().set_visible(False)
    #     cax.get_yaxis().set_visible(False)


        if amp is None:
            amp = abs(DATA).max()
        ax_xz.imshow(DATA[:, y, :].T,
                     vmin=-amp,
                     vmax=amp,
                     cmap=cmap)
        ax_xz.axvline(x, ls=':', color=cbf.BLACK)
        ax_xz.axhline(z, ls=':', color=cbf.BLACK)

        ax_xy.imshow(DATA[:, :, z].T,
                     vmin=-amp,
                     vmax=amp,
                     cmap=cmap)
        ax_xy.axvline(x, ls=':', color=cbf.BLACK)
        ax_xy.axhline(y, ls=':', color=cbf.BLACK)

        im = ax_yz.imshow(DATA[x, :, :].T,
                          vmin=-amp,
                          vmax=amp,
                          cmap=cmap)
        ax_yz.axvline(y, ls=':', color=cbf.BLACK)
        ax_yz.axhline(z, ls=':', color=cbf.BLACK)
        fig.colorbar(im, ax=cax)
