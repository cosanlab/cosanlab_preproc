"""Handy utilities"""

__all__ = ['get_resource_path', 'get_anatomical', 'get_n_slices', 'get_ta', 'get_slice_order', 'get_n_volumes', 'get_vox_dims']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, sep as pathsep
import nibabel as nib
import os

def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep


def get_anatomical():
    """ Get nltools default anatomical image. """
    return nib.load(os.path.join(get_resource_path(), 'MNI152_T1_2mm.nii.gz'))


def get_n_slices(volume):
    """ Get number of volumes of image. """

    import nibabel as nib
    nii = nib.load(volume)
    return nii.get_shape()[2]


def get_ta(tr, n_slices):
    """ Get slice timing. """

    return tr - tr/float(n_slices)


def get_slice_order(volume):
    """ Get order of slices """

    import nibabel as nib
    nii = nib.load(volume)
    n_slices = nii.get_shape()[2]
    return range(1, n_slices+1)


def get_n_volumes(volume):
    """ Get number of volumes of image. """

    import nibabel as nib
    nii = nib.load(volume)
    if len(nib.shape) < 4:
        return 1
    else:
        return nii.shape[-1]


def get_vox_dims(volume):
    """ Get voxel dimensions of image. """

    import nibabel as nib
    if isinstance(volume, list):
        volume = volume[0]
    nii = nib.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def file_getter(layout, subject_id, dist_corr=False, task_name='', session=None):
    """Helper function to search BIDS layout for session or non-session data. Returns full paths of anatomical, functional, and field map data"""

    if session:
        anat = layout.get(subject=subject_id, type='T1w', extensions='.nii.gz', session=session)[0].filename
        if task_name:
            funcs = [f.filename for f in layout.get(subject=subject_id, type='bold', task=task_name, session=session, extensions='.nii.gz')]
        else:
            funcs = [f.filename for f in layout.get(subject=subject_id, type='bold', extensions='.nii.gz')]
        if dist_corr:
            fmaps = [f.filename for f in layout.get(subject=subject_id, modality='fmap', session=session, extensions='.nii.gz')]
        else:
            fmaps = []
    else:
        anat = layout.get(subject=subject_id, type='T1w', extensions='.nii.gz')[0].filename
        if task_name:
            funcs = [f.filename for f in layout.get(subject=subject_id, type='bold', task=task_name, extensions='.nii.gz')]
        else:
            funcs = [f.filename for f in layout.get(subject=subject_id, type='bold', extensions='.nii.gz')]
        if dist_corr:
            fmaps = [f.filename for f in layout.get(subject=subject_id, modality='fmap', extensions='.nii.gz')]
        else:
            fmaps = []

    return anat, funcs, fmaps
