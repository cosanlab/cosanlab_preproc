from __future__ import division
from ._builder import builder
import os
from bids.grabbids import BIDSLayout
from .utils import file_getter
import six

"""
Workflow Maker
==============

Handy function to build dynamic workflows using BIDS formatted data files.

"""


def wfmaker(project_dir, raw_dir, subject_id, task_name='', apply_trim=False, apply_dist_corr=False, apply_smooth=False, apply_filter=False, mni_template='2mm', apply_n4=True, ants_threads=8, readable_crash_files=False):
    """
    This function returns a "standard" workflow based on requested settings. Assumes data is in the following directory structure in BIDS format:

    *Work flow steps*:

    1) EPI Distortion Correction (FSL; optional)
    2) Trimming (nipy)
    3) Realignment/Motion Correction (FSL)
    4) Artifact Detection (rapidART/python)
    5) Brain Extraction + N4 Bias Correction (ANTs)
    6) Coregistration (rigid) (ANTs)
    7) Normalization to MNI (non-linear) (ANTs)
    8) Low-pass filtering (nilearn; optional)
    8) Smoothing (FSL; optional)
    9) Downsampling to INT16 precision to save space (nibabel)

    If data contains multiple sessions, this returns a *list* of workflows each of which should be run independently.

    Args:
        project_dir (str): full path to the root of project folder, e.g. /my/data/myproject. All preprocessed data will be placed under this foler and the raw_dir folder will be searched for under this folder
        raw_dir (str): folder name for raw data, e.g. 'raw' which would be automatically converted to /my/data/myproject/raw
        subject_id (str/int): subject ID to process. Can be either a subject ID string e.g. 'sid-0001' or an integer to index the entire list of subjects in raw_dir, e.g. 0, which would process the first subject
        apply_trim (int/bool; optional): number of volumes to trim from the beginning of each functional run; default is None
        task_name (str; optional): which functional task runs to process; default is all runs
        apply_dist_corr (bool; optional): look for fmap files and perform distortion correction; default False
        smooth (int/list; optional): smoothing to perform in FWHM mm; if a list is provided will create outputs for each smoothing kernel separately; default False
        apply_filter (float/list; optional): low-pass/high-freq filtering cut-offs in Hz; if a list is provided will create outputs for each filter cut-off separately. With high temporal resolution scans .25Hz is a decent value to capture respitory artifacts; default None/False
        mni_template (str; optional): which mm resolution template to use, e.g. '3mm'; default '2mm'
        apply_n4 (bool; optional): perform N4 Bias Field correction on the anatomical image; default true
        ants_threads (int; optional): number of threads ANTs should use for its processes; default 8
        readable_crash_files (bool; optional): should nipype crash files be saved as txt? This makes them easily readable, but sometimes interferes with nipype's ability to use cached results of successfully run nodes (i.e. picking up where it left off after bugs are fixed); default False

    Examples:

        >>> from cosanlab_preproc.wfmaker import wfmaker
        >>> # Create workflow that performs no distortion correction, trims first 5 TRs, no filtering, 6mm smoothing, and normalizes to 2mm MNI space. Run it with 16 cores.
        >>>
        >>> workflow = wfmaker(
                        project_dir = '/data/project',
                        raw_dir = 'raw',
                        apply_trim = 5)
        >>>
        >>> workflow.run('MultiProc',plugin_args = {'n_procs': 16})
        >>>
        >>> # Create workflow that performs distortion correction, trims first 25 TRs, no filtering and filtering .25hz, 6mm and 8mm smoothing, and normalizes to 3mm MNI space. Run it serially (will be super slow!).
        >>>
        >>> workflow = wfmaker(
                        project_dir = '/data/project',
                        raw_dir = 'raw',
                        apply_trim = 25,
                        apply_dist_corr = True,
                        apply_filter = [0, .25],
                        apply_smooth = [6.0, 8.0],
                        mni = '3mm')
        >>>
        >>> workflow.run()

    """

    ##################
    ### PATH SETUP ###
    ##################
    if mni_template not in ['1mm', '2mm', '3mm']:
        raise ValueError("MNI template must be: 1mm, 2mm, or 3mm")

    data_dir = os.path.join(project_dir, raw_dir)
    output_dir = os.path.join(project_dir, 'preprocessed')
    output_final_dir = os.path.join(output_dir, 'final')
    output_interm_dir = os.path.join(output_dir, 'intermediate')
    log_dir = os.path.join(project_dir, 'logs', 'nipype')

    if not os.path.exists(output_final_dir):
        os.makedirs(output_final_dir)
    if not os.path.exists(output_interm_dir):
        os.makedirs(output_interm_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    layout = BIDSLayout(data_dir)
    # Dartmouth subjects are named with the sub- prefix, handle whether we receive an integer identifier for indexing or the full subject id with prefixg
    if isinstance(subject_id, six.string_types):
        subId = subject_id[4:]
    elif isinstance(subject_id, int):
        subId = layout.get_subjects()[subject_id]
        subject_id = 'sub-' + subId
    else:
        raise TypeError("subject_id should be a string or integer")

    # For multi-session datasets return a list of workflows consisting of pipelines specific to all data within that session
    # Otherwise return a single workflow
    sessions = layout.get_sessions()
    if len(sessions) > 0:
        workflow = []
        for s in sessions:
            anat, funcs, fmaps = file_getter(layout, subId, apply_dist_corr, task_name, session=s)
            w = builder(subject_id=subject_id, subId=subId, project_dir=project_dir, data_dir=data_dir, output_dir=output_dir, output_final_dir=output_final_dir, output_interm_dir=output_interm_dir, log_dir=log_dir, layout=layout, anat=anat, funcs=funcs, fmaps=fmaps, task_name=task_name, session=s, apply_trim=apply_trim, apply_dist_corr=apply_dist_corr, apply_smooth=apply_smooth, apply_filter=apply_filter, mni_template=mni_template, apply_n4=apply_n4, ants_threads=ants_threads, readable_crash_files=readable_crash_files)
            workflow.append(w)

    else:
        anat, funcs, fmaps = file_getter(layout, subId, apply_dist_corr, task_name)
        workflow = builder(subject_id=subject_id, subId=subId, project_dir=project_dir, data_dir=data_dir, output_dir=output_dir, output_final_dir=output_final_dir, output_interm_dir=output_interm_dir, log_dir=log_dir, layout=layout, anat=anat, funcs=funcs, fmaps=fmaps, task_name=task_name, session=None, apply_trim=apply_trim, apply_dist_corr=apply_dist_corr, apply_smooth=apply_smooth, apply_filter=apply_filter, mni_template=mni_template, apply_n4=apply_n4, ants_threads=ants_threads, readable_crash_files=readable_crash_files)

    return workflow
