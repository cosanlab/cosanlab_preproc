from __future__ import division

"""
Workflow Maker
==============

Handy function to build dynamic workflows using BIDS formatted data files.

"""

import os
import matplotlib
matplotlib.use('Agg')
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Merge, IdentityInterface
from nipype.pipeline.engine import Node, Workflow
from .interfaces import	Plot_Coregistration_Montage,Plot_Quality_Control,Plot_Realignment_Parameters,Create_Covariates,Down_Sample_Precision,Create_Encoding_File, Filter_In_Mask
from .utils import get_resource_path
from bids.grabbids import BIDSLayout
from nipype.interfaces.nipy.preprocess import ComputeMask
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.fsl import MCFLIRT, TOPUP, ApplyTOPUP

from nipype.interfaces.fsl.maths import MeanImage
from nipype.interfaces.fsl import Merge as MERGE
from nipype.interfaces.fsl.utils import Smooth
from nipype.interfaces.nipy.preprocess import Trim


def wfmaker(project_dir,raw_dir,subject_id,task_name='',apply_trim=False,apply_dist_corr=False,apply_smooth=False,apply_filter=False,mni_template='2mm',ants_threads=12,readable_crash_files=False):

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

    Args:
        project_dir (str): absolute root of project folder, e.g. /my/data/myproject. All preprocessed data will be placed under this foler and the raw_dir folder will be searched for under this folder
        raw_dir (str): folder name for raw data, e.g. 'raw' which would be treated as /my/data/myproject/raw
        subject_id (str/int): subject ID to process. Can be either a subject ID string e.g. 'sid-0001' or an integer to index the entire list of subjects in raw_dir, e.g. 0, which would process the first subject
        apply_trim (int/bool; optional): number of volumes to trim from the beginning of each functional run; default is None
        task_name (str; optional): which functional task runs to process; default is all runs
        apply_dist_corr (bool; optional): look for fmap files and perform distortion correction; default False
        smooth (int/list; optional): smoothing to perform in FWHM mm; if a list is provided will create outputs for each smoothing kernel separately; default False
        apply_filter (float/list; optional): low-pass/high-freq filtering cut-offs in Hz; if a list is provided will create outputs for each filter cut-off separately. With high temporal resolution scans .25Hz is a decent value to capture respitory artifacts; default None/False
        mni_template (str; optional): which mm resolution template to use, e.g. '3mm'; default '2mm'
        ants_threads (int; optional): number of threads ANTs should use for its processes; default 12
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
    if mni_template not in ['1mm','2mm','3mm']:
        raise ValueError("MNI template must be: 1mm, 2mm, or 3mm")

    data_dir = os.path.join(project_dir,raw_dir)
    output_dir = os.path.join(project_dir,'preprocessed')
    output_final_dir = os.path.join(output_dir,'final')
    output_interm_dir = os.path.join(output_dir,'intermediate')
    log_dir = os.path.join(project_dir,'logs','nipype')

    if not os.path.exists(output_final_dir):
        os.makedirs(output_final_dir)
    if not os.path.exists(output_interm_dir):
        os.makedirs(output_interm_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set MNI template
    MNItemplate = os.path.join(get_resource_path(),'MNI152_T1_' + mni_template + '_brain.nii.gz')
    MNImask = os.path.join(get_resource_path(),'MNI152_T1_' + mni_template + '_brain_mask.nii.gz')
    MNItemplatehasskull = os.path.join(get_resource_path(),'MNI152_T1_' + mni_template + '.nii.gz')

    # Set ANTs files
    bet_ants_template = os.path.join(get_resource_path(),'OASIS_template.nii.gz')
    bet_ants_prob_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumProbabilityMask.nii.gz')
    bet_ants_registration_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumRegistrationMask.nii.gz')

    ##################
    ### INPUT NODE ###
    ##################

    layout = BIDSLayout(data_dir)
    if isinstance(subject_id, int):
        subject_id = layout.get_subjects()[subject_id]
    subId = subject_id[4:]

    #Get anat file location
    anat = layout.get(subject=subId,type='T1w',extensions='.nii.gz')[0].filename

    #Get functional file locations
    if task_name:
        funcs = [f.filename for f in layout.get(subject=subId,type='bold',task=task_name,extensions='.nii.gz')]
    else:
        funcs = [f.filename for f in layout.get(subject=subId,type='bold',extensions='.nii.gz')]

    #Turn functional file list into interable Node
    func_scans = Node(IdentityInterface(fields=['scan']),name='func_scans')
    func_scans.iterables = ('scan',funcs)

    #Get TR for use in filtering below; we're assuming all BOLD runs have the same TR
    tr_length = layout.get_metadata(funcs[0])['RepetitionTime']

    #####################################
    ## TRIM ##
    #####################################
    if apply_trim:
        trim = Node(Trim(),name = 'trim')
        trim.inputs.begin_index = apply_trim

    #####################################
    ## DISTORTION CORRECTION ##
    #####################################

    if apply_dist_corr:
        #Get fmap file locations
        fmaps = [f.filename for f in layout.get(subject=subId,modality='fmap',extensions='.nii.gz')]
        if not fmaps:
            raise IOError("Distortion Correction requested but field map scans not found...")

        #Get fmap metadata
        totalReadoutTimes, measurements, fmap_pes = [],[],[]

        for i, fmap in enumerate(fmaps):
            # Grab total readout time for each fmap
            totalReadoutTimes.append(layout.get_metadata(fmap)['TotalReadoutTime'])

            # Grab measurements
            if len(layout.get_metadata(fmap)['dcmmeta_shape']) == 4:
                measurements.append(layout.get_metadata(fmap)['dcmmeta_shape'][-1])
            elif len(layout.get_metadata(fmap)['dcmmeta_shape']) == 3:
                measurements.append(1)

            # Get phase encoding direction
            fmap_pe = layout.get_metadata(fmap)["PhaseEncodingDirection"]
            fmap_pes.append(fmap_pe)


        encoding_file_writer = Node(interface=Create_Encoding_File(),name='create_encoding')
        encoding_file_writer.inputs.totalReadoutTimes = totalReadoutTimes
        encoding_file_writer.inputs.fmaps = fmaps
        encoding_file_writer.inputs.fmap_pes = fmap_pes
        encoding_file_writer.inputs.measurements = measurements
        encoding_file_writer.inputs.file_name='encoding_file.txt'


        merge_to_file_list = Node(interface=Merge(2), infields = ['in1','in2'],name='merge_to_file_list')
        merge_to_file_list.inputs.in1 = fmaps[0]
        merge_to_file_list.inputs.in1 = fmaps[1]

        #Merge AP and PA distortion correction scans
        merger = Node(interface=MERGE(dimension='t'),name='merger')
        merger.inputs.output_type = 'NIFTI_GZ'
        merger.inputs.in_files = fmaps
        merger.inputs.merged_file = 'merged_epi.nii.gz'

        #Create distortion correction map
        topup = Node(interface=TOPUP(),name='topup')
        topup.inputs.output_type = 'NIFTI_GZ'

        #Apply distortion correction to other scans
        apply_topup = Node(interface=ApplyTOPUP(),name='apply_topup')
        apply_topup.inputs.output_type = 'NIFTI_GZ'
        apply_topup.inputs.method = 'jac'
        apply_topup.inputs.interp = 'spline'


    ###################################
    ### REALIGN ###
    ###################################
    realign_fsl = Node(MCFLIRT(),name="realign")
    realign_fsl.inputs.cost = 'mutualinfo'
    realign_fsl.inputs.mean_vol = True
    realign_fsl.inputs.output_type = 'NIFTI_GZ'
    realign_fsl.inputs.save_mats = True
    realign_fsl.inputs.save_rms = True
    realign_fsl.inputs.save_plots = True

    ###################################
    ### MEAN EPIs ###
    ###################################
    #For coregistration after realignment
    mean_epi = Node(MeanImage(),name='mean_epi')
    mean_epi.inputs.dimension = 'T'

    #For after normalization is done to plot checks
    mean_norm_epi = Node(MeanImage(),name='mean_norm_epi')
    mean_norm_epi.inputs.dimension = 'T'

    ###################################
    ### MASK, ART, COV CREATION ###
    ###################################
    compute_mask = Node(ComputeMask(), name='compute_mask')
    compute_mask.inputs.m = .05

    art = Node(ArtifactDetect(),name='art')
    art.inputs.use_differences = [True, False]
    art.inputs.use_norm = True
    art.inputs.norm_threshold = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type = 'file'
    art.inputs.parameter_source = 'FSL'

    make_cov = Node(Create_Covariates(),name='make_cov')

    ###################################
    ### BRAIN EXTRACTION ###
    ###################################
    brain_extraction_ants = Node(BrainExtraction(),name='brain_extraction')
    brain_extraction_ants.inputs.anatomical_image = anat #from BIDS
    brain_extraction_ants.inputs.dimension = 3
    brain_extraction_ants.inputs.use_floatingpoint_precision = 1
    brain_extraction_ants.inputs.num_threads = ants_threads
    brain_extraction_ants.inputs.brain_probability_mask = bet_ants_prob_mask
    brain_extraction_ants.inputs.keep_temporary_files = 1
    brain_extraction_ants.inputs.brain_template = bet_ants_template
    brain_extraction_ants.inputs.extraction_registration_mask = bet_ants_registration_mask
    brain_extraction_ants.inputs.out_prefix = ''

    ###################################
    ### COREGISTRATION ###
    ###################################
    coregistration = Node(Registration(), name='coregistration')
    coregistration.inputs.float = False
    coregistration.inputs.output_transform_prefix = "meanEpi2highres"
    coregistration.inputs.transforms = ['Rigid']
    coregistration.inputs.transform_parameters = [(0.1,), (0.1,)]
    coregistration.inputs.number_of_iterations = [[1000,500,250,100]]
    coregistration.inputs.dimension = 3
    coregistration.inputs.num_threads = ants_threads
    coregistration.inputs.write_composite_transform = True
    coregistration.inputs.collapse_output_transforms = True
    coregistration.inputs.metric = ['MI']
    coregistration.inputs.metric_weight = [1]
    coregistration.inputs.radius_or_number_of_bins = [32]
    coregistration.inputs.sampling_strategy = ['Regular']
    coregistration.inputs.sampling_percentage = [0.25]
    coregistration.inputs.convergence_threshold = [1.e-8]
    coregistration.inputs.convergence_window_size = [10]
    coregistration.inputs.smoothing_sigmas = [[3,2,1,0]]
    coregistration.inputs.sigma_units = ['mm']
    coregistration.inputs.shrink_factors = [[8,4,2,1]]
    coregistration.inputs.use_estimate_learning_rate_once = [True]
    coregistration.inputs.use_histogram_matching = [False]
    coregistration.inputs.initial_moving_transform_com = True
    coregistration.inputs.output_warped_image = True
    coregistration.inputs.winsorize_lower_quantile = 0.01
    coregistration.inputs.winsorize_upper_quantile = 0.99

    ###################################
    ### NORMALIZATION ###
    ###################################
    #ANTS step through several different iterations starting with linear, affine and finally non-linear diffuseomorphic alignment. The settings below increase the run time but yield a better alignment solution
    normalization = Node(Registration(),name='normalization')
    normalization.inputs.float = False
    normalization.inputs.collapse_output_transforms=True
    normalization.inputs.convergence_threshold=[1e-06]
    normalization.inputs.convergence_window_size=[10]
    normalization.inputs.dimension = 3
    normalization.inputs.fixed_image = MNItemplate #MNI 152 1mm
    normalization.inputs.initial_moving_transform_com=True
    normalization.inputs.metric=['MI', 'MI', 'CC']
    normalization.inputs.metric_weight=[1.0]*3
    normalization.inputs.number_of_iterations=[[1000, 500, 250, 100],
                                     [1000, 500, 250, 100],
                                     [100, 70, 50, 20]]
    normalization.inputs.num_threads= ants_threads
    normalization.inputs.output_transform_prefix = 'anat2template'
    normalization.inputs.output_inverse_warped_image=True
    normalization.inputs.output_warped_image = True
    normalization.inputs.radius_or_number_of_bins=[32, 32, 4]
    normalization.inputs.sampling_percentage=[0.25, 0.25, 1]
    normalization.inputs.sampling_strategy=['Regular',
                                  'Regular',
                                  'None']
    normalization.inputs.shrink_factors=[[8, 4, 2, 1]]*3
    normalization.inputs.sigma_units=['vox']*3
    normalization.inputs.smoothing_sigmas=[[3, 2, 1, 0]]*3
    normalization.inputs.transforms = ['Rigid','Affine','SyN']
    normalization.inputs.transform_parameters=[(0.1,),
                                     (0.1,),
                                     (0.1, 3.0, 0.0)]
    normalization.inputs.use_histogram_matching=True
    normalization.inputs.winsorize_lower_quantile=0.005
    normalization.inputs.winsorize_upper_quantile=0.995
    normalization.inputs.write_composite_transform=True

    ###################################
    ### APPLY TRANSFORMS AND SMOOTH ###
    ###################################
    merge_transforms = Node(Merge(2), iterfield=['in2'], name ='merge_transforms')

    # Used for epi -> mni, via (coreg + norm)
    apply_transforms = Node(ApplyTransforms(),iterfield=['input_image'],name='apply_transforms')
    apply_transforms.inputs.input_image_type = 3
    apply_transforms.inputs.float = False
    apply_transforms.inputs.num_threads = 12
    apply_transforms.inputs.environ = {}
    apply_transforms.inputs.interpolation = 'BSpline'
    apply_transforms.inputs.invert_transform_flags = [False, False]
    apply_transforms.inputs.reference_image = MNItemplate

    # Used for t1 segmented -> mni, via (norm)
    apply_transform_seg = Node(ApplyTransforms(),name='apply_transform_seg')
    apply_transform_seg.inputs.input_image_type = 3
    apply_transform_seg.inputs.float = False
    apply_transform_seg.inputs.num_threads = 12
    apply_transform_seg.inputs.environ = {}
    apply_transform_seg.inputs.interpolation = 'MultiLabel'
    apply_transform_seg.inputs.invert_transform_flags = [False, False]
    apply_transform_seg.inputs.reference_image = MNItemplate

    ###################################
    ### PLOTS ###
    ###################################
    plot_realign = Node(Plot_Realignment_Parameters(),name="plot_realign")
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")
    plot_normalization_check = Node(Plot_Coregistration_Montage(),name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = MNItemplatehasskull

    ############################################
    ### FILTER, SMOOTH, DOWNSAMPLE PRECISION ###
    ############################################
    #Use cosanlab_preproc for down sampling
    down_samp = Node(Down_Sample_Precision(),name="down_samp")

    #Use FSL for smoothing
    if apply_smooth:
        smooth = Node(Smooth(),name='smooth')
        if isinstance(apply_smooth, list):
            smooth.iterables = ("fwhm",apply_smooth)
        elif isinstance(apply_smooth, int) or isinstance(apply_smooth, float):
            smooth.inputs.fwhm = apply_smooth
        else:
            raise ValueError("apply_smooth must be a list or int/float")

    #Use cosanlab_preproc for low-pass filtering
    if apply_filter:
        lp_filter = Node(Filter_In_Mask(),name='lp_filter')
        lp_filter.inputs.mask = MNImask
        lp_filter.inputs.sampling_rate = tr_length
        lp_filter.inputs.high_pass_cutoff = 0
        if isinstance(apply_filter,list):
            lp_filter.iterables = ("low_pass_cutoff",filter)
        elif isinstance(apply_filter, int) or isinstance(apply_filter, float):
            lp_filter.inputs.low_pass_cutoff = apply_filter
        else:
            raise ValueError("apply_filter must be a list or int/float")

    ###################
    ### OUTPUT NODE ###
    ###################
    #Collect all final outputs in the output dir and get rid of file name additions
    datasink = Node(DataSink(),name='datasink')
    datasink.inputs.base_directory = output_final_dir
    datasink.inputs.container = subject_id

    # Remove substitutions
    data_dir_parts = data_dir.split('/')[1:]
    prefix = ['_scan_'] + data_dir_parts + [subject_id] + ['func']
    func_scan_names = [os.path.split(elem)[-1] for elem in funcs]
    to_replace = []
    for elem in func_scan_names:
        bold_name = elem.split(subject_id + '_')[-1]
        bold_name = bold_name.split('.nii.gz')[0]
        to_replace.append(('..'.join(prefix + [elem]), bold_name))
    datasink.inputs.substitutions = to_replace

    ##################
    ### WIRE IT UP ###
    ##################

    # Init workflow
    workflow = Workflow(name=subId)
    workflow.base_dir = output_interm_dir
    workflow.config['logging'] = {'log_directory': log_dir, 'log_to_file': True}

    if readable_crash_files:
        workflow.config['execution'] = {'crashfile_format': 'txt'}

    ############################
    ######### PART (1) #########
    # func -> discorr -> trim -> realign
    # OR
    # func -> trim -> realign
    # OR
    # func -> discorr -> realign
    # OR
    # func -> realign
    ############################
    if apply_dist_corr:
        workflow.connect([
            (encoding_file_writer, topup,[('encoding_file','encoding_file')]),
            (encoding_file_writer, apply_topup,[('encoding_file','encoding_file')]),
            (merger,topup,[('merged_file','in_file')]),
            (func_scans,apply_topup,[('scan','in_files')]),
            (topup,apply_topup,[('out_fieldcoef','in_topup_fieldcoef'),
                                ('out_movpar','in_topup_movpar')])
        ])
        if apply_trim:
            # Dist Corr + Trim
            workflow.connect([
                (apply_topup,trim,[('out_corrected','in_file')]),
                (trim, realign_fsl, [('out_file','in_file')])
            ])
        else:
            # Dist Corr + No Trim
            workflow.connect([
                (apply_topup,realign_fsl,[('out_corrected','in_file')])
            ])
    else:
        if apply_trim:
            # No Dist Corr + Trim
            workflow.connect([
                (func_scans, trim, [('scan','in_file')]),
                (trim, realign_fsl, [('out_file','in_file')])
            ])
        else:
            # No Dist Corr + No Trim
            workflow.connect([
                (func_scans, realign_fsl, [('scan','in_file')]),
            ])

    ##########################################
    ############### PART (2) #################
    # realign -> coreg -> mni (via t1)
    # t1 -> mni
    # covariate creation
    # plot creation
    ###########################################

    workflow.connect([
        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
        (realign_fsl, plot_qa, [('out_file','dat_img')]),
        (realign_fsl, art, [('out_file','realigned_files'),
                           ('par_file','realignment_parameters')]),
        (realign_fsl, mean_epi, [('out_file','in_file')]),
        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
        (mean_epi, compute_mask, [('out_file','mean_volume')]),
        (compute_mask, art, [('brain_mask','mask_file')]),
        (art, make_cov, [('outlier_files','spike_id')]),
        (art, plot_realign, [('outlier_files','outliers')]),
        (plot_qa, make_cov, [('fd_outliers','fd_outliers')]),
        (brain_extraction_ants, coregistration, [('N4Corrected0','fixed_image')]),
        (mean_epi, coregistration, [('out_file','moving_image')]),
        (brain_extraction_ants, normalization, [('N4Corrected0','moving_image')]),
        (coregistration, merge_transforms, [('composite_transform','in2')]),
        (normalization, merge_transforms, [('composite_transform','in1')]),
        (merge_transforms, apply_transforms, [('out','transforms')]),
        (realign_fsl, apply_transforms, [('out_file','input_image')]),
        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
        (normalization, apply_transform_seg, [('composite_transform','transforms')]),
        (brain_extraction_ants, apply_transform_seg, [('BrainExtractionSegmentation','input_image')]),
        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')])
    ])

    ##################################################
    ################### PART (3) #####################
    # epi (in mni) -> filter -> smooth -> down sample
    # OR
    # epi (in mni) -> filter -> down sample
    # OR
    # epi (in mni) -> smooth -> down sample
    # OR
    # epi (in mni) -> down sample
    ###################################################

    if apply_filter:
        workflow.connect([
            (apply_transforms, lp_filter, [('output_image','in_file')])
        ])

        if apply_smooth:
            # Filtering + Smoothing
            workflow.connect([
                (lp_filter, smooth, [('out_file','in_file')]),
                (smooth, down_samp, [('smoothed_file','in_file')])
                ])
        else:
            # Filtering + No Smoothing
            workflow.connect([
                (lp_filter, down_samp, [('out_file','in_file')])
            ])
    else:
        if apply_smooth:
            # No Filtering + Smoothing
            workflow.connect([
                (apply_transforms, smooth, [('output_image', 'in_file')]),
                (smooth, down_samp, [('smoothed_file','in_file')])
                ])
        else:
            # No Filtering + No Smoothing
            workflow.connect([
                (apply_transforms, down_samp, [('output_image', 'in_file')])
            ])

    ##########################################
    ############### PART (4) #################
    # down sample -> save
    # plots -> save
    # covs -> save
    # t1 (in mni) -> save
    # t1 segmented masks (in mni) -> save
    ##########################################

    workflow.connect([
        (down_samp, datasink, [('out_file','functional.@down_samp')]),
        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
        (make_cov, datasink, [('covariates','functional.@covariates')]),
        (normalization, datasink, [('warped_image','structural.@normanat')]),
        (apply_transform_seg, datasink,[('output_image','structural.@normanatseg')])
    ])

    if not os.path.exists(os.path.join(output_dir,'pipeline.png')):
        workflow.write_graph(dotfilename=os.path.join(output_dir,'pipeline'),format='png')

    print(f"Creating workflow for subject: {subject_id}")
    return workflow
