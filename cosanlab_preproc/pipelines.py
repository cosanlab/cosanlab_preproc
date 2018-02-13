
'''
    Preproc Nipype Pipelines
    ========================
    Various nipype pipelines

'''

__all__ = ['Couple_Preproc_Pipeline','TV_Preproc_Pipeline']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from cosanlab_preproc.utils import get_n_slices, get_ta, get_slice_order, get_vox_dims

# def create_spm_preproc_func_pipeline(data_dir=None, subject_id=None, task_list=None):
#
#     '''REQUIRES FIXS'''
#     import nipype.interfaces.io as nio
#     import nipype.interfaces.utility as util
#     from nipype.pipeline.engine import Node, Workflow
#     from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits
#     import nipype.algorithms.rapidart as ra
#     from nipype.interfaces import spm
#     from nipype.interfaces.nipy.preprocess import ComputeMask
#     import nipype.interfaces.matlab as mlab
#     import os
#     import nibabel as nib
#     from IPython.display import Image
#     import glob
#     from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates
#
# 	###############################
# 	## Set up Nodes
# 	###############################
#
#     ds = Node(nio.DataGrabber(infields=['subject_id', 'task_id'], outfields=['func', 'struc']),name='datasource')
#     ds.inputs.base_directory = os.path.abspath(data_dir + '/' + subject_id)
#     ds.inputs.template = '*'
#     ds.inputs.sort_filelist = True
#     ds.inputs.template_args = {'func': [['task_id']], 'struc':[]}
#     ds.inputs.field_template = {'func': 'Functional/Raw/%s/func.nii','struc': 'Structural/SPGR/spgr.nii'}
#     ds.inputs.subject_id = subject_id
#     ds.inputs.task_id = task_list
#     ds.iterables = ('task_id',task_list)
#     # ds.run().outputs #show datafiles
#
#     # #Setup Data Sinker for writing output files
#     # datasink = Node(nio.DataSink(), name='sinker')
#     # datasink.inputs.base_directory = '/path/to/output'
#     # workflow.connect(realigner, 'realignment_parameters', datasink, 'motion.@par')
#     # datasink.inputs.substitutions = [('_variable', 'variable'),('file_subject_', '')]
#
#     #Get Timing Acquisition for slice timing
#     tr = 2
#     ta = Node(interface=util.Function(input_names=['tr', 'n_slices'], output_names=['ta'],  function = get_ta), name="ta")
#     ta.inputs.tr=tr
#
#     #Slice Timing: sequential ascending
#     slice_timing = Node(interface=spm.SliceTiming(), name="slice_timing")
#     slice_timing.inputs.time_repetition = tr
#     slice_timing.inputs.ref_slice = 1
#
#     #Realignment - 6 parameters - realign to first image of very first series.
#     realign = Node(interface=spm.Realign(), name="realign")
#     realign.inputs.register_to_mean = True
#
#     #Plot Realignment
#     plot_realign = Node(interface=Plot_Realignment_Parameters(), name="plot_realign")
#
#     #Artifact Detection
#     art = Node(interface=ra.ArtifactDetect(), name="art")
#     art.inputs.use_differences      = [True,False]
#     art.inputs.use_norm             = True
#     art.inputs.norm_threshold       = 1
#     art.inputs.zintensity_threshold = 3
#     art.inputs.mask_type            = 'file'
#     art.inputs.parameter_source     = 'SPM'
#
#     #Coregister - 12 parameters, cost function = 'nmi', fwhm 7, interpolate, don't mask
#     #anatomical to functional mean across all available data.
#     coregister = Node(interface=spm.Coregister(), name="coregister")
#     coregister.inputs.jobtype = 'estimate'
#
#     # Segment structural, gray/white/csf,mni,
#     segment = Node(interface=spm.Segment(), name="segment")
#     segment.inputs.save_bias_corrected = True
#
#     #Normalize - structural to MNI - then apply this to the coregistered functionals
#     normalize = Node(interface=spm.Normalize(), name = "normalize")
#     normalize.inputs.template = os.path.abspath(t1_template_file)
#
#     #Plot normalization Check
#     plot_normalization_check = Node(interface=Plot_Coregistration_Montage(), name="plot_normalization_check")
#     plot_normalization_check.inputs.canonical_img = canonical_file
#
#     #Create Mask
#     compute_mask = Node(interface=ComputeMask(), name="compute_mask")
#     #remove lower 5% of histogram of mean image
#     compute_mask.inputs.m = .05
#
#     #Smooth
#     #implicit masking (.im) = 0, dtype = 0
#     smooth = Node(interface=spm.Smooth(), name = "smooth")
#     fwhmlist = [0,5,8]
#     smooth.iterables = ('fwhm',fwhmlist)
#
#     #Create Covariate matrix
#     make_covariates = Node(interface=Create_Covariates(), name="make_covariates")
#
#     ###############################
#     ## Create Pipeline
#     ###############################
#
#     Preprocessed = Workflow(name="Preprocessed")
#     Preprocessed.base_dir = os.path.abspath(data_dir + '/' + subject_id + '/Functional')
#
#     Preprocessed.connect([(ds, ta, [(('func', get_n_slices), "n_slices")]),
#     					(ta, slice_timing, [("ta", "time_acquisition")]),
#     					(ds, slice_timing, [('func', 'in_files'),
#     										(('func', get_n_slices), "num_slices"),
#     										(('func', get_slice_order), "slice_order"),
#     										]),
#     					(slice_timing, realign, [('timecorrected_files', 'in_files')]),
#     					(realign, compute_mask, [('mean_image','mean_volume')]),
#     					(realign,coregister, [('mean_image', 'target')]),
#     					(ds,coregister, [('struc', 'source')]),
#     					(coregister,segment, [('coregistered_source', 'data')]),
#     					(segment, normalize, [('transformation_mat','parameter_file'),
#     										('bias_corrected_image', 'source'),]),
#     					(realign,normalize, [('realigned_files', 'apply_to_files'),
#     										(('realigned_files', get_vox_dims), 'write_voxel_sizes')]),
#     					(normalize, smooth, [('normalized_files', 'in_files')]),
#     					(compute_mask,art,[('brain_mask','mask_file')]),
#     					(realign,art,[('realignment_parameters','realignment_parameters')]),
#     					(realign,art,[('realigned_files','realigned_files')]),
#     					(realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
#     					(normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
#     					(realign, make_covariates, [('realignment_parameters', 'realignment_parameters')]),
#     					(art, make_covariates, [('outlier_files', 'spike_id')]),
#     					])
#     return Preprocessed

def Couple_Preproc_Pipeline(base_dir=None, output_dir=None, subject_id=None, spm_path=None):
    """ Create a preprocessing workflow for the Couples Conflict Study using nipype

    Args:
        base_dir: path to data folder where raw subject folder is located
        output_dir: path to where key output files should be saved
        subject_id: subject_id (str)
        spm_path: path to spm folder

    Returns:
        workflow: a nipype workflow that can be run

    """

    from nipype.interfaces.dcm2nii import Dcm2nii
    from nipype.interfaces.fsl import Merge, TOPUP, ApplyTOPUP
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as util
    from nipype.interfaces.utility import Merge as Merge_List
    from nipype.pipeline.engine import Node, Workflow
    from nipype.interfaces.fsl.maths import UnaryMaths
    from nipype.interfaces.nipy.preprocess import Trim
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces import spm
    from nipype.interfaces.spm import Normalize12
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.nipy.preprocess import ComputeMask
    import nipype.interfaces.matlab as mlab
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates
    from cosanlab_preproc.utils import get_resource_path, get_vox_dims, get_n_volumes
    import os
    import glob

    ########################################
    ## Setup Paths and Nodes
    ########################################

    # Specify Paths
    canonical_file = os.path.join(spm_path,'canonical','single_subj_T1.nii')
    template_file = os.path.join(spm_path,'tpm','TPM.nii')

    # Set the way matlab should be called
    mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
    mlab.MatlabCommand.set_default_paths(spm_path)

    # Get File Names for different types of scans.  Parse into separate processing streams
    datasource = Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=['struct', 'ap', 'pa']),name='datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.field_template = {'struct':'%s/Study*/t1w_32ch_mpr_08mm*',
                                        'ap':'%s/Study*/distortion_corr_32ch_ap*',
                                        'pa':'%s/Study*/distortion_corr_32ch_pa*'}
    datasource.inputs.template_args = {'struct':[['subject_id']],'ap':[['subject_id']],'pa':[['subject_id']]}
    datasource.inputs.subject_id = subject_id
    datasource.inputs.sort_filelist=True

    # iterate over functional scans to define paths
    scan_file_list = glob.glob(os.path.join(base_dir,subject_id,'Study*','*'))
    func_list = [s for s in scan_file_list if "romcon_ap_32ch_mb8" in s]
    func_list = [s for s in func_list if "SBRef" not in s] # Exclude sbref for now.
    func_source = Node(interface=util.IdentityInterface(fields=['scan']),name="func_source")
    func_source.iterables = ('scan', func_list)

    # Create Separate Converter Nodes for each different type of file. (dist corr scans need to be done before functional)
    ap_dcm2nii = Node(interface = Dcm2nii(),name='ap_dcm2nii')
    ap_dcm2nii.inputs.gzip_output = True
    ap_dcm2nii.inputs.output_dir = '.'
    ap_dcm2nii.inputs.date_in_filename = False

    pa_dcm2nii = Node(interface = Dcm2nii(),name='pa_dcm2nii')
    pa_dcm2nii.inputs.gzip_output = True
    pa_dcm2nii.inputs.output_dir = '.'
    pa_dcm2nii.inputs.date_in_filename = False

    f_dcm2nii = Node(interface = Dcm2nii(),name='f_dcm2nii')
    f_dcm2nii.inputs.gzip_output = True
    f_dcm2nii.inputs.output_dir = '.'
    f_dcm2nii.inputs.date_in_filename = False

    s_dcm2nii = Node(interface = Dcm2nii(),name='s_dcm2nii')
    s_dcm2nii.inputs.gzip_output = True
    s_dcm2nii.inputs.output_dir = '.'
    s_dcm2nii.inputs.date_in_filename = False

    ########################################
    ## Setup Nodes for distortion correction
    ########################################

    # merge output files into list
    merge_to_file_list = Node(interface=Merge_List(2), infields=['in1','in2'], name='merge_to_file_list')

    # fsl merge AP + PA files (depends on direction)
    merger = Node(interface=Merge(dimension = 't'),name='merger')
    merger.inputs.output_type = 'NIFTI_GZ'

    # use topup to create distortion correction map
    topup = Node(interface=TOPUP(), name='topup')
    topup.inputs.encoding_file = os.path.join(get_resource_path(),'epi_params_APPA_MB8.txt')
    topup.inputs.output_type = "NIFTI_GZ"
    topup.inputs.config = 'b02b0.cnf'

    # apply topup to all functional images
    apply_topup = Node(interface = ApplyTOPUP(), name='apply_topup')
    apply_topup.inputs.in_index = [1]
    apply_topup.inputs.encoding_file = os.path.join(get_resource_path(),'epi_params_APPA_MB8.txt')
    apply_topup.inputs.output_type = "NIFTI_GZ"
    apply_topup.inputs.method = 'jac'
    apply_topup.inputs.interp = 'spline'

    # Clear out Zeros from spline interpolation using absolute value.
    abs_maths = Node(interface=UnaryMaths(), name='abs_maths')
    abs_maths.inputs.operation = 'abs'

    ########################################
    ## Preprocessing
    ########################################

    # Trim - remove first 10 TRs
    n_vols = 10
    trim = Node(interface = Trim(), name='trim')
    trim.inputs.begin_index=n_vols

    #Realignment - 6 parameters - realign to first image of very first series.
    realign = Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    #Coregister - 12 parameters
    coregister = Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype = 'estwrite'

    #Plot Realignment
    plot_realign = Node(interface=Plot_Realignment_Parameters(), name="plot_realign")

    #Artifact Detection
    art = Node(interface=ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,False]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'

    # Gunzip - unzip the functional and structural images
    gunzip_struc = Node(Gunzip(), name="gunzip_struc")
    gunzip_func = Node(Gunzip(), name="gunzip_func")

    # Normalize - normalizes functional and structural images to the MNI template
    normalize = Node(interface=Normalize12(jobtype='estwrite',tpm=template_file),
                     name="normalize")

    #Plot normalization Check
    plot_normalization_check = Node(interface=Plot_Coregistration_Montage(), name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = canonical_file

    #Plot QA
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")

    #Create Mask
    compute_mask = Node(interface=ComputeMask(), name="compute_mask")

    #remove lower 5% of histogram of mean image
    compute_mask.inputs.m = .05

    #Smooth
    #implicit masking (.im) = 0, dtype = 0
    smooth = Node(interface=spm.Smooth(), name = "smooth")
    smooth.inputs.fwhm=6

    #Create Covariate matrix
    make_cov = Node(interface=Create_Covariates(), name="make_cov")

    # Create a datasink to clean up output files
    datasink = Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id

    ########################################
    # Create Workflow
    ########################################

    workflow = Workflow(name = 'Preprocessed')
    workflow.base_dir = os.path.join(base_dir,subject_id)
    workflow.connect([(datasource, ap_dcm2nii,[('ap','source_dir')]),
                        (datasource, pa_dcm2nii,[('pa','source_dir')]),
                        (datasource, s_dcm2nii,[('struct','source_dir')]),
                        (func_source, f_dcm2nii,[('scan','source_dir')]),
                        (ap_dcm2nii, merge_to_file_list,[('converted_files','in1')]),
                        (pa_dcm2nii, merge_to_file_list,[('converted_files','in2')]),
                        (merge_to_file_list, merger,[('out','in_files')]),
                        (merger, topup,[('merged_file','in_file')]),
                        (topup, apply_topup,[('out_fieldcoef','in_topup_fieldcoef'),
                                            ('out_movpar','in_topup_movpar')]),
                        (f_dcm2nii, trim,[('converted_files','in_file')]),
                        (trim, apply_topup,[('out_file','in_files')]),
                        (apply_topup, abs_maths,[('out_corrected','in_file')]),
                        (abs_maths, gunzip_func, [('out_file', 'in_file')]),
                        (gunzip_func, realign, [('out_file', 'in_files')]),
                        (s_dcm2nii, gunzip_struc,[('converted_files','in_file')]),
                        (gunzip_struc,coregister, [('out_file', 'source')]),
                        (coregister, normalize,[('coregistered_source','image_to_align')]),
                        (realign, plot_qa, [('realigned_files','dat_img')]),
                        (realign,coregister, [('mean_image', 'target'),
                                              ('realigned_files', 'apply_to_files')]),
                        (realign,normalize, [(('mean_image', get_vox_dims), 'write_voxel_sizes')]),
                        (coregister,normalize, [('coregistered_files', 'apply_to_files')]),
                        (normalize, smooth, [('normalized_files', 'in_files')]),
                        (realign, compute_mask, [('mean_image','mean_volume')]),
                        (compute_mask,art,[('brain_mask','mask_file')]),
                        (realign,art,[('realignment_parameters','realignment_parameters'),
                                      ('realigned_files','realigned_files')]),
                        (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
                        (normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
                        (realign, make_cov, [('realignment_parameters', 'realignment_parameters')]),
                        (art, make_cov, [('outlier_files', 'spike_id')]),
                        (normalize, datasink, [('normalized_files', 'structural.@normalize')]),
                        (coregister, datasink, [('coregistered_source', 'structural.@struct')]),
                        (topup, datasink, [('out_fieldcoef', 'distortion.@fieldcoef')]),
                        (topup, datasink, [('out_movpar', 'distortion.@movpar')]),
                        (smooth, datasink, [('smoothed_files', 'functional.@smooth')]),
                        (plot_realign, datasink, [('plot', 'functional.@plot_realign')]),
                        (plot_normalization_check, datasink, [('plot', 'functional.@plot_normalization')]),
                        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
                        (make_cov, datasink, [('covariates', 'functional.@covariates')])])
    return workflow

def TV_Preproc_Pipeline_OLD(base_dir=None, output_dir=None, subject_id=None, spm_path=None):
    """ Create a preprocessing workflow for the Couples Conflict Study using nipype

    Args:
        base_dir: path to data folder where raw subject folder is located
        output_dir: path to where key output files should be saved
        subject_id: subject_id (str)
        spm_path: path to spm folder

    Returns:
        workflow: a nipype workflow that can be run

    """

    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as util
    from nipype.interfaces.utility import Merge as Merge_List
    from nipype.pipeline.engine import Node, Workflow
    from nipype.interfaces.fsl.maths import UnaryMaths
    from nipype.interfaces.nipy.preprocess import Trim
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces import spm
    from nipype.interfaces.spm import Normalize12
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.nipy.preprocess import ComputeMask
    import nipype.interfaces.matlab as mlab
    from cosanlab_preproc.utils import get_resource_path, get_vox_dims, get_n_volumes
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Realignment_Parameters, Create_Covariates, Plot_Quality_Control
    import os
    import glob

    ########################################
    ## Setup Paths and Nodes
    ########################################

    # Specify Paths
    canonical_file = os.path.join(spm_path,'canonical','single_subj_T1.nii')
    template_file = os.path.join(spm_path,'tpm','TPM.nii')

    # Set the way matlab should be called
    mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
    mlab.MatlabCommand.set_default_paths(spm_path)

    # Get File Names for different types of scans.  Parse into separate processing streams
    datasource = Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=[
                'struct', 'func']),name='datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.field_template = {'struct':'%s/T1.nii.gz',
                                        'func':'%s/*ep*.nii.gz'}
    datasource.inputs.template_args = {'struct':[['subject_id']],
                                       'func':[['subject_id']]}
    datasource.inputs.subject_id = subject_id
    datasource.inputs.sort_filelist=True

    # iterate over functional scans to define paths
    func_source = Node(interface=util.IdentityInterface(fields=['scan']),name="func_source")
    func_source.iterables = ('scan', glob.glob(os.path.join(base_dir,subject_id,'*ep*nii.gz')))


    ########################################
    ## Preprocessing
    ########################################

    # Trim - remove first 5 TRs
    n_vols = 5
    trim = Node(interface = Trim(), name='trim')
    trim.inputs.begin_index=n_vols

    #Realignment - 6 parameters - realign to first image of very first series.
    realign = Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    #Coregister - 12 parameters
    coregister = Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype = 'estwrite'

    #Plot Realignment
    plot_realign = Node(interface=Plot_Realignment_Parameters(), name="plot_realign")

    #Artifact Detection
    art = Node(interface=ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,False]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'

    # Gunzip - unzip the functional and structural images
    gunzip_struc = Node(Gunzip(), name="gunzip_struc")
    gunzip_func = Node(Gunzip(), name="gunzip_func")

    # Normalize - normalizes functional and structural images to the MNI template
    normalize = Node(interface=Normalize12(jobtype='estwrite',tpm=template_file),
                     name="normalize")

    #Plot normalization Check
    plot_normalization_check = Node(interface=Plot_Coregistration_Montage(), name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = canonical_file

    #Plot QA
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")

    #Create Mask
    compute_mask = Node(interface=ComputeMask(), name="compute_mask")
    #remove lower 5% of histogram of mean image
    compute_mask.inputs.m = .05

    #Smooth
    #implicit masking (.im) = 0, dtype = 0
    smooth = Node(interface=spm.Smooth(), name = "smooth")
    smooth.inputs.fwhm=6

    #Create Covariate matrix
    make_cov = Node(interface=Create_Covariates(), name="make_cov")

    #Plot Quality Control Check
    quality_control = Node(interface=Plot_Quality_Control(), name='quality_control')

    # Create a datasink to clean up output files
    datasink = Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id

    ########################################
    # Create Workflow
    ########################################

    workflow = Workflow(name = 'Preprocessed')
    workflow.base_dir = os.path.join(base_dir,subject_id)
    workflow.connect([(datasource, gunzip_struc,[('struct','in_file')]),
                        (func_source, trim,[('scan','in_file')]),
                        (trim, gunzip_func,[('out_file','in_file')]),
                        (gunzip_func, realign, [('out_file', 'in_files')]),
                        (realign, quality_control, [('realigned_files', 'dat_img')]),
                        (gunzip_struc,coregister, [('out_file', 'source')]),
                        (coregister, normalize,[('coregistered_source','image_to_align')]),
                        (realign,coregister, [('mean_image', 'target'),
                                              ('realigned_files', 'apply_to_files')]),
                        (realign,normalize, [(('mean_image', get_vox_dims), 'write_voxel_sizes')]),
                        (coregister,normalize, [('coregistered_files', 'apply_to_files')]),
                        (normalize, smooth, [('normalized_files', 'in_files')]),
                        (realign, compute_mask, [('mean_image','mean_volume')]),
                        (compute_mask,art,[('brain_mask','mask_file')]),
                        (realign,art,[('realignment_parameters','realignment_parameters'),
                                      ('realigned_files','realigned_files')]),
                        (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
                        (normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
                        (realign, make_cov, [('realignment_parameters', 'realignment_parameters')]),
                        (art, make_cov, [('outlier_files', 'spike_id')]),
                        (normalize, datasink, [('normalized_files', 'structural.@normalize')]),
                        (coregister, datasink, [('coregistered_source', 'structural.@struct')]),
                        (smooth, datasink, [('smoothed_files', 'functional.@smooth')]),
                        (plot_realign, datasink, [('plot', 'functional.@plot_realign')]),
                        (plot_normalization_check, datasink, [('plot', 'functional.@plot_normalization')]),
                        (make_cov, datasink, [('covariates', 'functional.@covariates')]),
                        (quality_control, datasink, [('plot', 'functional.@quality_control')])
                     ])
    return workflow

def TV_Preproc_Pipeline(base_dir=None, output_dir=None, resources_dir=None, subject_id=None, volsToTrim = 5, smoothingKernel = 4):

    """
    Create a nipype preprocessing workflow to analyze data from the TV study.
    THIS IS DESIGNED TO BE RUN IN A DOCKER CONTAINER WITH FSL AND ANTS
    Pre-processing steps include:
    Trimming extra scans (nipy)
    Realignment/Motion Correction (fsl)
    Artifact Detection (nipype)
    Brain Extraction + Bias Correction (ANTs)
    Coregistration (rigid) (ANTs)
    Normalization to MNI 152 2mm (non-linear) (ANTs)
    Quality Control figure generation:
        - Realignment parameters
        - Quality check of mean signal, sd and frame differences
        - Normalization check

    Args:
        base_dir: path to raw data folder with subjects listed as sub-folders
        output_dir: path where final outputted files and figures should go
        resources_dir: path where template files for MNI and ANTs live
        subject_id: subject to run (should match folder name)

    Return:
        workflow: A complete nipype workflow
    """

    import os
    from glob import glob
    import matplotlib
    matplotlib.use('Agg')
    from nipype.interfaces.io import DataSink, DataGrabber
    from nipype.interfaces.utility import Merge, IdentityInterface
    from nipype.pipeline.engine import Node, Workflow
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates
    from cosanlab_preproc.utils import get_resource_path
    from nipype.interfaces.nipy.preprocess import Trim, ComputeMask
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.ants import Registration, ApplyTransforms
    from nipype.interfaces.fsl import MCFLIRT
    from nipype.interfaces.fsl.maths import MeanImage
    from nipype.interfaces.fsl.utils import Smooth

    ###################################
    ### GLOBALS, PATHS ###
    ###################################
    MNItemplate = os.path.join(get_resource_path,'MNI152_T1_2mm_brain.nii.gz')
    MNItemplatehasskull = os.path.join(get_resource_path,'MNI152_T1_2mm.nii.gz')
    bet_ants_template = os.path.join(get_resource_path,'OASIS_template.nii.gz')
    bet_ants_prob_mask = os.path.join(get_resource_path,'OASIS_BrainCerebellumProbabilityMask.nii.gz')
    bet_ants_registration_mask = os.path.join(get_resource_path,'OASIS_BrainCerebellumRegistrationMask.nii.gz')
    bet_ants_extraction_mask = os.path.join(get_resource_path,'OASIS_BrainCerebellumExtractionMask.nii.gz')

    ###################################
    ### DATA INPUT ###
    ###################################
    #Create a datagrabber that takes a subid as input and creates func and struct dirs
    datasource = Node(DataGrabber(
        infields=['subject_id'],
        outfields = ['func','struct']),
        name = 'datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.subject_id = subject_id
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    datasource.inputs.field_template = {'struct': '%s/T1.nii',
                                        'func': '%s/*ep*.nii'}
    datasource.inputs.template_args = {'struct' :[['subject_id']],
                                       'func': [['subject_id']]}

    #Then grab all epis using an Identity Interface which is an iterable node
    func_scans = Node(IdentityInterface(fields=['scan']),name='func_scans')
    func_scans.inputs.subject_id  = subject_id
    func_scans.iterables = ('scan', glob(os.path.join(base_dir,subject_id,'*ep*.nii')))

    ###################################
    ### TRIM ###
    ###################################
    trim = Node(Trim(), name = 'trim')
    trim.inputs.begin_index = volsToTrim

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
    brain_extraction_ants.inputs.dimension = 3
    brain_extraction_ants.inputs.use_floatingpoint_precision = 1
    brain_extraction_ants.inputs.num_threads = 12
    brain_extraction_ants.inputs.brain_probability_mask = bet_ants_prob_mask
    brain_extraction_ants.inputs.brain_template = bet_ants_template
    brain_extraction_ants.inputs.extraction_registration_mask = bet_ants_registration_mask

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
    coregistration.inputs.num_threads = 12
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
    normalization.inputs.num_threads=12
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
    normalization.inputs.terminal_output='stream'
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
    #The nodes above compute the required transformation matrices but don't actually apply them to the data. Here we're merging both matrices and applying them in a single transformation step to reduce the amount of data interpolation.

    merge_transforms = Node(Merge(2), iterfield=['in2'], name ='merge_transforms')

    apply_transforms = Node(ApplyTransforms(),iterfield=['input_image'],name='apply_transforms')
    apply_transforms.inputs.input_image_type = 3
    apply_transforms.inputs.float = False
    apply_transforms.inputs.num_threads = 12
    apply_transforms.inputs.environ = {}
    apply_transforms.inputs.interpolation = 'BSpline'
    apply_transforms.inputs.invert_transform_flags = [False, False]
    apply_transforms.inputs.terminal_output = 'stream'
    apply_transforms.inputs.reference_image = MNItemplate

    #Use FSL for smoothing
    smooth = Node(Smooth(),name='smooth')
    smooth.inputs.sigma = smoothingKernel

    ###################################
    ### PLOTS ###
    ###################################

    plot_realign = Node(Plot_Realignment_Parameters(),name="plot_realign")
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")
    plot_normalization_check = Node(Plot_Coregistration_Montage(),name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = MNItemplatehasskull

    ###################################
    ### DATA OUTPUT ###
    ###################################
    #Collect all final outputs in the output dir and get rid of file name additions
    datasink = Node(DataSink(),name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
    datasink.inputs.substitutions = [('_scan_..data..fmriData..' + subject_id + '..','')]


    ###################################
    ### HOOK IT ALL CAPTAIN! ###
    ###################################
    workflow = Workflow(name='Preprocessing')
    workflow.base_dir = os.path.join(base_dir,subject_id)

    workflow.connect([
        (func_scans, trim, [('scan','in_file')]),
        (trim, realign_fsl, [('out_file','in_file')]),
        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
        (realign_fsl, plot_qa, [('out_file','dat_img')]),
        (realign_fsl, art, [('out_file','realigned_files'),
                           ('par_file','realignment_parameters')]),
        (realign_fsl, mean_epi, [('out_file','in_file')]),
        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
        (mean_epi, compute_mask, [('out_file','mean_volume')]),
        (compute_mask, art, [('brain_mask','mask_file')]),
        (art, make_cov, [('outlier_files','spike_id')]),
        (datasource, brain_extraction_ants, [('struct','anatomical_image')]),
        (brain_extraction_ants, coregistration, [('BrainExtractionBrain','fixed_image')]),
        (mean_epi, coregistration, [('out_file','moving_image')]),
        (brain_extraction_ants, normalization, [('BrainExtractionBrain','moving_image')]),
        (coregistration, merge_transforms, [('composite_transform','in2')]),
        (normalization, merge_transforms, [('composite_transform','in1')]),
        (merge_transforms, apply_transforms, [('out','transforms')]),
        (realign_fsl, apply_transforms, [('out_file','input_image')]),
        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')]),
        (apply_transforms, datasink, [('output_image', 'functional.@normalize')]),
        (apply_transforms, smooth, [('output_image','in_file')]),
        (smooth, datasink, [('smoothed_file','functional.@smooth')]),
        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
        (make_cov, datasink, [('covariates','functional.@covariates')]),
        (brain_extraction_ants, datasink, [('BrainExtractionBrain','structural.@struct')]),
        (normalization, datasink, [('warped_image','structural.@normalize')])
    ])

    if not os.path.exists(os.path.join(output_dir,'Preprocsteps.png')):
        workflow.write_graph(dotfilename=os.path.join(output_dir,'Preprocsteps'),format='png')

    return workflow

def ScanParams_Preproc_Pipeline(base_dir=None, output_dir=None, subject_id=None, smoothingKernel=4):

    """
    Create a nipype preprocessing workflow to analyze data from the scanParams testing acquisitions.
    THIS IS DESIGNED TO BE RUN IN A DOCKER CONTAINER WITH FSL AND ANTS
    Pre-processing steps include:
    Realignment/Motion Correction (fsl)
    Artifact Detection (nipype)
    Brain Extraction + Bias Correction (ANTs)
    Coregistration (rigid) (ANTs)
    Normalization to MNI 152 2mm (non-linear) (ANTs)
    Quality Control figure generation:
        - Realignment parameters
        - Quality check of mean signal, sd and frame differences
        - Normalization check
    Makes 3 design matrices: standard block design (Right, Left), "mvpa" design (R1, R2, R3, L1, L2,L3), contrast block (R-L)
    Fits 3 first level models (REQUIRES NLTOOLS!)

    Args:
        base_dir: path to raw data folder with subjects listed as sub-folders
        output_dir: path where final outputted files and figures should go
        resources_dir: path where template files for MNI and ANTs live
        subject_id: subject to run (should match folder name)

    Return:
        workflow: A complete nipype workflow
    """
    import os
    from glob import glob
    import matplotlib
    matplotlib.use('Agg')
    from nipype.interfaces.io import DataSink, DataGrabber
    from nipype.interfaces.utility import Merge, IdentityInterface, Function
    from nipype.pipeline.engine import Node, Workflow
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates, Build_Xmat, GLM
    from cosanlab_preproc.utils import get_resource_path
    from nipype.interfaces.nipy.preprocess import ComputeMask
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.ants import Registration, ApplyTransforms
    from nipype.interfaces.fsl import MCFLIRT
    from nipype.interfaces.fsl.maths import MeanImage
    from nipype.interfaces.fsl.utils import Smooth

    ###################################
    ### GLOBALS, PATHS ###
    ###################################
    MNItemplate = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
    MNItemplatehasskull = os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
    bet_ants_template = os.path.join(get_resource_path(),'OASIS_template.nii.gz')
    bet_ants_prob_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumProbabilityMask.nii.gz')
    bet_ants_registration_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumRegistrationMask.nii.gz')
    #bet_ants_extraction_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumExtractionMask.nii.gz')

    ###################################
    ### DATA INPUT ###
    ###################################
    #Create a datagrabber that takes a subid as input and creates func and struct dirs
    datasource = Node(DataGrabber(
        infields=['subject_id'],
        outfields = ['func','struct']),
        name = 'datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.subject_id = subject_id
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    datasource.inputs.field_template = {'struct': '%s/T1.nii.gz',
                                        'func': '%s/*mm.nii.gz'}
    datasource.inputs.template_args = {'struct' :[['subject_id']],
                                       'func': [['subject_id']]}

    #Then grab all epis using an Identity Interface which is an iterable node
    func_scans = Node(IdentityInterface(fields=['scan']),name='func_scans')
    func_scans.inputs.subject_id  = subject_id
    func_scans.iterables = ('scan', glob(os.path.join(base_dir,subject_id,'*mm.nii.gz')))

    ###################################
    ### TR GRABBER ###
    ###################################
    def getTR(fName):
        '''
        Gets TR length of scan by reading in the nifti header.
        '''
        import nibabel as nib
        f = nib.load(fName)
        return round(f.header.get_zooms()[-1]*1000)/1000

    get_tr = Node(interface=Function(input_names=['fName'],
                                     output_names=['TR'],
                                     function=getTR),
                                     name='get_tr')

    ###################################
    ### ONSETS GRABBER ###
    ###################################
    def getOnsets(fName):
        '''
        Gets onsets txt file given path to a .nii.gz file.
        Assumes both files are named the same.
        '''
        import os
        fPieces = os.path.split(fName)
        scanId = fPieces[-1].split('.nii.gz')[0]
        return os.path.join(fPieces[0],scanId+'.txt')

    get_onsets = Node(interface=Function(input_names=['fName'],
                                     output_names=['onsetsFile'],
                                     function=getOnsets),
                                     name='get_onsets')

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
    brain_extraction_ants.inputs.dimension = 3
    brain_extraction_ants.inputs.use_floatingpoint_precision = 1
    brain_extraction_ants.inputs.num_threads = 12
    brain_extraction_ants.inputs.brain_probability_mask = bet_ants_prob_mask
    brain_extraction_ants.inputs.keep_temporary_files = 1
    brain_extraction_ants.inputs.brain_template = bet_ants_template
    brain_extraction_ants.inputs.extraction_registration_mask = bet_ants_registration_mask

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
    coregistration.inputs.num_threads = 12
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
    normalization.inputs.num_threads=12
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
    normalization.inputs.terminal_output='stream'
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
    #The nodes above compute the required transformation matrices but don't actually apply them to the data. Here we're merging both matrices and applying them in a single transformation step to reduce the amount of data interpolation.

    merge_transforms = Node(Merge(2), iterfield=['in2'], name ='merge_transforms')

    apply_transforms = Node(ApplyTransforms(),iterfield=['input_image'],name='apply_transforms')
    apply_transforms.inputs.input_image_type = 3
    apply_transforms.inputs.float = False
    apply_transforms.inputs.num_threads = 12
    apply_transforms.inputs.environ = {}
    apply_transforms.inputs.interpolation = 'BSpline'
    apply_transforms.inputs.invert_transform_flags = [False, False]
    apply_transforms.inputs.terminal_output = 'stream'
    apply_transforms.inputs.reference_image = MNItemplate

    #Use FSL for smoothing
    smooth = Node(Smooth(),name='smooth')
    smooth.inputs.sigma = smoothingKernel

    ###################################
    ### PLOTS ###
    ###################################

    plot_realign = Node(Plot_Realignment_Parameters(),name="plot_realign")
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")
    plot_normalization_check = Node(Plot_Coregistration_Montage(),name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = MNItemplatehasskull

    ###################################
    ### Xmat ###
    ###################################

    build_xmat = Node(Build_Xmat(),name="build_xmat")
    build_xmat.inputs.header = False
    build_xmat.inputs.delim = '\t'
    build_xmat.inputs.fillNa = True
    build_xmat.inputs.dur = 8

    ###################################
    ### CONTRAST Xmat ###
    ###################################
    def buildContrastXmat(covFile,onsetsFile,TR):

        import matplotlib
        matplotlib.use('Agg')
        from nipy.modalities.fmri.hemodynamic_models import glover_hrf
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        dur = np.ceil(8./TR)
        header = None
        delim = '\t'
        hrf = glover_hrf(tr = TR,oversampling=1)

        #Just a single file
        C = pd.read_csv(covFile)
        C['intercept'] = 1
        O = pd.read_csv(onsetsFile,header=header,delimiter=delim)
        if header is None:
            if isinstance(O.iloc[0,0],str):
                O.columns = ['Stim','Onset']
            else:
                O.columns = ['Onset','Stim']
        O['Onset'] = O['Onset'].apply(lambda x: int(np.floor(x/TR)))

        #Build dummy codes
        #Subtract one from onsets row, because pd DFs are 0-indexed but TRs are 1-indexed
        X = pd.DataFrame(columns=['contrast'],data=np.zeros([C.shape[0],1]))
        for i, row in O.iterrows():
            #do dur-1 for slicing because .ix includes the last element of the slice
            if row['Stim'] == 'right':
                X.ix[row['Onset']-1:(row['Onset']-1)+dur-1,'contrast'] = 1
            else:
                X.ix[row['Onset']-1:(row['Onset']-1)+dur-1,'contrast'] = -1

        X['contrast']= np.convolve(hrf,X.contrast.values)[:X.shape[0]]
        X = pd.concat([X,C],axis=1)
        X = X.fillna(0)

        matplotlib.rcParams['axes.edgecolor'] = 'black'
        matplotlib.rcParams['axes.linewidth'] = 2
        fig, ax = plt.subplots(1,figsize=(12,10))

        ax = sns.heatmap(X,cmap='gray', cbar=False,ax=ax);

        for _, spine in ax.spines.items():
            spine.set_visible(True)
        for i, label in enumerate(ax.get_yticklabels()):
            if i > 0 and i < X.shape[0]:
                label.set_visible(False)

        plotFile = 'Xmat_con.png'
        fig.savefig(plotFile)
        plt.close(fig)
        del fig

        xmatFile = 'Xmat_con.csv'
        X.to_csv(xmatFile,index=False)

        return plotFile, xmatFile

    build_xmat_con = Node(interface=Function(input_names=['covFile','onsetsFile','TR'],
                                     output_names=['plot','xmat'],
                                     function=buildContrastXmat),
                                     name='build_xmat_con')


    ###################################
    ### MVPA Xmat ###
    ###################################
    def buildMVPAXmat(covFile,onsetsFile,TR):

        import matplotlib
        matplotlib.use('Agg')
        from nipy.modalities.fmri.hemodynamic_models import glover_hrf
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        dur = np.ceil(8./TR)
        header = None
        delim = '\t'
        hrf = glover_hrf(tr = TR,oversampling=1)

        #Just a single file
        C = pd.read_csv(covFile)
        C['intercept'] = 1
        O = pd.read_csv(onsetsFile,header=header,delimiter=delim)
        if header is None:
            if isinstance(O.iloc[0,0],str):
                O.columns = ['Stim','Onset']
            else:
                O.columns = ['Onset','Stim']
        O['Onset'] = O['Onset'].apply(lambda x: int(np.floor(x/TR)))

        #Uniquify stims
        leftCount = 1
        rightCount = 1
        newStims = []
        for i, row in O.iterrows():
            if row['Stim'] == 'right':
                newStims.append('right_'+str(rightCount))
                rightCount +=1
            elif row['Stim'] == 'left':
                newStims.append('left_'+str(leftCount))
                leftCount +=1
        O['Stim'] = newStims

        #Build dummy codes
        #Subtract one from onsets row, because pd DFs are 0-indexed but TRs are 1-indexed
        X = pd.DataFrame(columns=O.Stim.unique(),data=np.zeros([C.shape[0],len(O.Stim.unique())]))
        for i, row in O.iterrows():
            #do dur-1 for slicing because .ix includes the last element of the slice
            X.ix[row['Onset']-1:(row['Onset']-1)+dur-1,row['Stim']] = 1
        X = X.reindex_axis(sorted(X.columns), axis=1)

        for i in range(X.shape[1]):
            X.iloc[:,i] = np.convolve(hrf,X.iloc[:,i])[:X.shape[0]]
        X = pd.concat([X,C],axis=1)
        X = X.fillna(0)

        matplotlib.rcParams['axes.edgecolor'] = 'black'
        matplotlib.rcParams['axes.linewidth'] = 2
        fig, ax = plt.subplots(1,figsize=(12,10))

        ax = sns.heatmap(X,cmap='gray', cbar=False,ax=ax);

        for _, spine in ax.spines.items():
            spine.set_visible(True)
        for i, label in enumerate(ax.get_yticklabels()):
            if i > 0 and i < X.shape[0]:
                label.set_visible(False)

        plotFile = 'Xmat_mvpa.png'
        fig.savefig(plotFile)
        plt.close(fig)
        del fig

        xmatFile = 'Xmat_mvpa.csv'
        X.to_csv(xmatFile,index=False)

        return plotFile, xmatFile

    build_xmat_mvpa = Node(interface=Function(input_names=['covFile','onsetsFile','TR'],
                                     output_names=['plot','xmat'],
                                     function=buildMVPAXmat),
                                     name='build_xmat_mvpa')

    ###################################
    ### GLM CONTRAST###
    ###################################
    glm = Node(GLM(),name="glm")
    glm.inputs.detrend = True

    ###################################
    ### GLM CONTRAST###
    ###################################
    glm_con = Node(GLM(),name="glm_con")
    glm_con.inputs.detrend = True
    glm_con.inputs.prependName = 'con'

    ###################################
    ### GLM MVPA###
    ###################################
    glm_mvpa = Node(GLM(),name="glm_mvpa")
    glm_mvpa.inputs.detrend = True
    glm_mvpa.inputs.prependName = 'mvpa'

    ###################################
    ### DATA OUTPUT ###
    ###################################
    #Collect all final outputs in the output dir and get rid of file name additions
    datasink = Node(DataSink(),name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
    datasink.inputs.substitutions = [('_scan_..data..fmriData..' + subject_id + '..','')]

    ###################################
    ### FILE GETTER (used to run alt glms after preproc) ###
    ###################################
    # def fileGetter(subject_id,output_dir):
    #   '''
    #   Gets onsets txt file given path to a .nii.gz file.
    #   Assumes both files are named the same.
    #   '''
    #   import os
    #   #Get final epi

    #   fPieces = os.path.split(fName)
    #   scanId = fPieces[-1].split('.nii.gz')[0]
    #   return os.path.join(fPieces[0],scanId+'.txt')

    # get_onsets = Node(interface=Function(input_names=['fName'],
    #                                output_names=['onsetsFile'],
    #                                function=getOnsets),
    #                                name='get_onsets')

    ###################################
    ### HOOK IT ALL CAPTAIN! ###
    ###################################
    workflow = Workflow(name='Preprocessing')
    workflow.base_dir = os.path.join(base_dir,subject_id)

    workflow.connect([
        (func_scans, realign_fsl, [('scan','in_file')]),
        (func_scans, get_tr, [('scan','fName')]),
        (func_scans, get_onsets, [('scan','fName')]),
        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
        (realign_fsl, plot_qa, [('out_file','dat_img')]),
        (realign_fsl, art, [('out_file','realigned_files'),
                           ('par_file','realignment_parameters')]),
        (realign_fsl, mean_epi, [('out_file','in_file')]),
        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
        (mean_epi, compute_mask, [('out_file','mean_volume')]),
        (compute_mask, art, [('brain_mask','mask_file')]),
        (art, make_cov, [('outlier_files','spike_id')]),
        (datasource, brain_extraction_ants, [('struct','anatomical_image')]),
        (brain_extraction_ants, coregistration, [('BrainExtractionBrain','fixed_image')]),
        (mean_epi, coregistration, [('out_file','moving_image')]),
        (brain_extraction_ants, normalization, [('BrainExtractionBrain','moving_image')]),
        (coregistration, merge_transforms, [('composite_transform','in2')]),
        (normalization, merge_transforms, [('composite_transform','in1')]),
        (merge_transforms, apply_transforms, [('out','transforms')]),
        (realign_fsl, apply_transforms, [('out_file','input_image')]),
        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')]),
        (get_tr, build_xmat, [('TR','TR')]),
        (get_onsets, build_xmat, [('onsetsFile', 'onsetsFile')]),
        (make_cov, build_xmat, [('covariates','covFile')]),
        (build_xmat, datasink, [('xmat', 'functional.@xmat'),
                                ('plot', 'functional.@xmatplot')]),
        (build_xmat, glm, [('xmat','xmatFile')]),
        (smooth, glm, [('smoothed_file','epiFile')]),
        (glm, datasink, [('betaImage','glm.@beta'),
                         ('tstatImage','glm.@tstat'),
                         ('pvalImage','glm.@pval')]),

        (get_tr, build_xmat_con, [('TR','TR')]),
        (get_onsets, build_xmat_con, [('onsetsFile', 'onsetsFile')]),
        (make_cov, build_xmat_con, [('covariates','covFile')]),
        (build_xmat_con, datasink, [('xmat', 'functional.@xmatcon'),
                                ('plot', 'functional.@xmatconplot')]),
        (build_xmat_con, glm_con, [('xmat','xmatFile')]),
        (smooth, glm_con, [('smoothed_file','epiFile')]),
        (glm_con, datasink, [('betaImage','glm.@betacon'),
                         ('tstatImage','glm.@tstatcon'),
                         ('pvalImage','glm.@pvalcon')]),

        (get_tr, build_xmat_mvpa, [('TR','TR')]),
        (get_onsets, build_xmat_mvpa, [('onsetsFile', 'onsetsFile')]),
        (make_cov, build_xmat_mvpa, [('covariates','covFile')]),
        (build_xmat_mvpa, datasink, [('xmat', 'functional.@xmatmvpa'),
                                ('plot', 'functional.@xmatmvpaplot')]),
        (build_xmat_mvpa, glm_mvpa, [('xmat','xmatFile')]),
        (smooth, glm_mvpa, [('smoothed_file','epiFile')]),
        (glm_mvpa, datasink, [('betaImage','glm.@betamvpa'),
                         ('tstatImage','glm.@tstatmvpa'),
                         ('pvalImage','glm.@pvalmvpa')]),


        (apply_transforms, datasink, [('output_image', 'functional.@normalize')]),
        (apply_transforms, smooth, [('output_image','in_file')]),
        (smooth, datasink, [('smoothed_file','functional.@smooth')]),
        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
        (make_cov, datasink, [('covariates','functional.@covariates')]),
        (brain_extraction_ants, datasink, [('BrainExtractionBrain','structural.@struct')]),
        (normalization, datasink, [('warped_image','structural.@normalize')])
    ])



## Old workflow using a single glm
    # workflow.connect([
 #        (func_scans, realign_fsl, [('scan','in_file')]),
 #        (func_scans, get_tr, [('scan','fName')]),
 #        (func_scans, get_onsets, [('scan','fName')]),
 #        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
 #        (realign_fsl, plot_qa, [('out_file','dat_img')]),
 #        (realign_fsl, art, [('out_file','realigned_files'),
 #                           ('par_file','realignment_parameters')]),
 #        (realign_fsl, mean_epi, [('out_file','in_file')]),
 #        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
 #        (mean_epi, compute_mask, [('out_file','mean_volume')]),
 #        (compute_mask, art, [('brain_mask','mask_file')]),
 #        (art, make_cov, [('outlier_files','spike_id')]),
 #        (datasource, brain_extraction_ants, [('struct','anatomical_image')]),
 #        (brain_extraction_ants, coregistration, [('BrainExtractionBrain','fixed_image')]),
 #        (mean_epi, coregistration, [('out_file','moving_image')]),
 #        (brain_extraction_ants, normalization, [('BrainExtractionBrain','moving_image')]),
 #        (coregistration, merge_transforms, [('composite_transform','in2')]),
 #        (normalization, merge_transforms, [('composite_transform','in1')]),
 #        (merge_transforms, apply_transforms, [('out','transforms')]),
 #        (realign_fsl, apply_transforms, [('out_file','input_image')]),
 #        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
 #        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')]),
 #        (get_tr, build_xmat, [('TR','TR')]),
 #        (get_onsets, build_xmat, [('onsetsFile', 'onsetsFile')]),
 #        (make_cov, build_xmat, [('covariates','covFile')]),
 #        (build_xmat, datasink, [('xmat', 'functional.@xmat'),
 #                              ('plot', 'functional.@xmatplot')]),
 #        (build_xmat, glm, [('xmat','xmatFile')]),


 #        (get_tr, build_xmat_con, [('TR','TR')]),
 #        (get_onsets, build_xmat_con, [('onsetsFile', 'onsetsFile')]),
 #        (make_cov, build_xmat_con, [('covariates','covFile')]),
 #        (build_xmat_con, datasink, [('xmat', 'functional.@xmat'),
 #                              ('plot', 'functional.@xmatplot')]),
 #        (build_xmat_con, glm_con, [('xmat','xmatFile')]),
 #        (smooth, glm_con, [('smoothed_file','epiFile')]),

 #        (get_tr, build_xmat_mvpa, [('TR','TR')]),
 #        (get_onsets, build_xmat_mvpa, [('onsetsFile', 'onsetsFile')]),
 #        (make_cov, build_xmat_mvpa, [('covariates','covFile')]),
 #        (build_xmat_mvpa, datasink, [('xmat', 'functional.@xmat'),
 #                              ('plot', 'functional.@xmatplot')]),
 #        (build_xmat_mvpa, glm_mvpa, [('xmat','xmatFile')]),
 #        (smooth, glm_mvpa, [('smoothed_file','epiFile')]),
 #        (glm_mvpa, datasink, [('betaImage','glm.@beta'),
 #                       ('tstatImage','glm.@tstat'),
 #                       ('pvalImage','glm.@pval')]),

 #        (smooth, glm, [('smoothed_file','epiFile')]),
 #        (glm, datasink, [('betaImage','glm.@beta'),
 #                       ('tstatImage','glm.@tstat'),
 #                       ('pvalImage','glm.@pval')]),
 #        (apply_transforms, datasink, [('output_image', 'functional.@normalize')]),
 #        (apply_transforms, smooth, [('output_image','in_file')]),
 #        (smooth, datasink, [('smoothed_file','functional.@smooth')]),
 #        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
 #        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
 #        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
 #        (make_cov, datasink, [('covariates','functional.@covariates')]),
 #        (brain_extraction_ants, datasink, [('BrainExtractionBrain','structural.@struct')]),
 #        (normalization, datasink, [('warped_image','structural.@normalize')])
 #    ])

    if not os.path.exists(os.path.join(output_dir,'Preprocsteps.png')):
        workflow.write_graph(dotfilename=os.path.join(output_dir,'Preprocsteps'),format='png')

    return workflow

def Pinel_Preproc_Pipeline(base_dir=None, output_dir=None, subject_id=None):

    """
    Create a nipype preprocessing workflow to analyze data from the Pinel localizer task.
    Pre-processing steps include:
    Distortion correction (fsl)
    Realignment/Motion Correction (fsl)
    Artifact Detection (nipype)
    Brain Extraction + Bias Correction (ANTs)
    Coregistration (rigid) (ANTs)
    Normalization to MNI 152 2mm (non-linear) (ANTs)
    Qualitry Control figure generation:
        - Realignment parameters
        - Quality check of mean signal, sd and frame differences
        - Normalization check

    Args:
        base_dir: path to raw data folder with subjects listed as sub-folders
        output_dir: path where final outputted files and figures should go
        resources_dir: path where template files for MNI and ANTs live
        subject_id: subject to run (should match folder name)

    Return:
        workflow: A complete nipype workflow
    """
    import os
    from glob import glob
    import matplotlib
    matplotlib.use('Agg')
    import nibabel as nib
    from nipype.interfaces.io import DataSink, DataGrabber
    from nipype.interfaces.utility import Merge, IdentityInterface, Function
    from nipype.pipeline.engine import Node, Workflow
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates, Down_Sample_Precision
    from cosanlab_preproc.utils import get_resource_path
    from bids.grabbids import BIDSLayout
    from nipype.interfaces.nipy.preprocess import ComputeMask
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.ants import Registration, ApplyTransforms
    from nipype.interfaces.fsl import MCFLIRT, TOPUP
    from nipype.interfaces.fsl import ApplyTOPUP as APPLYTOPUP
    from nipype.interfaces.fsl import Merge as MERGE
    from nipype.interfaces.fsl.maths import MeanImage
    from nipype.interfaces.fsl.utils import Smooth

    ###################################
    ### GLOBALS, PATHS ###
    ###################################
    MNItemplate = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
    MNItemplatehasskull = os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
    bet_ants_template = os.path.join(get_resource_path(),'OASIS_template.nii.gz')
    bet_ants_prob_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumProbabilityMask.nii.gz')
    bet_ants_registration_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumRegistrationMask.nii.gz')
    acquistions = [
            'p1Xs2X3mmXsl48Xap',
            'p1Xs4X3mmXsl48Xap',
            'p1Xs6X3mmXsl48Xap',
            'p1Xs8X3mmXsl48Xap',
            ]
    encoding_file = os.path.join(base_dir,'encoding_file.txt')

    ###################################
    ### DATA INPUT ###
    ###################################
    layout = BIDSLayout(base_dir)

    #BIDS needs the 'sub' part of sid removed
    subId = subject_id[4:]
    #Straight up grab the single anat nifti
    anat = layout.get(subject=subId,type='T1w',extensions='.nii.gz')[0].filename

    #Get a list of all epis and wrap them in an iterable node
    funcs = [f.filename for f in layout.get(subject=subId,type='bold',extensions='.nii.gz') if f.acquisition in acquistions]
    func_scans = Node(IdentityInterface(fields=['scan']),name='func_scans')
    func_scans.iterables = ('scan',funcs)

    #Get a list of all distortion correction scans
    dis_corrs = [f.filename for f in layout.get(subject=subId,type='bold',extensions='.nii.gz',task='discorr')]

    #####################################
    ## DISTORTION CORRECTION ##
    #####################################

    #Merge AP and PA distortion correction scans
    merge_discorr = Node(interface=MERGE(dimension='t'),name='merge_discorr')
    merge_discorr.inputs.output_type = 'NIFTI_GZ'
    merge_discorr.inputs.in_files = dis_corrs

    #Create distortion correction map
    topup = Node(interface=TOPUP(),name='topup')
    topup.inputs.output_type = 'NIFTI_GZ'
    topup.inputs.encoding_file = encoding_file

    #Apply distortion correction to other scans
    apply_topup = Node(interface=APPLYTOPUP(),name='apply_topup')
    apply_topup.inputs.output_type = 'NIFTI_GZ'
    apply_topup.inputs.method = 'jac'
    apply_topup.inputs.encoding_file = encoding_file

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
    brain_extraction_ants.inputs.num_threads = 12
    brain_extraction_ants.inputs.brain_probability_mask = bet_ants_prob_mask
    brain_extraction_ants.inputs.keep_temporary_files = 1
    brain_extraction_ants.inputs.brain_template = bet_ants_template
    brain_extraction_ants.inputs.extraction_registration_mask = bet_ants_registration_mask

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
    coregistration.inputs.num_threads = 12
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
    normalization.inputs.num_threads=12
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
    normalization.inputs.terminal_output='stream'
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
    #The nodes above compute the required transformation matrices but don't actually apply them to the data. Here we're merging both matrices and applying them in a single transformation step to reduce the amount of data interpolation.

    merge_transforms = Node(Merge(2), iterfield=['in2'], name ='merge_transforms')

    apply_transforms = Node(ApplyTransforms(),iterfield=['input_image'],name='apply_transforms')
    apply_transforms.inputs.input_image_type = 3
    apply_transforms.inputs.float = False
    apply_transforms.inputs.num_threads = 12
    apply_transforms.inputs.environ = {}
    apply_transforms.inputs.interpolation = 'BSpline'
    apply_transforms.inputs.invert_transform_flags = [False, False]
    apply_transforms.inputs.terminal_output = 'stream'

    apply_transforms.inputs.reference_image = MNItemplate

    #Use FSL for smoothing
    smooth = Node(Smooth(),name='smooth')
    smooth.inputs.sigma = 6.0

    #####################################
    ### DOWNSAMPLE PRECISION ###
    #####################################
    down_samp = Node(Down_Sample_Precision(),name='down_samp')

    ###################################
    ### PLOTS ###
    ###################################

    plot_realign = Node(Plot_Realignment_Parameters(),name="plot_realign")
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")
    plot_normalization_check = Node(Plot_Coregistration_Montage(),name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = MNItemplatehasskull


    ###################################
    ### DATA OUTPUT ###
    ###################################
    #Collect all final outputs in the output dir and get rid of file name additions
    datasink = Node(DataSink(),name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
    datasink.inputs.substitutions = [('_scan_..mnt..Raw..' + subject_id + '..func..',''),
                                    (subject_id+'_acq-p1X',''),
                                    ('X3mmXsl48Xap_bold.nii.gz','')]


    ###################################
    ### HOOK IT ALL CAPTAIN! ###
    ###################################
    workflow = Workflow(name='Preprocessing')
    workflow.base_dir = os.path.join(base_dir,subject_id)

    workflow.connect([
        (merge_discorr, topup, [('merged_file','in_file')]),
        (topup, apply_topup,[('out_fieldcoef','in_topup_fieldcoef'),
                            ('out_movpar','in_topup_movpar')]),
        (func_scans, apply_topup, [('scan','in_files')]),
        (apply_topup, realign_fsl, [('out_corrected','in_file')]),
        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
        (realign_fsl, plot_qa, [('out_file','dat_img')]),
        (realign_fsl, art, [('out_file','realigned_files'),
                           ('par_file','realignment_parameters')]),
        (realign_fsl, mean_epi, [('out_file','in_file')]),
        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
        (mean_epi, compute_mask, [('out_file','mean_volume')]),
        (compute_mask, art, [('brain_mask','mask_file')]),
        (art, make_cov, [('outlier_files','spike_id')]),
        (brain_extraction_ants, coregistration, [('BrainExtractionBrain','fixed_image')]),
        (mean_epi, coregistration, [('out_file','moving_image')]),
        (brain_extraction_ants, normalization, [('BrainExtractionBrain','moving_image')]),
        (coregistration, merge_transforms, [('composite_transform','in2')]),
        (normalization, merge_transforms, [('composite_transform','in1')]),
        (merge_transforms, apply_transforms, [('out','transforms')]),
        (realign_fsl, apply_transforms, [('out_file','input_image')]),
        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')]),
        (apply_transforms, smooth, [('output_image','in_file')]),
        (smooth, down_samp, [('smoothed_file','in_file')]),
        (down_samp, datasink, [('out_file','functional.@down_samp')]),
        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
        (make_cov, datasink, [('covariates','functional.@covariates')]),
        (brain_extraction_ants, datasink, [('BrainExtractionBrain','structural.@struct')]),
        (normalization, datasink, [('warped_image','structural.@normalize')])
        ])

    if not os.path.exists(os.path.join(output_dir,'Preprocsteps.png')):
        workflow.write_graph(dotfilename=os.path.join(output_dir,'Preprocsteps'),format='png')

        return workflow

def NeuroExpSampling_PreProc_Pipeline(base_dir=None, output_dir=None, subject_id=None):

    """
    Create a nipype preprocessing workflow to analyze data from the Neuro-Experience-Sampling scan data.

    This data was originally collected with SMS 8, TR = 419ms.

    Pre-processing steps include:
    Distortion correction (fsl)
    Realignment/Motion Correction (fsl)
    Artifact Detection (nipype)
    Brain Extraction + Bias Correction (ANTs)
    Coregistration (rigid) (ANTs)
    Normalization to MNI 152 2mm (non-linear) (ANTs)
    Low-pass filtering (nltools/nilearn) - filter out high-freq SMS physio noise
    Qualitry Control figure generation:
        - Realignment parameters
        - Quality check of mean signal, sd and frame differences
        - Normalization check

    Args:
        base_dir: path to raw data folder with subjects listed as sub-folders
        output_dir: path where final outputted files and figures should go
        resources_dir: path where template files for MNI and ANTs live
        subject_id: subject to run (should match folder name)

    Return:
        workflow: A complete nipype workflow
    """
    import os
    from glob import glob
    import matplotlib
    matplotlib.use('Agg')
    import nibabel as nib
    from nipype.interfaces.io import DataSink, DataGrabber
    from nipype.interfaces.utility import Merge, IdentityInterface, Function
    from nipype.pipeline.engine import Node, Workflow
    from cosanlab_preproc.interfaces import Plot_Coregistration_Montage, Plot_Quality_Control, Plot_Realignment_Parameters, Create_Covariates, Down_Sample_Precision
    from cosanlab_preproc.utils import get_resource_path
    from bids.grabbids import BIDSLayout
    from nipype.interfaces.nipy.preprocess import ComputeMask
    from nipype.algorithms.rapidart import ArtifactDetect
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.ants import Registration, ApplyTransforms
    from nipype.interfaces.fsl import MCFLIRT, TOPUP
    from nipype.interfaces.fsl import ApplyTOPUP as APPLYTOPUP
    from nipype.interfaces.fsl import Merge as MERGE
    from nipype.interfaces.fsl.maths import MeanImage
    from nipype.interfaces.fsl.utils import Smooth

    ###################################
    ### GLOBALS, PATHS ###
    ###################################
    MNItemplate = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain.nii.gz')
    MNItemplatehasskull = os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')
    bet_ants_template = os.path.join(get_resource_path(),'OASIS_template.nii.gz')
    bet_ants_prob_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumProbabilityMask.nii.gz')
    bet_ants_registration_mask = os.path.join(get_resource_path(),'OASIS_BrainCerebellumRegistrationMask.nii.gz')
    acquistions = [
            'p1Xs2X3mmXsl48Xap',
            'p1Xs4X3mmXsl48Xap',
            'p1Xs6X3mmXsl48Xap',
            'p1Xs8X3mmXsl48Xap',
            ]
    encoding_file = os.path.join(base_dir,'encoding_file.txt')

    ###################################
    ### DATA INPUT ###
    ###################################
    layout = BIDSLayout(base_dir)

    #BIDS needs the 'sub' part of sid removed
    subId = subject_id[4:]
    #Straight up grab the single anat nifti
    anat = layout.get(subject=subId,type='T1w',extensions='.nii.gz')[0].filename

    #Get a list of all epis and wrap them in an iterable node
    funcs = [f.filename for f in layout.get(subject=subId,type='bold',extensions='.nii.gz') if f.acquisition in acquistions]
    func_scans = Node(IdentityInterface(fields=['scan']),name='func_scans')
    func_scans.iterables = ('scan',funcs)

    #Get a list of all distortion correction scans
    dis_corrs = [f.filename for f in layout.get(subject=subId,type='bold',extensions='.nii.gz',task='discorr')]

    #####################################
    ## DISTORTION CORRECTION ##
    #####################################

    #Merge AP and PA distortion correction scans
    merge_discorr = Node(interface=MERGE(dimension='t'),name='merge_discorr')
    merge_discorr.inputs.output_type = 'NIFTI_GZ'
    merge_discorr.inputs.in_files = dis_corrs

    #Create distortion correction map
    topup = Node(interface=TOPUP(),name='topup')
    topup.inputs.output_type = 'NIFTI_GZ'
    topup.inputs.encoding_file = encoding_file

    #Apply distortion correction to other scans
    apply_topup = Node(interface=APPLYTOPUP(),name='apply_topup')
    apply_topup.inputs.output_type = 'NIFTI_GZ'
    apply_topup.inputs.method = 'jac'
    apply_topup.inputs.encoding_file = encoding_file

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
    brain_extraction_ants.inputs.num_threads = 12
    brain_extraction_ants.inputs.brain_probability_mask = bet_ants_prob_mask
    brain_extraction_ants.inputs.keep_temporary_files = 1
    brain_extraction_ants.inputs.brain_template = bet_ants_template
    brain_extraction_ants.inputs.extraction_registration_mask = bet_ants_registration_mask

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
    coregistration.inputs.num_threads = 12
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
    normalization.inputs.num_threads=12
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
    normalization.inputs.terminal_output='stream'
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
    #The nodes above compute the required transformation matrices but don't actually apply them to the data. Here we're merging both matrices and applying them in a single transformation step to reduce the amount of data interpolation.

    merge_transforms = Node(Merge(2), iterfield=['in2'], name ='merge_transforms')

    apply_transforms = Node(ApplyTransforms(),iterfield=['input_image'],name='apply_transforms')
    apply_transforms.inputs.input_image_type = 3
    apply_transforms.inputs.float = False
    apply_transforms.inputs.num_threads = 12
    apply_transforms.inputs.environ = {}
    apply_transforms.inputs.interpolation = 'BSpline'
    apply_transforms.inputs.invert_transform_flags = [False, False]
    apply_transforms.inputs.terminal_output = 'stream'

    apply_transforms.inputs.reference_image = MNItemplate

    #Use FSL for smoothing
    smooth = Node(Smooth(),name='smooth')
    smooth.inputs.sigma = 6.0

    #####################################
    ### DOWNSAMPLE PRECISION ###
    #####################################
    down_samp = Node(Down_Sample_Precision(),name='down_samp')

    ###################################
    ### PLOTS ###
    ###################################

    plot_realign = Node(Plot_Realignment_Parameters(),name="plot_realign")
    plot_qa = Node(Plot_Quality_Control(),name="plot_qa")
    plot_normalization_check = Node(Plot_Coregistration_Montage(),name="plot_normalization_check")
    plot_normalization_check.inputs.canonical_img = MNItemplatehasskull


    ###################################
    ### DATA OUTPUT ###
    ###################################
    #Collect all final outputs in the output dir and get rid of file name additions
    datasink = Node(DataSink(),name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
    datasink.inputs.substitutions = [('_scan_..mnt..Raw..' + subject_id + '..func..',''),
                                    (subject_id+'_acq-p1X',''),
                                    ('X3mmXsl48Xap_bold.nii.gz','')]


    ###################################
    ### HOOK IT ALL CAPTAIN! ###
    ###################################
    workflow = Workflow(name='Preprocessing')
    workflow.base_dir = os.path.join(base_dir,subject_id)

    workflow.connect([
        (merge_discorr, topup, [('merged_file','in_file')]),
        (topup, apply_topup,[('out_fieldcoef','in_topup_fieldcoef'),
                            ('out_movpar','in_topup_movpar')]),
        (func_scans, apply_topup, [('scan','in_files')]),
        (apply_topup, realign_fsl, [('out_corrected','in_file')]),
        (realign_fsl, plot_realign, [('par_file','realignment_parameters')]),
        (realign_fsl, plot_qa, [('out_file','dat_img')]),
        (realign_fsl, art, [('out_file','realigned_files'),
                           ('par_file','realignment_parameters')]),
        (realign_fsl, mean_epi, [('out_file','in_file')]),
        (realign_fsl, make_cov, [('par_file','realignment_parameters')]),
        (mean_epi, compute_mask, [('out_file','mean_volume')]),
        (compute_mask, art, [('brain_mask','mask_file')]),
        (art, make_cov, [('outlier_files','spike_id')]),
        (brain_extraction_ants, coregistration, [('BrainExtractionBrain','fixed_image')]),
        (mean_epi, coregistration, [('out_file','moving_image')]),
        (brain_extraction_ants, normalization, [('BrainExtractionBrain','moving_image')]),
        (coregistration, merge_transforms, [('composite_transform','in2')]),
        (normalization, merge_transforms, [('composite_transform','in1')]),
        (merge_transforms, apply_transforms, [('out','transforms')]),
        (realign_fsl, apply_transforms, [('out_file','input_image')]),
        (apply_transforms, mean_norm_epi, [('output_image','in_file')]),
        (mean_norm_epi, plot_normalization_check, [('out_file','wra_img')]),
        (apply_transforms, smooth, [('output_image','in_file')]),
        (smooth, down_samp, [('smoothed_file','in_file')]),
        (down_samp, datasink, [('out_file','functional.@down_samp')]),
        (plot_realign, datasink, [('plot','functional.@plot_realign')]),
        (plot_qa, datasink, [('plot','functional.@plot_qa')]),
        (plot_normalization_check, datasink, [('plot','functional.@plot_normalization')]),
        (make_cov, datasink, [('covariates','functional.@covariates')]),
        (brain_extraction_ants, datasink, [('BrainExtractionBrain','structural.@struct')]),
        (normalization, datasink, [('warped_image','structural.@normalize')])
        ])

    if not os.path.exists(os.path.join(output_dir,'Preprocsteps.png')):
        workflow.write_graph(dotfilename=os.path.join(output_dir,'Preprocsteps'),format='png')

        return workflow
