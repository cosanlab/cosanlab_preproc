# Cosanlab Preprocessing Tools

Preprocessing tools used in the [Cosanlab](http://cosanlab.com/) built using [nipype](http://nipype.readthedocs.io/en/latest/).  
Many of these tools can be used in conjunction with our [neuroimaging analysis toolbox](https://github.com/ljchang/nltools), and easily run from our [Docker based analysis container](https://github.com/cosanlab/cosanToolsDocker).  

## Installation  

As of 06/18/19 we're still using an older version of [pybids](https://github.com/bids-standard/pybids) to gather data files. This *does not* affect estimation in anyway. Make sure you install by doing:  
1. `pip uninstall pybids` (remove any existing pybids installation) 
2. `pip install pybids==0.6.5` (install the version we need)
3. `pip install git+https://github.com/cosanlab/cosanlab_preproc` (install cosanlab_preproc)

## Usage   

While this package provides interfaces for use within custom nipype workflows, and prebuilt workflows for reproducing specific analysis pipelines, we also have a common pipeline used across numerous studies. It's also easily adaptable to specific use cases. The simplest way to use it is via the **workflow maker** in `cosanlab_preproc.wfmaker` following the examples at the bottom of this page.

### Required directory structure  

Using the workflow maker assumes your data are in BIDS format within a top level directory organized as follows. The preprocessed and log directories will be created for you at the same level as your BIDS data:  

```
project/
    raw_dir_name/
        sub1/
            anat/
            fmap/
            func/
                sub1-task-something-run-01.bold.nii.gz
                sub1-task-something-run-01.bold.json
                sub1-task-something-run-02.bold.nii.gz
                sub1-task-something-run-02.bold.json
                .
                .
                .
        .
        .
        .
    preprocessed/
        final/
            [Final outputted files will be put here, organized by subject]
        intermediate/
            [Intermediate files will be put here, organized by subject]
        pipeline.png (image of entire pipeline)
    logs/
        nipype/
            [Nipype logs will be put here]
```

### Work flow steps  

The common workflow performs the following routines. Optional routines are italicized.  

1) *EPI Distortion Correction (FSL; requires reverse-blipped "fieldmaps")*
2) *Trimming (nipy)*
3) Realignment/Motion Correction (FSL)
4) Artifact Detection (rapidART/python)
5) Brain Extraction + N4 Bias Correction (ANTs)
6) Coregistration (rigid) (ANTs)
7) Normalization to MNI (non-linear) (ANTs)
8) *Low-pass/High-Frequency filtering (nilearn)*
8) *Smoothing (FSL)*
9) Downsampling to INT16 precision to save space (nibabel)

### Generated QA files  

1) EPI data in MNI space
2) T1 data in MNI space
3) T1 tissue segmentation masks in MNI space
4) Realignment parameters, motion, and intensity spikes (single file)
5) QA plots of global signal mean, sd, and intensity frame-differences
6) QA figures for normalization check  
7) QA plots of motion

### Notes

- Assumes TR is the same for all functional runs
- Fine-tuning node settings not provided as input arguments, requires manually editing specific nodes
- Multi-session data will generate a _list_ of workflows that need to be run manually in sequence by the user

### Example workflows  

The simplest workflow performs only realignment and non-linear normalization to the 2mm MNI template. We run the workflow serially, one step at a time.

```
from cosanlab_preproc.wfmaker import wfmaker

workflow = wfmaker(
                project_dir = '/data/project',
                raw_dir = 'raw',
                subject_id = 's01')

workflow.run()
```

A more common workflow involves adding a few steps to the above workflow, namely: trimming the first few TRs (in this case 5) to ignore pre-steady-state volumes, and smoothing to 6mm after normalization. We run the workflow in parallel using 16-cores.

```
from cosanlab_preproc.wfmaker import wfmaker

workflow = wfmaker(
                project_dir = '/data/project',
                raw_dir = 'raw',
                apply_trim = 5,
                apply_smooth = 6.0)

# Run it with 16 parallel cores
workflow.run('MultiProc',plugin_args = {'n_procs': 16})
```

Create a more sophisticated workflow that additionally adds: EPI distortion correction (assumes reverse-blip "fmap" scans have been acquired) to each functional run, outputs that include both non-filtered and filtered data (.25hz), smoothed outputs at both 6mm and 8mm. Here we also trim the first 25 TRs of each functional run, and normalize to *3mm MNI space* instead. Parallelize running the workflow as well.

```
from cosanlab_preproc.wfmaker import wfmaker

workflow = wfmaker(
                project_dir = '/data/project',
                raw_dir = 'raw',
                apply_trim = 25,
                apply_dist_corr = True,
                apply_filter = [0, .25],
                apply_smooth = [6.0, 8.0],
                mni = '3mm')

workflow.run('MultiProc',plugin_args = {'n_procs': 16})
```

#### Getting help  

In general you can view the help for the workflow builder by doing the following in an interactive python session or looking [here](https://github.com/cosanlab/cosanlab_preproc/blob/master/cosanlab_preproc/wfmaker.py#L33):  

```
from cosanlab_preproc.wfmaker import wfmaker

wfmaker?
```
