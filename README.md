# Cosanlab Preprocessing Tools

Preprocessing tools used in the [Cosanlab](http://cosanlab.com/) built using [nipype](http://nipype.readthedocs.io/en/latest/).  
Many of these tools can be used in conjunction with our [neuroimaging analysis toolbox](https://github.com/ljchang/nltools), and easily run from our [Docker based analysis container](https://github.com/cosanlab/cosanToolsDocker).  

## Common study workflow  

While this package provides interfaces for use within custom nipype workflows, and prebuilt workflows for reproducing specific analysis pipelines, we also have a common pipeline used across numerous studies. It's also easily adaptable to specific use cases. The simplest way to use it is via the **workflow maker**, following the examples below.  Using this function assumes a BIDS formatted directory organized as follows. The preprocessed and log directories will be created for you:  

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

#### Work flow steps  

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

#### Generated QA files  

1) EPI data in MNI space
2) T1 data in MNI space
3) T1 tissue segmentation masks in MNI space
4) Realignment parameters, motion, and intensity spikes (single file)
5) QA plots of global signal mean, sd, and intensity frame-differences
6) QA figures for normalization check  
7) QA plots of motion

#### Example Usage  

Create a simple workflow w/o distortion, w/o filtering, w/ 6mm smoothing, normalizing to the 2mm MNI template, and trimming the first 5 TRs of the each functional run.

```
from cosanlab_preproc.wfmaker import wfmaker

workflow = wfmaker(
                project_dir = '/data/project',
                raw_dir = 'raw',
                vols_to_trim = 5)

# Run it with 16 parallel cores
workflow.run('MultiProc',plugin_args = {'n_procs': 16})

```

Create a more complicated workflow that performs distortion correction on each functional run, creates filtered (at .25hz) and non-filtered output, creates 6mm and 8mm smoothed output, trims the first 25 TRs of each functional run, and normalizes to 3mm MNI space.

```
from cosanlab_preproc.wfmaker import wfmaker

workflow = wfmaker(
                project_dir = '/data/project',
                raw_dir = 'raw',
                vols_to_trim = 25,
                dist_corr = True,
                apply_filter = [0, .25],
                apply_smooth = [6.0, 8.0],
                mni = '3mm')

# Run it serially...super slow
workflow.run()

```
