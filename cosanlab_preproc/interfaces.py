from __future__ import division

'''
Preproc Nipype Interfaces
=========================

Classes for various nipype interfaces

'''

__all__ = ['Plot_Coregistration_Montage', 'Plot_Realignment_Parameters',
           'Create_Covariates', 'Down_Sample_Precision', 'Filter_In_Mask', 'Create_Encoding_File']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import nibabel as nib
from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits
from nilearn import plotting, image


class Plot_Coregistration_Montage_InputSpec(TraitedSpec):
    wra_img = File(exists=True, mandatory=True)
    canonical_img = File(exists=True, mandatory=True)
    title = traits.Str("Normalized Functional Check", usedefault=True)


class Plot_Coregistration_Montage_OutputSpec(TraitedSpec):
    plot = File(exists=True)


class Plot_Coregistration_Montage(BaseInterface):
    # This function creates an axial montage of the average normalized functional data
    # and overlays outline of the normalized single subject overlay.
    # Could probably pick a better overlay later.

    input_spec = Plot_Coregistration_Montage_InputSpec
    output_spec = Plot_Coregistration_Montage_OutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt

        wra_img = nib.load(self.inputs.wra_img)
        canonical_img = nib.load(self.inputs.canonical_img)
        title = self.inputs.title
        mean_wraimg = image.mean_img(wra_img)

        if title != "":
            filename = title.replace(" ", "_") + ".pdf"
        else:
            filename = "plot.pdf"

        f, axes = plt.subplots(6, figsize=(15, 20))
        titles = ["sag: wrafunc & canonical single subject", "sag: wrafunc & canonical single subject",
                  "axial: wrafunc & canonical single subject", "axial: wrafunc & canonical single subject",
                  "coronal: wrafunc & canonical single subject", "coronal: wrafunc & canonical single subject"
                  ]
        cut_coords = [range(-50, 0, 10), range(0, 51, 10), range(-30, 15, 9),
                      range(0, 61, 10), range(-60, 0, 12), range(0, 31, 6)]
        display_modes = ['x', 'x', 'z', 'z', 'y', 'y']
        for i, ax in enumerate(axes):
            fig = plotting.plot_anat(
                mean_wraimg, title=titles[i], cut_coords=cut_coords[i], display_mode=display_modes[i], axes=ax)
            fig.add_edges(canonical_img)

        f.savefig(filename)
        plt.close(f)
        plt.close()
        del f
        self._plot = filename
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        return outputs


class Plot_Quality_Control_InputSpec(TraitedSpec):
    dat_img = File(exists=True, mandatory=True)
    title = traits.Str("Signal quality", usedefault=True)
    global_outlier_cutoff = traits.Float(3, usedefault=True)
    frame_outlier_cutoff = traits.Float(3, usedefault=True)
    dpi = traits.Int(300, usedefault=True)


class Plot_Quality_Control_OutputSpec(TraitedSpec):
    plot = File(exists=True)
    fd_outliers = File(exists=True)
    global_outliers = File(exists=True)


class Plot_Quality_Control(BaseInterface):

    """
    This is a QA interface that does two things:
    1) It generates plots that include: tSNR brain images, global signal mean, std, and frame-differences (of signal intensities)
    2) It computes two kinds of outliers: a) TRs where the global signal > 3 stds (default) from the mean; b) TRs where successive differences between TRs (i.e. frame differences) are > 3 stds (default) from the mean frame-diff

    Args:
        dat_img: epi nifti file
        title: plot title (optional)
        global_outlier_cutoff: cutoff to identify outlier TRs based on global signal intensity; default 3 standard deviations from mean
        frame_outlier_cutoff: cutoff to identifiy outlier TRs based on intensity differences between successive TRs; default 3 standard deviations from mean
        dpi: figure dpi; default 300

    Returns:
        plot: QA plot file
        fd_outliers: outlier TRs based on frame-differences
        global_outliers: outlier TRs based on global intensity

    """

    input_spec = Plot_Quality_Control_InputSpec
    output_spec = Plot_Quality_Control_OutputSpec

    def _run_interface(self, runtime):
        # from __future__ import division
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        import numpy as np
        import nibabel as nib
        from nilearn.masking import compute_epi_mask, apply_mask, unmask
        from nilearn.plotting import plot_stat_map

        dat_img = nib.load(self.inputs.dat_img)
        # Apply mask first to deal with 0 sd for computing tsnr
        # nilearn defaults are lower = 0.2; upper = 0.85
        mask = compute_epi_mask(dat_img)
        masked_data = apply_mask(dat_img, mask)
        # Compute mean across time within each voxel
        mn = np.mean(masked_data, axis=0)
        sd = np.std(masked_data, axis=0)
        tsnr = np.true_divide(mn, sd)
        # Compute mean across voxels within each TR
        global_mn = np.mean(masked_data, axis=1)
        global_sd = np.std(masked_data, axis=1)
        # Unmask data for plotting below
        mn = unmask(mn, mask)
        sd = unmask(sd, mask)
        tsnr = unmask(tsnr, mask)

        # Identify global signal outliers
        global_outliers = np.append(np.where(global_mn > np.mean(global_mn) + np.std(global_mn) * self.inputs.global_outlier_cutoff),
                                   np.where(global_mn < np.mean(global_mn) - np.std(global_mn) * self.inputs.global_outlier_cutoff))

        # Identify frame difference outliers
        frame_diff = np.mean(np.abs(np.diff(masked_data, axis=0)), axis=1)
        frame_outliers = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * self.inputs.frame_outlier_cutoff),
                                  np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * self.inputs.frame_outlier_cutoff))

        fd_file_name = "fd_outliers.txt"
        global_file_name = "global_outliers.txt"
        np.savetxt(fd_file_name, frame_outliers)
        np.savetxt(global_file_name, global_outliers)

        title = self.inputs.title
        colspan = 2
        loc = 9
        F = plt.figure(figsize=(8.3, 11.7))
        F.text(0.5, .93, title, horizontalalignment='center',fontsize=16)
        F.text(0.5, .01, 'TR', horizontalalignment='center',fontsize=16)

        # Plot brain images first
        ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=colspan)
        plot_stat_map(mn, title="Mean", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax1,
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)
        ax2 = plt.subplot2grid((6, 2), (1, 0), colspan=colspan)
        plot_stat_map(sd, title="Standard Deviation", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax2,
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)
        ax3 = plt.subplot2grid((6, 2), (2, 0), colspan=colspan)
        plot_stat_map(tsnr, title="tSNR (mn/sd)", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax3,
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)

        # Plot global mean, std, diffs next
        ax4 = plt.subplot2grid((6, 2), (3, 0), colspan=colspan)
        handles = ax4.plot(global_mn)
        ax4.set(xlabel='',ylabel='Global mean',xticklabels=[])
        v_ax = ax4.vlines(global_outliers,ax4.get_ylim()[0],ax4.get_ylim()[-1],color='r', linestyle='--',zorder=3)
        handles.append(v_ax)
        ax4.legend([v_ax],['intensity outliers'],loc=loc)
        ax4.tick_params(direction='in')

        ax5 = plt.subplot2grid((6, 2), (4, 0), colspan=colspan)
        handles = ax5.plot(global_sd)
        ax5.set(xlabel='',ylabel='Global std',xticklabels=[])
        ax5.tick_params(direction='in')

        ax6 = plt.subplot2grid((6, 2), (5, 0), colspan=colspan)
        handles = ax6.plot(frame_diff)
        ax6.set(xlabel='',ylabel='Global abs diffs')
        v_ax = ax6.vlines(frame_outliers,ax6.get_ylim()[0],ax6.get_ylim()[-1],color='r', linestyle='--',zorder=3)
        ax6.legend([v_ax],['diff outliers'],loc=loc)
        ax6.tick_params(direction='in')

        if title != "":
            filename = title.replace(" ", "_") + ".pdf"
        else:
            filename = "plot.pdf"

        F.savefig(filename, papertype="a4", dpi=self.inputs.dpi)
        plt.clf()
        plt.close()
        del F

        self._plot = filename
        self._fd_outliers = fd_file_name
        self._global_outliers = global_file_name

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        outputs["fd_outliers"] = os.path.abspath(self._fd_outliers)
        outputs["global_outliers"] = os.path.abspath(self._global_outliers)
        return outputs


class Plot_Realignment_Parameters_InputSpec(TraitedSpec):
    realignment_parameters = File(exists=True, mandatory=True)
    outliers = File(exists=True)
    title = traits.Str("Realignment parameters", usedefault=True)
    dpi = traits.Int(300, usedefault=True)


class Plot_Realignment_Parameters_OutputSpec(TraitedSpec):
    plot = File(exists=True)


class Plot_Realignment_Parameters(BaseInterface):

    """
    Create a plot of realignment parameters.
    *NOTE*, this function assumes the output of FSL's McFlirt which returns Rotation X, Rotation Y, Rotation Z (all in rads), Translation X, Translation Y, Translation Z (all in mm).
    For reference of this FSL mailing list:
    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;cda6e2ea.1112
    And nipype.utils.normalize_mc_params:
    https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;cda6e2ea.1112

    To make plots on the same scale we convert rotations to arc-length in mm using assuming a 50mm sphere which is approximately the mean distance from the cerebral cortex to the cetner of the head (Power et al, 2012, NeuroImage). This conversion is not applied to any covariate files generated and is purely for visualization.

    Formuala: arc-length (mm) = rotation (rad) * 50mm

    Args:
        realignment_parameters: FSL Mcflirt's output parameter file
        outliers: text file with outlier time-points, e.g. from rapidart
        title: plot title; default 'Realignment parameters'
        dpi: figure dpi; default 300

    Returns:
        plot: plot file

    """

    input_spec = Plot_Realignment_Parameters_InputSpec
    output_spec = Plot_Realignment_Parameters_OutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        realignment_parameters = np.loadtxt(self.inputs.realignment_parameters)
        realignment_parameter_diffs = np.abs(np.diff(realignment_parameters,axis=0))
        outliers = np.loadtxt(self.inputs.outliers)

        title = self.inputs.title
        colspan = 2
        loc = 9
        realign_lims = (-2,2)
        diff_lims = (-0.01,2)
        F = plt.figure(figsize=(8.3, 11.7))
        F.text(0.5, .97, title, horizontalalignment='center',fontsize=16)
        F.text(0.5, .01, 'TR', horizontalalignment='center',fontsize=16)

        # Plot x,y,z first
        ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=colspan)
        handles = ax1.plot(realignment_parameters[:, 3:6])
        ax1.legend(handles, ["x","y", "z"], loc=loc, ncol=3)
        ax1.set(ylabel="Translation (mm)",xticklabels=[],xlim=(0,realignment_parameters.shape[0]-1),ylim=(realign_lims))
        ax1.tick_params(direction = 'in')

        # Plot pitch, roll, yaw second
        ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=colspan)
        handles = ax2.plot(realignment_parameters[:, 0:3] * 50)
        ax2.legend(handles, ["pitch", "roll", "yaw"], loc=loc, ncol=3)
        ax2.set(ylabel="Rotation (mm)",xticklabels=[],xlim=(0,realignment_parameters.shape[0]-1),ylim=realign_lims)
        ax2.tick_params(direction = 'in')

        # Plot x,y,z diffs with rapidart third
        ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=colspan)
        handles = ax3.plot(realignment_parameter_diffs[:, 3:6])
        v_ax = ax3.vlines(outliers,ymin=realign_lims[0],ymax=realign_lims[1],color='r',linestyle='--')
        handles.append(v_ax)
        ax3.legend(handles, ["x","y","z","rapid art"], loc=loc, ncol=4)
        ax3.set(ylabel="Translation diffs (abs mm)",xticklabels=[],xlim=(0,realignment_parameter_diffs.shape[0]-1),ylim=(diff_lims))
        ax3.tick_params(direction = 'in')

        # Plot pitch,roll,raw diffs with rapidart fourth
        ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=colspan)
        handles = ax4.plot(realignment_parameter_diffs[:, 0:3]*50)
        v_ax = ax4.vlines(outliers,ymin=realign_lims[0],ymax=realign_lims[1],color='r',linestyle='--')
        handles.append(v_ax)
        ax4.legend(handles, ["pitch","roll", "yaw", "rapid art"], loc=loc, ncol=4)
        ax4.set(ylabel="Rotation diffs (abs mm)",xlim=(0,realignment_parameter_diffs.shape[0]-1),ylim=diff_lims)
        ax4.tick_params(direction = 'in')

        # Fine-tune spacing
        plt.subplots_adjust(top=.96,bottom=.05,hspace=.1)

        if title != "":
            filename = title.replace(" ", "_") + ".pdf"
        else:
            filename = "plot.pdf"

        F.savefig(filename, papertype="a4", dpi=self.inputs.dpi)
        plt.clf()
        plt.close()
        del F

        self._plot = filename

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        return outputs


class Down_Sample_Precision_InputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True)
    data_type = traits.Str("int16", usedefault=True)


class Down_Sample_Precision_OutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Down_Sample_Precision(BaseInterface):
    input_spec = Down_Sample_Precision_InputSpec
    output_spec = Down_Sample_Precision_OutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import os
        data_type = self.inputs.data_type
        in_file = self.inputs.in_file

        dat = nib.load(in_file)
        out = nib.Nifti1Image(dat.get_data().astype(data_type), dat.affine)

        # Generate output file name
        out_file = os.path.split(
            in_file)[-1].split('.nii.gz')[0] + '_' + data_type + '.nii.gz'
        out.to_filename(out_file)

        self._out_file = out_file

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self._out_file)
        return outputs


class Filter_In_Mask_InputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True)
    mask = File(exists=True, mandatory=True)
    low_pass_cutoff = traits.Float(0.25, usedefault=True)
    high_pass_cutoff = traits.Float(0, usedefault=True)
    sampling_rate = traits.Float(mandatory=True)


class Filter_In_Mask_OutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Filter_In_Mask(BaseInterface):
    """
    Node to perform high and/or low-pass filtering using nltools which utilizes nilearn's 5th order butterworth filter. If no low or high-pass cutoffs are provided, simply masks the data and returns as-is. This can be useful if the output is subsequently passed to a smoothing node, to act like AFNI's blur in mask functionality.

    Args:
        in_file: file to filter
        mask: mask to apply to data to filer; typically something like MNI152 mask
        low_pass_cutoff: frequencies above this will be filtered; default 0.25hz
        high_pass_cutoff: frequenceies below this will be filtered; default None
        sampling_rate: TR in seconds

    Returns:
        out_file: filtered and masked data
    """

    input_spec = Filter_In_Mask_InputSpec
    output_spec = Filter_In_Mask_OutputSpec

    def _run_interface(self, runtime):
        from nltools.data import Brain_Data
        import os
        in_file = self.inputs.in_file
        mask = self.inputs.mask
        low_pass = self.inputs.low_pass_cutoff
        high_pass = self.inputs.high_pass_cutoff
        TR = self.inputs.sampling_rate

        if low_pass == 0:
            low_pass = None
        if high_pass == 0:
            high_pass = None

        dat = Brain_Data(in_file, mask=mask)
        # Handle no filtering
        if low_pass or high_pass:
            dat = dat.filter(sampling_rate=TR, low_pass=low_pass,high_pass=high_pass)

        # Generate output file name
        out_file = os.path.split(
            in_file)[-1].split('.nii.gz')[0] + '_filtered.nii.gz'
        dat.write(out_file)

        self._out_file = out_file

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self._out_file)
        return outputs


class Create_Covariates_InputSpec(TraitedSpec):
    realignment_parameters = File(exists=True, mandatory=True)
    spike_id = File(exists=True, mandatory=True)
    fd_outliers = File(exists=True)


class Create_Covariates_OutputSpec(TraitedSpec):
    covariates = File(exists=True)


class Create_Covariates(BaseInterface):
    input_spec = Create_Covariates_InputSpec
    output_spec = Create_Covariates_OutputSpec

    def _run_interface(self, runtime):
        ra = pd.read_table(self.inputs.realignment_parameters, header=None,
                           sep=r"\s*", names=['ra' + str(x) for x in range(1, 7)])
        spike = pd.read_table(self.inputs.spike_id,
                              header=None, names=['Spikes'])
        fd = pd.read_table(self.inputs.fd_outliers, header=None, names=['FDs'])

        ra = ra - ra.mean()  # mean center
        ra[['rasq' + str(x) for x in range(1, 7)]] = ra**2  # add squared
        ra[['radiff' + str(x) for x in range(1, 7)]
           ] = pd.DataFrame(ra[ra.columns[0:6]].diff())  # derivative
        ra[['radiffsq' + str(x) for x in range(1, 7)]] = pd.DataFrame(
            ra[ra.columns[0:6]].diff())**2  # derivatives squared

        # build spike regressors
        for i, loc in enumerate(spike['Spikes']):
            ra['spike' + str(i + 1)] = 0
            ra['spike' + str(i + 1)].iloc[int(loc)] = 1

        # build FD regressors
        for i, loc in enumerate(fd['FDs']):
            ra['FD' + str(i + 1)] = 0
            ra['FD' + str(i + 1)].iloc[int(loc)] = 1

        filename = 'covariates.csv'
        ra.to_csv(filename, index=False)  # write out to file
        self._covariates = filename

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["covariates"] = os.path.abspath(self._covariates)
        return outputs

class Create_Encoding_File_InputSpec(TraitedSpec):
    fmaps = traits.List()
    fmap_pes = traits.List()
    totalReadoutTimes = traits.List()
    measurements = traits.List()
    file_name = traits.Str()

class Create_Encoding_File_OutputSpec(TraitedSpec):
    encoding_file = traits.File()

class Create_Encoding_File(BaseInterface):
    """
    Create_Encoding_File interface creates encoding file necessary for FSL TOPUP interface.
    Args:
        fmaps: list of fieldmap files (e.g., [AP.nii.gz, PA.nii.gz] )
        fmap_pes: list of phase encoding directions for each file (e.g., [j-, j])
        totalReadoutTimes = list of totalReadoutTimes for each file (e.g., [.0423, .0423])
        measurements: list of number of measurements for each file (e.g., [2, 2])
        file_name: string for file name of encoding file (e.g., encoding_file.txt)
    Returns:
        encoding_file: encoding file to be used with topup
    """
    input_spec = Create_Encoding_File_InputSpec
    output_spec = Create_Encoding_File_OutputSpec

    def _run_interface(self,runtime):
        pe_to_encoding = {'i':'1 0 0','i-':'-1 0 0',
         'j':'0 1 0','j-':'0 -1 0',
         'k':'0 0 1','k-':'0 0 -1'}
        file_name = os.path.join(os.path.split(self.inputs.fmaps[0])[0], self.inputs.file_name)
        if os.path.isfile(file_name):
            with open(file_name, 'w') as fp:
                fp.write('')
        # Create encoding file and save to file_name
        for ix, fmap in enumerate(self.inputs.fmaps):
            with open(file_name, 'a') as fp:
                message = pe_to_encoding[self.inputs.fmap_pes[ix]]+' '+str(self.inputs.totalReadoutTimes[ix]) + '\n'
                fp.write(message*self.inputs.measurements[ix])
                print('wrote to file',file_name)
        self._encoding_file = file_name
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["encoding_file"] =os.path.abspath(self._encoding_file)
        return outputs
