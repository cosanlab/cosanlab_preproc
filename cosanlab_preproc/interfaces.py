from __future__ import division

'''
Preproc Nipype Interfaces
=========================

Classes for various nipype interfaces

'''

__all__ = ['Plot_Coregistration_Montage', 'Plot_Realignment_Parameters',
           'Create_Covariates', 'Down_Sample_Precision', 'Low_Pass_Filter', 'CreateEncodingFile']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import nibabel as nib
from nipype.interfaces.base import isdefined, BaseInterface, TraitedSpec, File, traits
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
    title = traits.Str("Quality Control Plot", usedefault=True)
    global_outlier_cutoff = traits.Float(3, usedefault=True)
    frame_outlier_cutoff = traits.Float(3, usedefault=True)


class Plot_Quality_Control_OutputSpec(TraitedSpec):
    plot = File(exists=True)
    fd_outliers = File(exists=True)


class Plot_Quality_Control(BaseInterface):
    # This function creates quality control plots for a 4D time series.
    # Recommend running this after realignment.

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
        mask = compute_epi_mask(dat_img)
        masked_data = apply_mask(dat_img, mask)
        mn = np.mean(masked_data, axis=0)
        sd = np.std(masked_data, axis=0)
        tsnr = np.true_divide(mn, sd)
        mn = unmask(mn, mask)
        sd = unmask(sd, mask)
        tsnr = unmask(tsnr, mask)
        global_mn = np.mean(masked_data, axis=1)
        global_sd = np.std(masked_data, axis=1)

        global_outlier = np.append(np.where(global_mn > np.mean(global_mn) + np.std(global_mn) * self.inputs.global_outlier_cutoff),
                                   np.where(global_mn < np.mean(global_mn) - np.std(global_mn) * self.inputs.global_outlier_cutoff))
        frame_diff = np.mean(np.abs(np.diff(masked_data, axis=0)), axis=1)
        frame_outlier = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * self.inputs.frame_outlier_cutoff),
                                  np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * self.inputs.frame_outlier_cutoff))

        fd_file_name = "fd_outlier.txt"
        np.savetxt(fd_file_name, frame_outlier)

        title = self.inputs.title

        if title != "":
            filename = title.replace(" ", "_") + ".pdf"
        else:
            filename = "Quality_Control_Plot.pdf"

        f, ax = plt.subplots(6, figsize=(15, 15))
        plot_stat_map(mn, title="Mean", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax[0],
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)

        plot_stat_map(sd, title="Standard Deviation", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax[1],
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)

        plot_stat_map(tsnr, title="SNR (mn/sd)", cut_coords=range(-40, 40, 10), display_mode='z', axes=ax[2],
                      draw_cross=False, black_bg=True, annotate=False, bg_img=None)

        ax[3].plot(global_mn)
        # ax[3].set_title('Average Signal Intensity')
        ax[3].set_xlabel('TR')
        ax[3].set_ylabel('Global Signal Mean')
        for x in global_outlier:
            ax[3].axvline(x, color='r', linestyle='--')

        ax[4].plot(global_sd)
        # ax[4].set_title('Frame Differencing')
        ax[4].set_xlabel('TR')
        ax[4].set_ylabel('Global Signal Std')

        ax[5].plot(frame_diff)
        # ax[4].set_title('Frame Differencing')
        ax[5].set_xlabel('TR')
        ax[5].set_ylabel('Avg Abs Diff')
        for x in frame_outlier:
            ax[5].axvline(x, color='r', linestyle='--')
        f.savefig(filename)
        plt.close(f)
        del f

        self._plot = filename
        self._fd_outliers = fd_file_name

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        outputs["fd_outliers"] = os.path.abspath(self._fd_outliers)
        return outputs


class Plot_Realignment_Parameters_InputSpec(TraitedSpec):
    realignment_parameters = File(exists=True, mandatory=True)
    outlier_files = File(exists=True)
    title = traits.Str("Realignment parameters", usedefault=True)
    dpi = traits.Int(300, usedefault=True)


class Plot_Realignment_Parameters_OutputSpec(TraitedSpec):
    plot = File(exists=True)


class Plot_Realignment_Parameters(BaseInterface):
    # This function is adapted from Chris Gorgolewski and creates a figure of the realignment parameters

    input_spec = Plot_Realignment_Parameters_InputSpec
    output_spec = Plot_Realignment_Parameters_OutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        realignment_parameters = np.loadtxt(self.inputs.realignment_parameters)
        title = self.inputs.title

        F = plt.figure(figsize=(8.3, 11.7))
        F.text(0.5, 0.96, self.inputs.title, horizontalalignment='center')
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        handles = ax1.plot(realignment_parameters[:, 0:3])
        ax1.legend(handles, ["x translation",
                             "y translation", "z translation"], loc=0)
        ax1.set_xlabel("image #")
        ax1.set_ylabel("mm")
        ax1.set_xlim((0, realignment_parameters.shape[0] - 1))
        ax1.set_ylim(bottom=realignment_parameters[:, 0:3].min(
        ), top=realignment_parameters[:, 0:3].max())

        ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        handles = ax2.plot(realignment_parameters[:, 3:6] * 180.0 / np.pi)
        ax2.legend(handles, ["pitch", "roll", "yaw"], loc=0)
        ax2.set_xlabel("image #")
        ax2.set_ylabel("degrees")
        ax2.set_xlim((0, realignment_parameters.shape[0] - 1))
        ax2.set_ylim(bottom=(realignment_parameters[:, 3:6] * 180.0 / np.pi).min(
        ), top=(realignment_parameters[:, 3:6] * 180.0 / np.pi).max())

        if isdefined(self.inputs.outlier_files):
            try:
                outliers = np.loadtxt(self.inputs.outlier_files)
            except IOError as e:
                if e.args[0] == "End-of-file reached before encountering data.":
                    pass
                else:
                    raise
            else:
                if outliers.size > 0:
                    ax1.vlines(outliers, ax1.get_ylim()[0], ax1.get_ylim()[1])
                    ax2.vlines(outliers, ax2.get_ylim()[0], ax2.get_ylim()[1])

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


class Low_Pass_Filter_InputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True)
    mask = File(exists=True, mandatory=True)
    low_pass_cutoff = traits.Float(0.25, usedefault=True)
    sampling_rate = traits.Float(.419, usedefault=True)


class Low_Pass_Filter_OutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Low_Pass_Filter(BaseInterface):
    input_spec = Low_Pass_Filter_InputSpec
    output_spec = Low_Pass_Filter_OutputSpec

    def _run_interface(self, runtime):
        from nltools.data import Brain_Data
        import os
        in_file = self.inputs.in_file
        mask = self.inputs.mask
        low_pass = self.inputs.low_pass_cutoff
        TR = self.inputs.sampling_rate

        dat = Brain_Data(in_file, mask=mask)
        dat = dat.filter(sampling_rate=TR, low_pass=low_pass)

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

class CreateEncodingFileInputSpec(TraitedSpec):
    fmaps = traits.List()
    fmap_pes = traits.List()
    totalReadoutTimes = traits.List()
    measurements = traits.List()
    file_name = traits.Str()

class CreateEncodingFileOutputSpec(TraitedSpec):
    encoding_file = traits.File()

class CreateEncodingFile(BaseInterface):
    """
    CreateEncodingFile interface creates encoding file necessary for FSL TOPUP interface.
    Args:
        fmaps: list of fieldmap files (e.g., [AP.nii.gz, PA.nii.gz] )
        fmap_pes: list of phase encoding directions for each file (e.g., [j-, j])
        totalReadoutTimes = list of totalReadoutTimes for each file (e.g., [.0423, .0423])
        measurements: list of number of measurements for each file (e.g., [2, 2])
        file_name: string for file name of encoding file (e.g., encoding_file.txt)
    Returns:
        encoding_file: encoding file to be used with topup
    """
    input_spec = CreateEncodingFileInputSpec
    output_spec = CreateEncodingFileOutputSpec

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
