from __future__ import division

'''
    Preproc Nipype Interfaces
    =========================
    
    Classes for various nipype interfaces

'''

__all__ = ['Plot_Coregistration_Montage', 'Plot_Realignment_Parameters', 'Create_Covariates', 'Build_Xmat']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import pylab as plt
import os
import nibabel as nib
from nipype.interfaces.base import isdefined, BaseInterface, TraitedSpec, File, traits
from nilearn import plotting, image
import nibabel as nib


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
			filename = title.replace(" ", "_")+".pdf"
		else:
			filename = "plot.pdf"

		# fig = plotting.plot_anat(mean_wraimg, title="wrafunc & canonical single subject", cut_coords=range(-40, 40, 10), display_mode='z')
		# fig.add_edges(canonical_img)     
		# fig.savefig(filename)
		# fig.close()

		#JC: Added Saggital slice plotting
		f, (ax1, ax2) = plt.subplots(2,figsize=(15,8))
		fig = plotting.plot_anat(mean_wraimg, title="sag: wrafunc & canonical single subject", cut_coords=range(-30, 20, 8), display_mode='x',axes=ax1)
		fig.add_edges(canonical_img)     
		fig = plotting.plot_anat(mean_wraimg, title="axial: wrafunc & canonical single subject", cut_coords=range(-40, 40, 10), display_mode='z',axes=ax2)
		fig.add_edges(canonical_img)  
		f.savefig(filename)
		plt.close(f) #f.close()
		plt.close()
		del f
		self._plot = filename

		runtime.returncode=0
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
		#Apply mask first to deal with 0 sd for computing tsnr
		mask = compute_epi_mask(dat_img)
		masked_data = apply_mask(dat_img, mask)
		global_mn = np.mean(masked_data,axis=1)
		global_sd = np.std(masked_data,axis=1)
		global_tsnr = np.divide(global_mn,global_sd)
		mn = unmask(global_mn, mask)
		sd = unmask(global_sd, mask)
		snr = unmask(global_tsnr, mask)


		global_outlier = np.append(np.where(global_mn>np.mean(global_mn)+np.std(global_mn)*self.inputs.global_outlier_cutoff),
		                           np.where(global_mn<np.mean(global_mn)-np.std(global_mn)*self.inputs.global_outlier_cutoff))
		frame_diff = np.mean(np.abs(np.diff(masked_data,axis=0)),axis=1)
		frame_outlier = np.append(np.where(frame_diff>np.mean(frame_diff)+np.std(frame_diff)*self.inputs.frame_outlier_cutoff),
		                           np.where(frame_diff<np.mean(frame_diff)-np.std(frame_diff)*self.inputs.frame_outlier_cutoff))


		#mn = nib.Nifti1Image(np.mean(dat_img.get_data(), axis=3), affine=dat_img.get_affine())
		#sd = nib.Nifti1Image(np.std(dat_img.get_data(), axis=3), affine=dat_img.get_affine())
		#snr = nib.Nifti1Image(mn.get_data()/sd.get_data(),affine=dat_img.get_affine())	
	
		title = self.inputs.title

		if title != "":
			filename = title.replace(" ", "_")+".pdf"
		else:
			filename = "Quality_Control_Plot.pdf"

		f, ax = plt.subplots(6,figsize=(15,15))
		plot_stat_map(mn, title="Mean",cut_coords=range(-40, 40, 10), display_mode='z',axes=ax[0],
		              draw_cross=False, black_bg=True,annotate=False,bg_img=None)

		plot_stat_map(sd, title="Standard Deviation",cut_coords=range(-40, 40, 10), display_mode='z',axes=ax[1],
		              draw_cross=False, black_bg=True,annotate=False,bg_img=None)

		plot_stat_map(snr, title="SNR (mn/sd)",cut_coords=range(-40, 40, 10), display_mode='z',axes=ax[2],
		              draw_cross=False, black_bg=True,annotate=False,bg_img=None)

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

		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["plot"] = os.path.abspath(self._plot)
		return outputs

class Plot_Realignment_Parameters_InputSpec(TraitedSpec):
	realignment_parameters = File(exists=True, mandatory=True)
	outlier_files = File(exists=True)
	title = traits.Str("Realignment parameters", usedefault=True)
	dpi = traits.Int(300, usedefault = True)
    
class Plot_Realignment_Parameters_OutputSpec(TraitedSpec):
	plot = File(exists=True)

class Plot_Realignment_Parameters(BaseInterface):
	#This function is adapted from Chris Gorgolewski and creates a figure of the realignment parameters

	input_spec = Plot_Realignment_Parameters_InputSpec
	output_spec = Plot_Realignment_Parameters_OutputSpec
    
	def _run_interface(self, runtime):
		import matplotlib
		matplotlib.use('Agg')
		import pylab as plt
		realignment_parameters = np.loadtxt(self.inputs.realignment_parameters)
		title = self.inputs.title
        
		F = plt.figure(figsize=(8.3,11.7))	
		F.text(0.5, 0.96, self.inputs.title, horizontalalignment='center')
		ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
		handles =ax1.plot(realignment_parameters[:,0:3])
		ax1.legend(handles, ["x translation", "y translation", "z translation"], loc=0)
		ax1.set_xlabel("image #")
		ax1.set_ylabel("mm")
		ax1.set_xlim((0,realignment_parameters.shape[0]-1))
		ax1.set_ylim(bottom = realignment_parameters[:,0:3].min(), top = realignment_parameters[:,0:3].max())
        
		ax2 = plt.subplot2grid((2,2),(1,0), colspan=2)
		handles= ax2.plot(realignment_parameters[:,3:6]*180.0/np.pi)
		ax2.legend(handles, ["pitch", "roll", "yaw"], loc=0)
		ax2.set_xlabel("image #")
		ax2.set_ylabel("degrees")
		ax2.set_xlim((0,realignment_parameters.shape[0]-1))
		ax2.set_ylim(bottom=(realignment_parameters[:,3:6]*180.0/np.pi).min(), top= (realignment_parameters[:,3:6]*180.0/np.pi).max())
        
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
			filename = title.replace(" ", "_")+".pdf"
		else:
			filename = "plot.pdf"

		F.savefig(filename, papertype="a4",dpi=self.inputs.dpi)
		plt.clf()
		plt.close()
		del F

		self._plot = filename
        
		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["plot"] = os.path.abspath(self._plot)
		return outputs

class Create_Covariates_InputSpec(TraitedSpec):	
	realignment_parameters = File(exists=True, mandatory=True) 
	spike_id = File(exists=True, mandatory=True)

class Create_Covariates_OutputSpec(TraitedSpec):
	covariates = File(exists=True)

class Create_Covariates(BaseInterface):
	input_spec = Create_Covariates_InputSpec
	output_spec = Create_Covariates_OutputSpec

	def _run_interface(self, runtime):
		ra = pd.read_table(self.inputs.realignment_parameters, header=None, sep=r"\s*", names=['ra' + str(x) for x in range(1,7)])
		spike = pd.read_table(self.inputs.spike_id, header=None,names=['Spikes'])

		ra = ra-ra.mean() #mean center
		ra[['rasq' + str(x) for x in range(1,7)]] = ra**2 #add squared
		ra[['radiff' + str(x) for x in range(1,7)]] = pd.DataFrame(ra[ra.columns[0:6]].diff()) #derivative
		ra[['radiffsq' + str(x) for x in range(1,7)]] = pd.DataFrame(ra[ra.columns[0:6]].diff())**2 #derivatives squared

		#build spike regressors
		for i,loc in enumerate(spike['Spikes']):
			ra['spike' + str(i+1)] = 0
			ra['spike' + str(i+1)].iloc[loc] = 1

		filename = 'covariates.csv'
		ra.to_csv(filename, index=False) #write out to file
		self._covariates = filename

		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["covariates"] = os.path.abspath(self._covariates)
		return outputs

class Build_Xmat_InputSpec(TraitedSpec):	
	onsetsFile = File(exists=True, mandatory=True) 
	covFile = File(exists=True, mandatory=True)
	TR = traits.Float(desc='TR length',mandatory=True)
	header = traits.Bool(desc='whether onsets file has a header or not',default=True)
	delim = traits.String(desc='delimiter used in onsets file',default=',')
	fillNa = traits.Bool(desc='Fill nans with 0',default=True)

class Build_Xmat_OutputSpec(TraitedSpec):
	xmat = File(exists=True)

class Build_Xmat(BaseInterface):
	input_spec = Build_Xmat_InputSpec
	output_spec = Build_Xmat_OutputSpec

	def _run_interface(self, runtime):
		import matplotlib
		matplotlib.use('Agg')
		import pandas as pd
		import numpy as np
		import seaborn as sns
		from nipy.modalities.fmri.hemodynamic_models import glover_hrf
		
		covFile = self.inputs.covFile
		onsetsFile =  self.inputs.onsertsFile
		TR = float(self.inputs.TR)
		fillNa = self.inputs.fillNa
		header = self.inputs.header
		if not self.inputs.header:
			header = None
		else:
			header = 0
		delim = self.inputs.delim

		hrf = glover_hrf(tr = TR,oversampling=1)

		#Check if we're dealing with multiple files that need to be concat
		assert type(covFile) == type(onsetsFile), "Covariates and onsets must both be a single a file or list of files!"
	            
		#COVARIATES

		if isinstance(covFile,list):
	 	    covs = []
		    for i, f in enumerate(covFile):
		        F = pd.read_csv(f)
		        F.columns = [str(i)+'_' + c if 'spike' in c else c for c in F.columns]
		        covs.append(F)
		    C = pd.concat(covs,axis=0,ignore_index=True)
		    
		    #Create runwise dummy coded regressors
		    numRuns = len(covs)
		    numTrs = C.shape[0]/numRuns
		    runDummies = np.zeros([C.shape[0],len(covs)])
		    
		    for runCount in xrange(len(covs)):
		        runDummies[runCount*numTrs:runCount*numTrs+numTrs,runCount] = 1
		    runDummies = pd.DataFrame(runDummies,columns = ['run'+str(elem) for elem in xrange(len(covs))])
		    
		    C = pd.concat([C,runDummies],axis=1)
		    
		    #ONSETS
		    #Load onsets, convert to TRs, get unique stimNames
		    onsets = []
		    for i, f in enumerate(onsetsFile):
		        F = pd.read_csv(f,header=header,delimiter=delim)
		        if header is None:
		        	if isinstance(F.iloc[0,0],str):
		        		F.columns = ['Stim','Onset']
		        	else:
		        		F.columns = ['Onset','Stim']
		        F['Onset'] = F['Onset'].apply(lambda x: int(np.floor(x/TR)))
		        F['Onset'] += numTrs*i
		        onsets.append(F)     
		    O = pd.concat(onsets,axis=0,ignore_index=True)
    
		else:
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
		    numRuns = 1
    
		#Build dummy codes
		#Subtract one from onsets row, because pd DFs are 0-indexed but TRs are 1-indexed
		X = pd.DataFrame(columns=O.Stim.unique(),data=np.zeros([C.shape[0],len(O.Stim.unique())]))
		for i, row in O.iterrows():
		    X.ix[row['Onset']-1,row['Stim']] = 1
		X = X.reindex_axis(sorted(X.columns), axis=1)

		#Convolve with hrf, concat with covs
		for i in xrange(X.shape[1]):
		    X.iloc[:,i] = np.convolve(hrf,X.iloc[:,i])[:X.shape[0]]
		X = pd.concat([X,C],axis=1)

		if fillNa:
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
		
		fig.savefig('Xmat.png')
		plt.close(fig)
		del fig

		filename = 'Xmat.csv'
		X.to_csv(filename,index=False)
		self._xmat = filename

		runtime.returncode=0
		return runtime
	    
	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["xmat"] = os.path.abspath(self._xmat)
		return outputs
