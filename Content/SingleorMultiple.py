import essentia
import essentia.standard as ess
import numpy as np
import scipy.interpolate as sci
import scipy.signal as scisig

M=2048
N=2048
H=512
fs=44100

def ComplexOrNot(audio):
        loudness_profile=[]
	loud=ess.Loudness()
	pd=ess.PeakDetection()
	spec=ess.Spectrum()
	bb=ess.BarkBands(numberBands=24,sampleRate = fs)

	bbloudness=[]
	loudness_profile=[]


	d=np.arange(0,22050+(44100/1024.0),44100/1024.0)
	d=d/1000.0
	A=(-2.184*(d**-0.8))+(6.5*np.e**(-0.6*((d-3.3)**2)))-(0.001*(d**3.6))
	
	MidEarFilter=10**(A/20.0)


	for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize=256):
    		#l=loud(frame)
    		#loudness_profile.append(l)
    		bands=bb(essentia.array(spec(frame)*MidEarFilter)) #bark band energy
    		bbloudness.append(sum(np.power(bands,0.23))) #Energy summation over bands for a frame
    
	#l_profile=np.array(loudness_profile) # loudness l(t)

	#-----bark band method----------
	bbl_profile=np.array(bbloudness)
	l_profile=bbl_profile # loudness l(t)


	fs_new=float(len(l_profile)*fs/len(audio)) #loudness sampling rate
	cutoff=10.0/fs_new # filter cut-off


	#---------------LP filter and Smoothing--------------------------------------------
	b, a = scisig.butter(5, cutoff, 'low') 
	l_profile_filt=scisig.lfilter(b,a,l_profile) # loudness smoothing
        l_profile_filt[l_profile_filt==0]=np.spacing(1)
	
	#-------------------10% threshold-------------------------------------------------- 
	lm=l_profile_filt.max()
	l_prof_filt_thresh=l_profile_filt[l_profile_filt>0.1*lm]
	
	# log-loudness
        l_prof_filt_thresh=np.log(l_prof_filt_thresh)
	
	# difference vector calculation
	#l_t_add=np.hstack((0.0,l_prof_filt_thresh))
        l_t_deriv=np.diff(l_prof_filt_thresh)
	if len(l_t_deriv)>0:
		deriv_max=np.max(l_t_deriv)
		if len(l_t_deriv)>1:	
			peak_pos,peak_amp=pd(essentia.array(l_t_deriv))
	#return peak_pos,peak_amp,l_t_deriv,l_prof_filt_thresh
	#return l_t_deriv,l_prof_filt_thresh

		
			if len(peak_pos[peak_amp>=0.1*deriv_max])>1:
				return 1
			else:
				return 0
		else:
			return 0
	else:
		return 0
	

	
