import essentia
import essentia.standard as ess
import numpy as np
import scipy.interpolate as sci
import scipy.signal as scisig
from dtw import dtw


M=2048
N=2048
H=512
fs=44100


def LoudnessProfile(audio):
        loudness_profile=[]
	loud=ess.Loudness()
	spec=ess.Spectrum()
	bb=ess.BarkBands(numberBands=24,sampleRate = fs)
	bbloudness=[]
	d=np.arange(0,22050+(44100/1024.0),44100/1024.0)
	d=d/1000.0
	A=(-2.184*(d**-0.8))+(6.5*np.e**(-0.6*((d-3.3)**2)))-(0.001*(d**3.6))
	
	MidEarFilter=10**(A/20.0)
	w=ess.Windowing(type="hann")

	for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize=256):
    		#l=loud(frame)
    		#loudness_profile.append(l)
    		bands=bb(essentia.array(spec(w(frame))*MidEarFilter)) #bark band energy
    		bbloudness.append(sum(np.power(bands,0.23))) #Energy summation over bands for a frame
    
	#l_profile=np.array(loudness_profile) # loudness l(t)

	#-----bark band method----------
	bbl_profile=np.array(bbloudness)
	l_profile=bbl_profile # loudness l(t)


	# LP filter/Smoothing
	fs_new=44100/256.0 #loudness sampling rate
	cutoff=2.0/fs_new # filter cut-off
	b, a = scisig.butter(5, cutoff, 'low') 
	l_profile_filt=scisig.lfilter(b,a,l_profile)

	# Taking care of zeros for log operation
        l_profile_filt[l_profile_filt==0]=np.spacing(1)

	# Thresholding
	lm=l_profile_filt.max()
	l_prof_filt_thresh=l_profile_filt[l_profile_filt>=0.1*lm]
	l_finalprofile=np.log(l_prof_filt_thresh)
	l_t_deriv=np.gradient(l_finalprofile)
	return l_prof_filt_thresh, l_finalprofile, lm,  l_t_deriv

def LoudnessProfileClassification(l_prof_filt_thresh, l_finalprofile, lm):
	
	fs_new=44100/256.0
	normalized_t=np.arange(0,len(l_prof_filt_thresh))/float(len(l_prof_filt_thresh)-1)
	tm=normalized_t[l_prof_filt_thresh.argmax()]
        
	ts=0
	te=1

	rd1=tm-ts
	rd2=te-tm
	# Spline Approximation
	if ts==tm or tm==te:
		if rd1==0:
			rd1=np.spacing(1)
		else:
			rd2=np.spacing(1)
		tck1=sci.splrep(normalized_t,l_finalprofile,k=1)
	else:
		tck1=sci.splrep(normalized_t,l_finalprofile,k=1,t=[tm])
        
	# Descriptor Computation
        spline_approx=sci.splev(normalized_t,tck1)
	s1=(sci.splev(tm,tck1)-sci.splev(ts,tck1))/rd1
	if tm==te:
		s2=-1/rd2
	else:
		s2=(sci.splev(te,tck1)-sci.splev(tm,tck1))/rd2
	#ed1=normalized_t[l_prof_filt_thresh>=0.4*lm][-1]- normalized_t[l_prof_filt_thresh>=0.4*lm][0]	
	ed1=len(normalized_t[l_prof_filt_thresh>=0.4*lm])/fs_new
	ed2=normalized_t[l_prof_filt_thresh>=0.8*lm][-1]- normalized_t[l_prof_filt_thresh>=0.8*lm][0]	
	absed=(len(l_prof_filt_thresh)/fs_new)

	#ed=ess.EffectiveDuration()
	#ed3=ed(audio)
	return rd1,rd2,s1,s2,ed1,ed2,absed
        
#----------------------------------------------------

def ModulationDescriptors(l_profile):
	zcr=ess.ZeroCrossingRate()
	n=len(l_profile)
	l_mprofile=l_profile[int(0.1*n):n-int(0.1*n)]
	mf=scisig.medfilt(l_mprofile,int(0.2*len(l_mprofile)))
	modulated_sig=l_mprofile-mf

	mod_extent=np.sqrt(np.sum(modulated_sig**2)/float(len(mf)))	
	mod_rate=zcr(modulated_sig)
	return mod_extent,mod_rate


def SMDetect(l_deriv,alph,sTh):
	if len(l_deriv)>1:
		ons=ess.Onsets(alpha=alph,silenceThreshold=sTh,frameRate=172)
		det=ons(essentia.array(l_deriv).reshape(1,len(l_deriv)),[1])
		
		if len(det)>1:
			return 1
		else:
			return 0
		
	else:
		return 0
			
#----------Trial Functions---------------------------

def SMOnset(audio):	
	pd=ess.PeakDetection()
	ons=ess.Onsets(alpha=1,silenceThreshold=0.2)
	odf=ess.OnsetDetectionGlobal(sampleRate=44100)
	audio_odf=odf(audio)
	#return audio_odf
	if len(audio_odf)==1 or np.all(audio_odf==0):
		return 0	
	else:
		audio_ons=ons(np.array(audio_odf).reshape(1,len(audio_odf)),[1])
		#peak_pos,peak_amp=pd(essentia.array(audio_odf))
		#peak_max=peak_amp.max()
		if len(audio_ons)>1:#len(peak_pos[peak_amp>=0.2*peak_max])>1:
			return 1
		else:
			return 0


