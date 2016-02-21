import essentia.standard as ess
import os,sys
import dynamic_profiles as dp
import SingleorMultiple as sm
import numpy as np
import json


fs=44100

inputDir='/media/mlpboon/D4EC9797EC97730A/SMC_MS_UPF_2014/Research project/Project-SMC/Dataset/DatasetPreviews'
sPath=list()
for path,dirs,files in os.walk(inputDir):
    for name in files:
        sPath.append(os.path.join(path,name))


index_single=[]
index_complex=[]
index_stable=[]
index_inc=[]
index_dec=[]
index_incdec=[]
index_impulse=[]
index_others=[]
index_othersdec=[]
index_othersinc=[]

testing_features=list()
lpresults_sfx=dict() 
 
for i in range(0,len(sPath)):
    
    sound_id=sPath[i].split('/')[-1].split('_')[0]
    lpresults_sfx[sound_id]=dict()
    print 'processing sound %d: %s' % (i, sound_id)
    loader = ess.MonoLoader(filename = sPath[i],sampleRate = fs)
    check_audio=loader()
    Profile,logProfile,lm,onset_func=dp.LoudnessProfile(check_audio)	
    complex_flag=dp.SMDetect(onset_func,0.5,0.02)
    	
    if complex_flag==0:	
	
	index_single.append(i)
	#testfeature=dp.dynamicProfile(check_audio)
        #testing_features.append(list(testfeature))
    #else:
	#continue 
		
	rd1,rd2,s1,s2,ed1,ed2,absed=dp.LoudnessProfileClassification(Profile,logProfile,lm)

	if (ed1<0.25 or absed<0.3):
		index_impulse.append(i)
		lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='imp'
	else:			
    		if ed2>=0.7:
    			index_stable.append(i)
    			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='stb'
		elif rd1>=0.7:# and np.abs(ed2-0.3)<0.1:
    			index_inc.append(i)
    			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='inc'
		elif rd2>=0.7:# and np.abs(ed2-0.3)<0.1:
			index_dec.append(i)
    			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='dec'			
		elif np.abs(rd1-0.5)<=0.1:
			index_incdec.append(i)
    			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='ind'		
		elif np.abs(rd1-0.35)<0.05:
			index_othersdec.append(i)
			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='othdec'
		elif np.abs(rd1-0.65)<0.05:
			index_othersinc.append(i)
			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='othinc'
		else:
			index_others.append(i)
			lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='oth'
    else:
	index_complex.append(i)
        lpresults_sfx[sound_id][".lowlevel.Loudness_Profile"]='cmp'
	
with open('LPDescriptors.json', 'w') as f:
    json.dump(lpresults_sfx, f)
	
  	
