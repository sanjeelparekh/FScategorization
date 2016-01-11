import os
import json
import features_2608 as features
import reduction
import evaluate
import numpy as np 
import lda


tags=list()
sound_idlist=list()

data=json.load(open('/media/mlpboon/D4EC9797EC97730A/SMC_MS_UPF_2014/Research project/Project-SMC/CCATest/DataContext.json','r'))

for sound_id,value in data.items():
	x_tag=[str(x) for x in value]
	tags += x_tag
		

unique_unfilteredTags=list(set(tags))
'''
nums=[num for num in unique_unfilteredTags if str.isdigit(str(num))]

sfx_words = ['fx','sfx','effects','effect']
sfx_case = [sfx for sfx in unique_unfilteredTags if str(sfx).lower() in sfx_words]

filter_out = nums+sfx_case

unique_filteredTags = [finset for finset in unique_unfilteredTags if not finset in filter_out]
final_vector=list()

'''
final_vector=list()
r=[]
unique_filteredTags = list(set([t.lower() for t in unique_unfilteredTags]))

IDlist=json.load(open('data/idList.json','r'))



for ids in IDlist:
	value=data[unicode(ids)]	
	x_tag=[str(x) for x in value]
	#try_tags=' '.join(x_tag)
	tags=x_tag
	#r.append(try_tags)
	feature_vector = np.zeros(len(unique_filteredTags))
	#r.append(sound_id)
	for tag in tags:
		try:
			pos = int(unique_filteredTags.index(tag.lower()))
                	feature_vector[pos] = 1
            	except:
                	pass
	final_vector.append(feature_vector)


n_dim=5
#tf-idf computation
usm=features.user_data()
cntxt=features.context(enable_descr=False)
redctxt=reduction.reduce_pca(cntxt,n=n_dim)
norm_matrix_pca=np.linalg.norm(redctxt,axis=1)

#X=np.array(cntxt.todense())			
X=np.array(final_vector)

vocab=unique_filteredTags
topics=n_dim
model = lda.LDA(n_topics=topics, n_iter=1500, random_state=1)
model.fit_transform(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
	print('Topic {}: {}'.format(i, ', '.join(topic_words)))


document_topic=model.doc_topic_
norm_matrix=np.linalg.norm(document_topic,axis=1)
csm=np.zeros((120,120))
for i in range(0,120):
		for j in range(0,120):
			csm[i,j]=np.dot(document_topic[i,:],document_topic[j,:])/(norm_matrix[i]*norm_matrix[j])


hsm=np.zeros((120,120))
for i in range(0,120):
		for j in range(0,120):
			hsm[i,j]=1-np.sum((np.sqrt(document_topic[i,:])-np.sqrt(document_topic[j,:]))**2)





con=np.zeros((120,120))
for i in range(0,120):
		for j in range(0,120):
			con[i,j]=np.dot(redctxt[i,:],redctxt[j,:])/(norm_matrix_pca[i]*norm_matrix_pca[j])
'''
csm=(0.5+(csm/2.0))-np.diag([1]*120)
hsm=(0.5+(hsm/2.0))-np.diag([1]*120)
con=(0.5+(con/2.0))-np.diag([1]*120)
'''




for i in range(0,120):
    csm[i,i]=0
    hsm[i,i]=0		
    con[i,i]=0

result_csm=evaluate.against_reference_sm(csm,usm)
result_hsm=evaluate.against_reference_sm(hsm,usm)
result_con=evaluate.against_reference_sm(con,usm)

#np.save('docTopic',document_topic)
'''
doc_top=dict()
dts_m,dts_n=document_topic.shape
for i in range(0,dts_m):	
	doc_top[sound_idlist[i]]=dict()
	for j in range(0,topics):
	
with open('TagTopics.json', 'w') as f:
    json.dump(doc_top, f)
'''

