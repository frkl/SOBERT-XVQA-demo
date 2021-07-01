import torch
import math

#Convert a list of list of tokens into a 2D matrix
def token2id(tokens,d=None,cutoff=0):
	if d==None:
		#Dictionary not available, build dictionary first
		words=dict();
		for s in tokens:
			for w in s:
				if w in words:
					words[w]+=1;
				else:
					words[w]=1;
		#Remove words occured less than threshold, sort by occurence.
		words=sorted([pair for pair in words.items() if pair[1]>cutoff],reverse=True,key=lambda x:x[1]);
		#The list of words is dictionary
		d=[p[0] for p in words];
	#Create a dictionary word to id mapping (1-indexing. 0 defaults to be "nothing" and len(d)+1 is UNK)
	words2id=dict(zip(d,list(range(1,len(d)+1))));
	#First pass, count max sentence length
	maxl=0;
	for s in tokens:
		maxl=max(maxl,len(s));
	#Second pass, parse tokens
	ids=torch.LongTensor(len(tokens),maxl).fill_(0);
	for i in range(len(tokens)):
		s=tokens[i];
		for j in range(len(s)):
			if s[j] in words2id:
				ids[i,j]=words2id[s[j]];
			else:
				ids[i,j]=len(d)+1; #UNK
	return ids,d;

def id2bow(tokens,d):
	batch=tokens.size(0);
	bow=torch.Tensor(batch,len(d)+1).fill_(0);
	for i in range(0,batch):
		l=torch.Tensor(len(d)+1).fill_(0);
		t=tokens[i];
		if t.sum()>0:
			ind=t.nonzero().view(-1);
			l[t[ind]-1]=1;
		bow[i,:]=l;
	return bow;
		

def id2words(arr,d):
	words=[];
	for i in range(0,arr.size(0)):
		w=[];
		for j in range(0,arr.size(1)):
			if arr[i,j]>0 and arr[i,j]<=len(d):
				w.append(d[arr[i,j]-1]);
			elif arr[i,j]==len(d)+1:
				w.append('UNK');
		words.append(w);
	return words;

