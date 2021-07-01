import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
	def __init__(self,ninput,nh,noutput,nlayers):
		super(MLP,self).__init__()
		self.layers=nn.ModuleList();
		if nlayers==1:
			self.layers.append(nn.Linear(ninput,noutput));
		else:
			self.layers.append(nn.Linear(ninput,nh));
			for i in range(1,nlayers-1):
				self.layers.append(nn.Linear(nh,nh));
			self.layers.append(nn.Linear(nh,noutput));
		return;
	#
	def forward(self,x,fw=None):
		if fw is None:
			batch=x.shape[0];
			x=x.view(batch,-1);
			x=self.layers[0](x);
			for i in range(1,len(self.layers)):
				x=F.relu(x);
				x=self.layers[i](x);
		else:
			batch=x.shape[0];
			x=x.view(batch,-1);
			x=torch.mm(x,fw.d['layers.0.weight'].t());
			x=x+fw.d['layers.0.bias'].view(1,-1);
			for i in range(1,len(self.layers)):
				x=F.relu(x);
				x=torch.mm(x,fw.d['layers.%d.weight'%i].t());
				x=x+fw.d['layers.%d.bias'%i].view(1,-1);
		
		return x;

class new(nn.Module):
	def __init__(self,params):
		super().__init__()
		self.nh=56;
		self.sz=224;
		self.t=MLP(nh,nh,4,2);
		self.initialize();
		return;
	
	def initialize(self,val=-5):
		#self.w.data.fill_(val);
		return;
	
	def generate_box(self,noise);
		box=F.sigmoid(self.t(noise));
		cx=box[:,0];
		cy=box[:,1];
		w=box[:,2];
		h=box[:,3];
		#First normalize things into 0,1
		x0=cx-w/2;
		y0=cy-h/2;
		x1=cx+w/2;
		y1=cy+h/2;
		x0=x0.clamp(0,1);
		y0=y0.clamp(0,1);
		x1=x1.clamp(0,1);
		y1=y1.clamp(0,1);
		sz=(x1-x0)*(y1-y0);
		return x0,y0,x1,y1,sz;
	
	#Generate weights for infilling
	#returns logit while the original implementation returns p
	def forward(self,N,T=0.1,eps=1e-9):
		noise=torch.Tensor(N,self.nh).normal_();
		x0,y0,x1,y1,sz=self.generate_box(noise);
		#And then quantize into ints
		x0=(x0.data*self.sz).clamp(0,self.sz-1)
		y0=(y0.data*self.sz).clamp(0,self.sz-1)
		x1=(x1.data*self.sz).clamp(0,self.sz-1)
		y1=(y1.data*self.sz).clamp(0,self.sz-1)
		mask=torch.Tensor(N,1,self.sz,self.sz).fill_(0);
		for i in range(N):
			mask[i,:,y0[i]:y1[i],x0[i]:x1[i]]=1;
		
		return mask,sz.data,noise;
	
	def policy_gradient(self,noise,reward):
		avg=reward.mean();
		reward=reward-avg;
		x0,y0,x1,y1,sz=self.generate_box(noise);
		
		
		return p.sum();
	
	def size_loss(self,direction):
		w=self.w.view(1,1,self.nh,self.nh);
		p=torch.sigmoid(-direction*w);
		if self.sz!=self.nh:
			p=F.upsample(p.clone(),size=[224,224],mode='bilinear');
		diff=((p[:,:,1:,:]-p[:,:,:-1,:])**2).sum()
		diff+=((p[:,:,:,1:]-p[:,:,:,:-1])**2).sum();
		return diff;
	