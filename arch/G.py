import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

class new(nn.Module):
	def __init__(self,params):
		super().__init__()
		self.nh=56;
		self.sz=224;
		self.w=nn.Parameter(torch.Tensor(1,self.nh,self.nh));
		self.initialize();
		return;
	
	def initialize(self,val=-5):
		self.w.data.fill_(val);
		return;
	
	#Generate weights for infilling
	#returns logit while the original implementation returns p
	def forward(self,N,T=0.1,eps=1e-9):
		w=self.w.view(1,1,self.nh,self.nh);
		p=torch.sigmoid(w);
		if self.sz!=self.nh:
			p=F.upsample(p,size=[224,224],mode='bilinear');
		p=p.repeat(N,1,1,1);
		#"concrete dropout"
		w=torch.log(p+eps)-torch.log(1-p+eps); #Essentially w
		noise=p.clone().uniform_();
		noise=torch.log(noise+eps)-torch.log(1-noise+eps)
		w=-(w+noise)/T;
		return w;
	
	def l1(self,direction):
		w=self.w.view(1,1,self.nh,self.nh);
		p=torch.sigmoid(-direction*w);
		if self.sz!=self.nh:
			p=F.upsample(p.clone(),size=[224,224],mode='bilinear');
		return p.sum();
	
	def smoothness(self,direction):
		w=self.w.view(1,1,self.nh,self.nh);
		p=torch.sigmoid(-direction*w);
		if self.sz!=self.nh:
			p=F.upsample(p.clone(),size=[224,224],mode='bilinear');
		diff=((p[:,:,1:,:]-p[:,:,:-1,:])**2).sum()
		diff+=((p[:,:,:,1:]-p[:,:,:,:-1])**2).sum();
		return diff;
	