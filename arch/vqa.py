import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util.basic_layers as basic

class new(nn.Module):
	def __init__(self,params):
		super(new,self).__init__()
		nh=params.nh;
		nwords=params.nwords;
		self.qembed=basic.sent_encoder_rnn(params.nwords,params.nh,params.nhrnn,params.nlayers);
		self.qproj=nn.Linear(2*params.nlayers*params.nhrnn,params.nhjoint);
		self.iproj=nn.Linear(params.nhimage,params.nhjoint);
		self.classify=nn.Linear(params.nhjoint,params.nanswers);
		return;
	#
	def forward(self,I,Q):
		Q=self.qembed(Q);
		Q=self.qproj(Q);
		I=F.normalize(self.iproj(I.mean(1)),p=2,dim=1);
		pred=self.classify(F.dropout(F.tanh(Q)*F.tanh(I)));
		return pred;
	