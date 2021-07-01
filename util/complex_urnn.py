import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch.optim as optim
import math
import numpy as np


class RMSprop(optim.Optimizer):
	"""Implements RMSprop algorithm.
	Proposed by G. Hinton in his
	`course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
	The centered version first appears in `Generating Sequences
	With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
	Arguments:
	params (iterable): iterable of parameters to optimize or dicts defining
	parameter groups
	lr (float, optional): learning rate (default: 1e-2)
	momentum (float, optional): momentum factor (default: 0)
	alpha (float, optional): smoothing constant (default: 0.99)
	eps (float, optional): term added to the denominator to improve
	numerical stability (default: 1e-8)
	centered (bool, optional) : if ``True``, compute the centered RMSProp,
	the gradient is normalized by an estimation of its variance
	weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
	"""

	def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if not 0.0 <= weight_decay:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if not 0.0 <= alpha:
			raise ValueError("Invalid alpha value: {}".format(alpha))

		defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
		super(RMSpropU, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(RMSpropU, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('momentum', 0)
			group.setdefault('centered', False)

	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
		closure (callable, optional): A closure that reevaluates the model
		and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()
		
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('RMSprop does not support sparse gradients')
				state = self.state[p]
				
				# State initialization
				if len(state) == 0:
					state['step'] = 0
					state['square_avg'] = torch.zeros_like(p.data)
					if group['momentum'] > 0:
						state['momentum_buffer'] = torch.zeros_like(p.data)
					if group['centered']:
						state['grad_avg'] = torch.zeros_like(p.data)
				
				square_avg = state['square_avg']
				alpha = group['alpha']
				
				state['step'] += 1
				
				if group['weight_decay'] != 0:
					grad = grad.add(group['weight_decay'], p.data)
				
				square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
				
				if group['centered']:
					grad_avg = state['grad_avg']
					grad_avg.mul_(alpha).add_(1 - alpha, grad)
					avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
				else:
					avg = square_avg.sqrt().add_(group['eps'])
				
				unitary=False;
				try:
					a=p.unitary;
					unitary=True;
				except:
					pass;
				
				if unitary==True:
					debug=False;
					#A=GHW-WHG? GWH-WGH?
					#w=p.data[0,:,:].clone();
					#wi=p.data[1,:,:].clone();
					#dw=p.grad.data[0,:,:].clone();
					#dwi=p.grad.data[1,:,:].clone();
					#gxh=torch.mm(dw,w.t())+torch.mm(dwi,wi.t());
					#gxhi=torch.mm(dwi,w.t())-torch.mm(dw,wi.t());
					#a=gxh-gxh.t();
					#ai=gxhi+gxhi.t();
					
					#a=torch.mm(dw.t(),w)+torch.mm(dwi.t(),wi)-torch.mm(w.t(),dw)-torch.mm(wi.t(),dwi);
					#ai=torch.mm(dw.t(),wi)-torch.mm(dwi.t(),w)-torch.mm(w.t(),dwi)+torch.mm(wi.t(),dw);
					#Y=(I+lr/2 A)^-1 (I-lr/2 A)
					#def complex_inverse(x,xi):
					#	a=torch.inverse(x+torch.mm(torch.mm(xi,torch.inverse(x)),xi));
					#	b=-torch.inverse(xi+torch.mm(torch.mm(x,torch.inverse(xi)),x));
					#	return a,b;
					
					#def complex_inverse_v2(x,xi):
					#	x=x.cpu().numpy().astype('complex64')
					#	xi=xi.cpu().numpy().astype('complex64')
					#	a=x+1j*xi;
					#	a=np.linalg.inv(a);
					#	y=torch.from_numpy(np.real(a)).cuda();
					#	yi=torch.from_numpy(np.imag(a)).cuda();
					#	return y,yi;
					
					#a=a*group['lr']/2;
					#ai=ai*group['lr']/2;
					#y0,y0i=complex_inverse_v2(torch.eye(N).cuda()+a,ai)
					#y1=torch.eye(N).cuda()-a;
					#y1i=-ai;
					#y=torch.mm(y0,y1)-torch.mm(y0i,y1i);
					#yi=torch.mm(y0i,y1)+torch.mm(y0,y1i);
					#Wt=YW
					#wt=torch.mm(y,w)-torch.mm(yi,wi);
					#wti=torch.mm(yi,w)+torch.mm(y,wi);
					
					
					w=p.data[0,:,:].clone().cpu().numpy().astype('complex64');
					wi=p.data[1,:,:].clone().cpu().numpy().astype('complex64');
					dw=p.grad.data[0,:,:].clone().cpu().numpy().astype('complex64');
					dwi=p.grad.data[1,:,:].clone().cpu().numpy().astype('complex64');
					N=w.shape[0];
					lr=group['lr'];
					
					w=w+1j*wi;
					dw=dw+1j*dwi;
					#a=np.dot(dw.conj().T,w)-np.dot(w.conj().T,dw);
					a=np.dot(dw,w.conj().T)-np.dot(w,dw.conj().T);
					y=np.linalg.inv(np.eye(N,dtype='complex64')+lr/2*a)
					y=np.dot(y,np.eye(N,dtype='complex64')-lr/2*a)
					y=np.dot(y,w);
					wt=torch.from_numpy(np.real(y)).cuda();
					wti=torch.from_numpy(np.imag(y)).cuda();
					
					p.data[0,:,:]=wt;
					p.data[1,:,:]=wti;
					
					if debug:
						#Check WtHWt
						r=torch.mm(wt.t(),wt)+torch.mm(wti.t(),wti);
						c=torch.mm(wt.t(),wti)-torch.mm(wti.t(),wt);
						print(r,c);
				else:
					if group['momentum'] > 0:
						buf = state['momentum_buffer']
						buf.mul_(group['momentum']).addcdiv_(grad, avg)
						p.data.add_(-group['lr'], buf)
					else:
						p.data.addcdiv_(-group['lr'], grad, avg)
		
		return loss


class Adam(optim.Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay, amsgrad=amsgrad)
		super(Adam, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(Adam, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('amsgrad', False)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
				amsgrad = group['amsgrad']
				
				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(p.data)
					if amsgrad:
						# Maintains max of all exp. moving avg. of sq. grad. values
						state['max_exp_avg_sq'] = torch.zeros_like(p.data)
				
				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				if amsgrad:
					max_exp_avg_sq = state['max_exp_avg_sq']
				beta1, beta2 = group['betas']
				
				state['step'] += 1
				
				if group['weight_decay'] != 0:
					grad = grad.add(group['weight_decay'], p.data)
				
				# Decay the first and second moment running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				if amsgrad:
					# Maintains the maximum of all 2nd moment running avg. till now
					torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
					# Use the max. for normalizing running avg. of gradient
					denom = max_exp_avg_sq.sqrt().add_(group['eps'])
				else:
					denom = exp_avg_sq.sqrt().add_(group['eps'])
				
				bias_correction1 = 1 - beta1 ** state['step']
				bias_correction2 = 1 - beta2 ** state['step']
				step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
				
				unitary=False;
				try:
					a=p.unitary;
					unitary=True;
				except:
					pass;
				
				debug=False;
				if unitary==True:
					w=p.data[0,:,:].clone().cpu().numpy().astype('complex64');
					wi=p.data[1,:,:].clone().cpu().numpy().astype('complex64');
					dw=exp_avg[0,:,:].clone().cpu().numpy().astype('complex64');
					dwi=exp_avg[1,:,:].clone().cpu().numpy().astype('complex64');
					N=w.shape[0];
					
					w=w+1j*wi;
					dw=dw+1j*dwi;
					#a=np.dot(dw.conj().T,w)-np.dot(w.conj().T,dw);
					a=np.dot(dw,w.conj().T)-np.dot(w,dw.conj().T);
					y=np.linalg.inv(np.eye(N,dtype='complex64')+step_size/2*a)
					y=np.dot(y,np.eye(N,dtype='complex64')-step_size/2*a)
					y=np.dot(y,w);
					wt=torch.from_numpy(np.real(y)).cuda();
					wti=torch.from_numpy(np.imag(y)).cuda();
					
					p.data[0,:,:]=wt;
					p.data[1,:,:]=wti;
					
					if debug:
						#Check WtHWt
						r=torch.mm(wt.t(),wt)+torch.mm(wti.t(),wti);
						c=torch.mm(wt.t(),wti)-torch.mm(wti.t(),wt);
						print(r,c);
				else:
					p.data.addcdiv_(-step_size, exp_avg, denom)
		
		return loss


class fc_urnn_unit(nn.Module):
	def __init__(self,nhinput,nh,nhoutput):
		super(fc_urnn_unit,self).__init__()
		self.w=nn.Parameter(torch.cat((torch.eye(nh).view(1,nh,nh),torch.zeros(1,nh,nh)),dim=0));
		self.w.unitary=True;
		self.v=nn.Parameter(torch.Tensor(nh,nhinput).uniform_(-0.001,0.001));
		self.vi=nn.Parameter(torch.Tensor(nh,nhinput).uniform_(-0.001,0.001));
		self.b=nn.Parameter(torch.Tensor(nh).uniform_(-0.001,0))
		
		self.u=nn.Parameter(torch.Tensor(nhoutput,nh).uniform_(-0.001,0.001));
		self.ui=nn.Parameter(torch.Tensor(nhoutput,nh).uniform_(-0.001,0.001));
		self.c=nn.Parameter(torch.Tensor(nhoutput).uniform_(-0.001,0.001))
	#
	def forward(self,h0,h0i,x):
		w=self.w[0,:,:].clone();
		wi=self.w[1,:,:].clone();
		#h=Wh0+Vx
		h=torch.mm(h0,w.t())-torch.mm(h0i,wi.t());
		hi=torch.mm(h0,wi.t())+torch.mm(h0i,w.t());
		h+=torch.mm(x,self.v.t());
		hi+=torch.mm(x,self.vi.t());
		#h=sigma(h)
		l=torch.sqrt(h**2+hi**2+1e-9);
		h=F.relu(l+self.b.view(1,-1))*h/(l+1e-9);
		hi=F.relu(l+self.b.view(1,-1))*hi/(l+1e-9);
		#y=Uh+c
		y=torch.mm(h,self.u.t())-torch.mm(hi,self.ui.t())+self.c.view(1,-1);
		return h,hi,y;

#RNN, trying to be compatible with LSTM. Using c as real and h as virtual.
class fc_urnn(nn.Module):
	def __init__(self,nhinput,nh,nlayers):
		super(fc_urnn,self).__init__()
		self.unit0=fc_urnn_unit(nhinput,nh,nh);
		self.unit1=fc_urnn_unit(nh,nh,nh);
		self.unit2=fc_urnn_unit(nh,nh,nh);
		self.nlayers=nlayers;
	#
	def forward(self,input,state0):
		self.layers=[self.unit0,self.unit1,self.unit2];
		output=[];
		prev_s=state0[0];
		prev_si=state0[1];
		length=input.size(0);
		batch=input.size(1);
		for i in range(0,length):
			x=input[i,:,:];
			s=[];
			si=[];
			for j in range(0,self.nlayers):
				sj,sij,x=self.layers[j](prev_s[j,:,:],prev_si[j,:,:],x);
				s.append(sj.view(1,batch,-1));
				si.append(sij.view(1,batch,-1));
			prev_s=torch.cat(s,dim=0);
			prev_si=torch.cat(si,dim=0);
			output.append(x.view(1,batch,-1));
		
		output=torch.cat(output,dim=0);
		return output,(prev_s,prev_si);
	
