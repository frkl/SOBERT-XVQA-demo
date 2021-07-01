import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class residual_1d_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(residual_1d_v2,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.conv2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.scale=scale;
	#
	def forward(self,x):
		residual=self.conv1(x);
		residual=F.avg_pool1d(residual,round(1/self.scale));
		residual=F.relu(residual);
		residual=self.conv2(residual);
		if self.scale!=1:
			pass_through=F.avg_pool1d(x,round(1/self.scale));
		else:
			pass_through=x;
		return residual+pass_through;

#3 residual layers
class residual_tri_1d_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(residual_tri_1d_v2,self).__init__()
		self.res1=residual_1d_v2(nh,k);
		self.res2=residual_1d_v2(nh,k);
		self.res3=residual_1d_v2(nh,k,scale);
	#
	def forward(self,x):
		out=self.res1(x);
		out=self.res2(out);
		out=self.res3(out);
		return out;

class residual_1d_t_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(residual_1d_t_v2,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.conv2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.scale=scale;
		self.nh=nh;
	#
	def forward(self,x,noise=None):
		residual=self.conv1(x);
		residual=F.adaptive_avg_pool1d(residual,round(x.size(2)*self.scale));
		if not(noise is None):
			residual=residual+noise;
		residual=F.relu(residual);
		residual=self.conv2(residual);
		if self.scale!=1:
			pass_through=F.adaptive_avg_pool1d(x,round(x.size(2)*self.scale));
		else:
			pass_through=x;
		return residual+pass_through;

class residual_1d_t_v3(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(residual_1d_t_v3,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.conv2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.scale=scale;
		self.nh=nh;
	#
	def forward(self,x,noise=None):
		residual=self.conv1(x);
		norm1=torch.sqrt((residual*residual).mean(1,keepdim=True)+1e-8)
		residual=residual/norm1;
		residual=F.adaptive_avg_pool1d(residual,round(x.size(2)*self.scale));
		if not(noise is None):
			residual=residual+noise;
		residual=F.relu(residual);
		residual=self.conv2(residual);
		#norm2=torch.sqrt((residual*residual).mean(1,keepdim=True)+1e-8)
		#residual=residual/norm2;
		if self.scale!=1:
			pass_through=F.adaptive_avg_pool1d(x,round(x.size(2)*self.scale));
		else:
			pass_through=x;
		return residual+pass_through;



#3 residual layers
class residual_tri_1d_t_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(residual_tri_1d_t_v2,self).__init__()
		self.res1=residual_1d_t_v2(nh,k,scale);
		self.res2=residual_1d_t_v2(nh,k);
		self.res3=residual_1d_t_v2(nh,k);
	#
	def forward(self,x,noise=None):
		out=self.res1(x,noise);
		out=self.res2(out);
		out=self.res3(out);
		return out;



class gated_residual_1d_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(gated_residual_1d_v2,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.conv2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.gate2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.scale=scale;
		self.nh=nh;
	#
	def forward(self,x):
		residual=self.conv1(x);
		residual=F.avg_pool1d(residual,round(1/self.scale));
		residual=F.relu(residual);
		g2=F.sigmoid(self.gate2(residual));
		residual=self.conv2(residual);
		residual=residual*g2;
		if self.scale!=1:
			pass_through=F.avg_pool1d(x,round(1/self.scale));
		else:
			pass_through=x;
		return residual+pass_through;

#3 residual layers
class gated_residual_tri_1d_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(gated_residual_tri_1d_v2,self).__init__()
		self.res1=gated_residual_1d_v2(nh,k);
		self.res2=gated_residual_1d_v2(nh,k);
		self.res3=gated_residual_1d_v2(nh,k,scale);
	#
	def forward(self,x):
		out=self.res1(x);
		out=self.res2(out);
		out=self.res3(out);
		return out;

class gated_residual_1d_t_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(gated_residual_1d_t_v2,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.conv2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.gate2=nn.Conv1d(nh,nh,k,padding=int((k-1)/2));
		self.scale=scale;
		self.nh=nh;
	#
	def forward(self,x,noise=None):
		residual=self.conv1(x);
		residual=F.adaptive_avg_pool1d(residual,round(x.size(2)*self.scale));
		if not(noise is None):
			residual=residual+noise;
		residual=F.relu(residual);
		g2=F.sigmoid(self.gate2(residual));
		residual=self.conv2(residual);
		residual=residual*g2;
		if self.scale!=1:
			pass_through=F.adaptive_avg_pool1d(x,round(x.size(2)*self.scale));
		else:
			pass_through=x;
		return residual+pass_through;

#3 residual layers
class gated_residual_tri_1d_t_v2(nn.Module):
	def __init__(self,nh,k,scale=1):
		super(gated_residual_tri_1d_t_v2,self).__init__()
		self.res1=gated_residual_1d_t_v2(nh,k,scale);
		self.res2=gated_residual_1d_t_v2(nh,k);
		self.res3=gated_residual_1d_t_v2(nh,k);
	#
	def forward(self,x,noise=None):
		out=self.res1(x,noise);
		out=self.res2(out);
		out=self.res3(out);
		return out;

def logexpm1(x):
	l=torch.clamp(x,min=9);
	m=torch.clamp(x,max=9,min=0.69314718056);
	s=torch.clamp(x,min=1e-20,max=0.69314718056);
	val_l=l;
	val_m=torch.log(torch.exp(m)-1);
	val_s=torch.log(expm1(s));
	return val_l+val_m+val_s-9;

def logsumexp(inputs, dim=None, keepdim=False):
	return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim);

def logsumexp_z(inputs, dim=None, keepdim=False):
	z=inputs.mean(dim=dim).view(-1,1);
	inputs_z=torch.cat((inputs,z),dim=dim);
	return (inputs_z - F.log_softmax(inputs_z)).mean(dim, keepdim=keepdim);

def expm1(x):
	v=x;
	v=(v/8+1)*x
	v=(v/7+1)*x
	v=(v/6+1)*x
	v=(v/5+1)*x
	v=(v/4+1)*x
	v=(v/3+1)*x
	v=(v/2+1)*x
	return v;

def log1p(x):
	v=x/10;
	v=-v*x+x/9;
	v=-v*x+x/8;
	v=-v*x+x/7;
	v=-v*x+x/6;
	v=-v*x+x/5;
	v=-v*x+x/4;
	v=-v*x+x/3;
	v=-v*x+x/2;
	v=-v*x+x;
	return v;

def logsigmoid(x):
	pos=torch.clamp(x,min=1);
	neg=torch.clamp(x,max=1);
	out=-log1p(torch.exp(-pos))+neg-torch.log(1+torch.exp(neg))+0.31326168751;
	return out;

def logitexp(logp):
	pos=torch.clamp(logp,min=-0.69314718056);
	neg=torch.clamp(logp,max=-0.69314718056);
	neg_val=neg-torch.log(1-torch.exp(neg));
	pos_val=-torch.log(torch.clamp(expm1(-pos),min=1e-20));
	return pos_val+neg_val;

class verdict(nn.Module):
	def __init__(self,nh,type='regress'):
		super(verdict,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,1);
		self.conv2=nn.Conv1d(nh,nh,1);
		self.conv3=nn.Conv1d(nh,nh,1);
		self.type=type;
		self.nh=nh;
	#
	def forward(self,x,y):
		batch=x.size(0);
		x=self.conv1(x);
		y=self.conv2(y);
		if x.size(2)>y.size(2):
			y=F.adaptive_avg_pool1d(y,x.size(2));
		elif x.size(2)<y.size(2):
			x=F.adaptive_avg_pool1d(x,y.size(2));
		x=self.conv3(F.relu(x+y));
		output=x.permute(0,2,1).clone().view(-1,self.nh).mean(dim=1).view(batch,-1);
		if self.type=='regress': #Good
			output=output.mean(dim=1);
		elif self.type=='threshold':
			output=-F.relu(-output).mean(dim=1);
		elif self.type=='min':
			output,_=output.min(dim=1);
		elif self.type=='diff-log': #very low quality
			p=logsumexp(output,dim=1);
			pinv=logsumexp(-output,dim=1);
			output=p-pinv;
		elif self.type=='diff': #ok
			pos,_=output.max(dim=1);
			neg,_=output.min(dim=1);neg=-neg;
			output=pos-neg;
		elif self.type=='odd-no-norm': #Good
			s=output.sum(dim=1);
			p=logsumexp(output,dim=1);
			output=(p-s)/output.size(1);
		elif self.type=='odd': #exploded
			N=output.size(1);
			output=output;
			p=logsumexp(output,dim=1);
			s=logsigmoid(-output).sum(dim=1);
			output=logitexp(p+s);
		elif self.type=='odd-logp': #garbage
			N=output.size(1);
			output=F.relu(output)/N;
			p=logsumexp(logexpm1(output),dim=1);
			s=-output.sum(dim=1);
			output=logitexp((p+s));
		else:
			print('MLP_ent: non-existing type')
		output=output.view(batch,1); #Normalize over animation length
		return output;

#Here y can be shorter than x and z.
class verdict3(nn.Module):
	def __init__(self,nh,type='regress'):
		super(verdict3,self).__init__()
		self.v12=verdict(nh,type);
		self.v13=verdict(nh,type);
		self.v32=verdict(nh,type);
	#
	def forward(self,x,y,z):
		return self.v12(x,y)+self.v32(z,y)+self.v13(x,z);

class plot(nn.Module):
	def __init__(self,nh):
		super(plot,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,1);
		self.conv2=nn.Conv1d(nh,nh,1);
		self.type=type;
		self.nh=nh;
	#
	def forward(self,context,L=0,noise=None):
		batch=context.size(0);
		if L>0:
			context=F.adaptive_avg_pool1d(context,L);
		if noise is None:
			noise=Variable(context.data.clone().normal_());
		x=self.conv1(noise);
		x=self.conv2(F.relu(x+context));
		return x;

class combine_mlp(nn.Module):
	def __init__(self,nh):
		super(combine_mlp,self).__init__()
		self.conv1=nn.Conv1d(nh,nh,1);
		self.conv2=nn.Conv1d(nh,nh,1);
		self.conv3=nn.Conv1d(nh,nh,1);
	#
	def forward(self,x,y):
		batch=x.size(0);
		x=self.conv1(x);
		y=self.conv2(y);
		if x.size(2)>y.size(2):
			y=F.adaptive_avg_pool1d(y,x.size(2));
		elif x.size(2)<y.size(2):
			x=F.adaptive_avg_pool1d(x,y.size(2));
		z=self.conv3(F.relu(x+y));
		return z;

class text_encoder_bow(nn.Module):
	def __init__(self,nwords,nh):
		super(text_encoder_bow,self).__init__()
		self.embed=nn.Embedding(nwords+2,nh,padding_idx=0);
		self.fc1=nn.Linear(nh,nh);
		self.nh=nh;
	#
	def forward(self,context):
		v,_=self.embed(context).max(1);
		#v=F.relu(v);
		v=self.fc1(v);
		return v;

class local_context_encoder(nn.Module):
	def __init__(self,nwords,nh):
		super(local_context_encoder,self).__init__()
		self.embed=nn.Embedding(nwords+2,nh,padding_idx=0);
		self.conv1=nn.Conv1d(nh,nh,1);
		self.nh=nh;
	#
	def forward(self,context):
		lvid=context.size(1);
		lsent=context.size(2);
		context=context.view(-1,lsent);
		v=self.embed(context).mean(1).view(-1,lvid,self.nh).permute(0,2,1);
		v=F.relu(v);
		v=self.conv1(v);
		return v;

class sent_encoder_rnn(nn.Module):
	def __init__(self,nwords,nh,nhrnn=-1,nlayers=2):
		super(sent_encoder_rnn,self).__init__()
		if nhrnn<0:
			nhrnn=nh;
		self.embed=nn.Embedding(nwords+2,nh,padding_idx=nwords+1); #Pad 0, UNK, hence +2
		self.encoder=nn.LSTM(nh,nhrnn,nlayers,batch_first=True);
		self.nh=nh;
		self.nhrnn=nhrnn;
		self.dummy_h=nn.Parameter(torch.zeros(nlayers,1,nhrnn));
		self.dummy_c=nn.Parameter(torch.zeros(nlayers,1,nhrnn));
	#
	def sort(self,sent):
		#self.encoder.flatten_parameters()
		l=sent.data.gt(0).long().sum(1);
		_,sort_ind=l.sort(0,True);
		_,sort_ind_inv=sort_ind.sort(0);
		n=l.gt(0).long().sum();
		if n>0:
			sort_ind=sort_ind[:n].clone();
			sent_sorted=sent[sort_ind,:];
			return sent_sorted,sort_ind,sort_ind_inv;
		else:
			return None,None,None;
	#
	def forward(self,sent):
		#sent: batch x max_length tensor. 0=nothing, >0=word. Left aligned
		#embed_sent: batch x (2 x nlayers x nhrnn) tensor.
		sent_sorted,sort_ind,sort_ind_inv=self.sort(sent);
		if sort_ind is None:
			dummy_h=self.dummy_h.repeat(1,sent.size(0),1);
			dummy_c=self.dummy_c.repeat(1,sent.size(0),1);
			embed_sent=torch.cat((dummy_c,dummy_h),0);
			return embed_sent.permute(1,0,2).clone().view(sent.size(0),-1);
		#
		l=sent_sorted.gt(0).long().sum(1).data.tolist();
		embed_w=self.embed(sent_sorted); #batch x max_length x nh
		dummy_h=self.dummy_h.repeat(1,sent_sorted.size(0),1);
		dummy_c=self.dummy_c.repeat(1,sent_sorted.size(0),1);
		packed=torch.nn.utils.rnn.pack_padded_sequence(embed_w,l,batch_first=True);
		_,hidden=self.encoder(packed,(dummy_c,dummy_h));
		embed_sent=torch.cat((hidden[0],hidden[1]),0);
		if sent_sorted.size(0)<sent.size(0):
			dummy_h_0=self.dummy_h.repeat(1,sent.size(0)-sent_sorted.size(0),1);
			dummy_c_0=self.dummy_c.repeat(1,sent.size(0)-sent_sorted.size(0),1);
			embed_sent_0=torch.cat((dummy_c_0,dummy_h_0),0);
			embed_sent=torch.cat((embed_sent,embed_sent_0),1);
		embed_sent=embed_sent[:,sort_ind_inv,:];
		embed_sent=F.relu(embed_sent.permute(1,0,2).clone().view(sent.size(0),-1));
		return embed_sent;

class sent_encoder_rnn_gru(nn.Module):
	def __init__(self,nwords,nh,nhrnn=-1,nlayers=2):
		super(sent_encoder_rnn_gru,self).__init__()
		if nhrnn<0:
			nhrnn=nh;
		self.embed=nn.Embedding(nwords+2,nh,padding_idx=nwords+1); #Pad 0, UNK, hence +2
		self.encoder=nn.GRU(nh,nhrnn,nlayers,batch_first=True);
		self.nh=nh;
		self.nhrnn=nhrnn;
		self.dummy_h=nn.Parameter(torch.zeros(nlayers,1,nhrnn));
	#
	def sort(self,sent):
		#self.encoder.flatten_parameters()
		l=sent.data.gt(0).long().sum(1);
		_,sort_ind=l.sort(0,True);
		_,sort_ind_inv=sort_ind.sort(0);
		n=l.gt(0).long().sum();
		if n>0:
			sort_ind=sort_ind[:n].clone();
			sent_sorted=sent[sort_ind,:];
			return sent_sorted,sort_ind,sort_ind_inv;
		else:
			return None,None,None;
	#
	def forward(self,sent):
		#sent: batch x max_length tensor. 0=nothing, >0=word. Left aligned
		#embed_sent: batch x (2 x nlayers x nhrnn) tensor.
		sent_sorted,sort_ind,sort_ind_inv=self.sort(sent);
		if sort_ind is None:
			dummy_h=self.dummy_h.repeat(1,sent.size(0),1);
			embed_sent=dummy_h;
			return embed_sent.permute(1,0,2).clone().view(sent.size(0),-1);
		#
		l=sent_sorted.gt(0).long().sum(1).data.tolist();
		embed_w=self.embed(sent_sorted); #batch x max_length x nh
		dummy_h=self.dummy_h.repeat(1,sent_sorted.size(0),1);
		packed=torch.nn.utils.rnn.pack_padded_sequence(embed_w,l,batch_first=True);
		_,hidden=self.encoder(packed,dummy_h);
		embed_sent=hidden;
		if sent_sorted.size(0)<sent.size(0):
			dummy_h_0=self.dummy_h.repeat(1,sent.size(0)-sent_sorted.size(0),1);
			embed_sent_0=dummy_h_0;
			embed_sent=torch.cat((embed_sent,embed_sent_0),1);
		embed_sent=embed_sent[:,sort_ind_inv,:];
		embed_sent=F.relu(embed_sent.permute(1,0,2).clone().view(sent.size(0),-1));
		return embed_sent;

class sent_encoder_rnn_0(nn.Module):
	def __init__(self,nwords,nh,nhrnn=-1,nlayers=2):
		super(sent_encoder_rnn_0,self).__init__()
		if nhrnn<0:
			nhrnn=nh;
		self.embed=nn.Embedding(nwords+2,nh,padding_idx=0); #Pad 0, UNK, hence +2
		self.encoder=nn.LSTM(nh,nhrnn,nlayers,batch_first=True);
		self.nh=nh;
		self.nhrnn=nhrnn;
		self.dummy_h=nn.Parameter(torch.zeros(nlayers,1,nhrnn));
		self.dummy_c=nn.Parameter(torch.zeros(nlayers,1,nhrnn));
	#
	def sort(self,sent):
		#self.encoder.flatten_parameters()
		l=sent.data.gt(0).long().sum(1);
		_,sort_ind=l.sort(0,True);
		_,sort_ind_inv=sort_ind.sort(0);
		n=l.gt(0).long().sum();
		if n>0:
			sort_ind=sort_ind[:n].clone();
			sent_sorted=sent[sort_ind,:];
			return sent_sorted,sort_ind,sort_ind_inv;
		else:
			return None,None,None;
	#
	def forward(self,sent):
		#sent: batch x max_length tensor. 0=nothing, >0=word. Left aligned
		#embed_sent: batch x (2 x nlayers x nhrnn) tensor.
		sent_sorted,sort_ind,sort_ind_inv=self.sort(sent);
		if sort_ind is None:
			dummy_h=self.dummy_h.repeat(1,sent.size(0),1)*0;
			dummy_c=self.dummy_c.repeat(1,sent.size(0),1)*0;
			embed_sent=torch.cat((dummy_c,dummy_h),0);
			return embed_sent.permute(1,0,2).clone().view(sent.size(0),-1);
		#
		l=sent_sorted.gt(0).long().sum(1).data.tolist();
		embed_w=self.embed(sent_sorted); #batch x max_length x nh
		dummy_h=self.dummy_h.repeat(1,sent_sorted.size(0),1)*0;
		dummy_c=self.dummy_c.repeat(1,sent_sorted.size(0),1)*0;
		packed=torch.nn.utils.rnn.pack_padded_sequence(embed_w,l,batch_first=True);
		_,hidden=self.encoder(packed,(dummy_c,dummy_h));
		embed_sent=torch.cat((hidden[0],hidden[1]),0);
		if sent_sorted.size(0)<sent.size(0):
			dummy_h_0=self.dummy_h.repeat(1,sent.size(0)-sent_sorted.size(0),1)*0;
			dummy_c_0=self.dummy_c.repeat(1,sent.size(0)-sent_sorted.size(0),1)*0;
			embed_sent_0=torch.cat((dummy_c_0,dummy_h_0),0);
			embed_sent=torch.cat((embed_sent,embed_sent_0),1);
		embed_sent=embed_sent[:,sort_ind_inv,:];
		embed_sent=embed_sent.permute(1,0,2).clone().view(sent.size(0),-1);
		return embed_sent;

class masked_joint_encoder(nn.Module):
	def __init__(self,njoint,dof,nhjoint,nh):
		super(masked_joint_encoder,self).__init__()
		self.embed=nn.Linear(dof,nhjoint);
		self.conv1=nn.Conv1d(nhjoint*njoint,nh,1);
		self.njoint=njoint;
		self.dof=dof;
	#
	def forward(self,joint,mask):
		batch=joint.size(0);
		lvid=joint.size(1);
		x=self.embed(joint.view(-1,self.dof)).view(batch,lvid,self.njoint,-1)*mask.view(batch,lvid,self.njoint,1);
		x=x.view(batch,lvid,-1).permute(0,2,1);
		x=self.conv1(x);
		return x;

class skeleton_encoder_3d(nn.Module):
	def __init__(self,dof,nh):
		super(skeleton_encoder_3d,self).__init__()
		self.conv1=nn.Conv1d(dof,nh,1);
	#
	def forward(self,video):
		x=video.permute(0,2,1);
		x=self.conv1(x);
		return x;

class skeleton_decoder_3d(nn.Module):
	def __init__(self,dof,nh):
		super(skeleton_decoder_3d,self).__init__()
		self.conv1=nn.Conv1d(nh,dof,1);
	#
	def forward(self,h):
		x=self.conv1(h);
		x=x.permute(0,2,1);
		return x;