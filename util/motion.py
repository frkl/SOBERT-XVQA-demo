import torch
import math
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F
import json

#Just assmue XYZ. Things will work out just fine...
def euler_to_quaternion(rx,ry,rz):
	#Assuming x->y->y rotation order
	rx=rx*(0.5/180*3.1415926); #theta/2
	ry=ry*(0.5/180*3.1415926);
	rz=rz*(0.5/180*3.1415926);
	cos_rx=rx.cos();
	sin_rx=rx.sin();
	cos_ry=ry.cos();
	sin_ry=ry.sin();
	cos_rz=rz.cos();
	sin_rz=rz.sin();
	return [cos_rx*cos_ry*cos_rz+sin_rx*sin_ry*sin_rz,sin_rx*cos_ry*cos_rz-cos_rx*sin_ry*sin_rz,cos_rx*sin_ry*cos_rz+sin_rx*cos_ry*sin_rz,cos_rx*cos_ry*sin_rz-sin_rx*sin_ry*cos_rz];

def quaternion_to_euler(s,i,j,k):
	#Assuming x->y->y rotation order
	rx=(2*(s*i+j*k)/(1-2*(i.pow(2)+j.pow(2)))).atan()/(3.1415926/180);
	ry=(2*(s*j-i*k)).asin()/(3.1415926/180);
	rz=(2*(s*k+i*j)/(1-2*(j.pow(2)+k.pow(2)))).atan()/(3.1415926/180);
	return rx,ry,rz;

#data: batch x 3, assuming zxy due to bvh special formatting.
def euler_to_axis_angle(x,y,z,order='xyz',cuda=True): #order means in which order transformations are applied
	if cuda:
		x=x.cuda();
		y=y.cuda();
		z=z.cuda();
	batch=x.size(0);
	Rx=torch.zeros(batch,3,3);
	Ry=torch.zeros(batch,3,3);
	Rz=torch.zeros(batch,3,3);
	if cuda:
		Rx=Rx.cuda();
		Ry=Ry.cuda();
		Rz=Rz.cuda();
	Rx[:,0,0]=1;
	Rx[:,1,1]=torch.cos(x/180*3.1415927);
	Rx[:,2,2]=torch.cos(x/180*3.1415927);
	Rx[:,2,1]=torch.sin(x/180*3.1415927);
	Rx[:,1,2]=-torch.sin(x/180*3.1415927);
	Ry[:,1,1]=1;
	Ry[:,0,0]=torch.cos(y/180*3.1415927);
	Ry[:,2,2]=torch.cos(y/180*3.1415927);
	Ry[:,0,2]=torch.sin(y/180*3.1415927);
	Ry[:,2,0]=-torch.sin(y/180*3.1415927);
	Rz[:,2,2]=1;
	Rz[:,0,0]=torch.cos(z/180*3.1415927);
	Rz[:,1,1]=torch.cos(z/180*3.1415927);
	Rz[:,1,0]=torch.sin(z/180*3.1415927);
	Rz[:,0,1]=-torch.sin(z/180*3.1415927);
	R=torch.bmm(Rz,torch.bmm(Ry,Rx));
	if cuda:
		R=R.cpu();
	angle=(R[:,0,0]+R[:,1,1]+R[:,2,2]-1)/2;
	angle=torch.acos(angle.view(-1,1).clamp(min=-1,max=1));
	u=torch.zeros(batch,3);
	u[:,0]=R[:,2,1]-R[:,1,2];
	u[:,1]=R[:,0,2]-R[:,2,0];
	u[:,2]=R[:,1,0]-R[:,0,1];
	u=u/2;
	u_degen=torch.zeros(batch,3);
	u_degen[:,0]=R[:,0,0]+1;
	u_degen[:,1]=R[:,1,1]+1;
	u_degen[:,2]=R[:,2,2]+1;
	u_degen=torch.sqrt(torch.clamp(u_degen/2,min=0));
	sign_y=R[:,0,1].gt(1e-6).float()*2-1;
	sign_z=(R[:,0,2].gt(1e-6).float()+(R[:,0,2].gt(-1e-6)*R[:,1,2].lt(-1e-6)).float()).ge(1).float()*2-1;
	u_degen[:,1]=u_degen[:,1]*sign_y;
	u_degen[:,2]=u_degen[:,2]*sign_z;
	axis=(u+1e-9*0.57735026919)/(u.norm(2,dim=1,keepdim=True)+1e-9);
	axis_degen=(u_degen+1e-9*0.57735026919)/(u_degen.norm(2,dim=1,keepdim=True)+1e-9);
	degen=angle.gt(3.14).float().repeat(1,3);
	axis=axis*(1-degen)+axis_degen*degen;
	return angle.repeat(1,3)*axis;

def axis_angle_to_euler(u,order='xyz'):
	#axis-angle to rotation matrix
	batch=u.size(0);
	angle=u.norm(2,dim=1).clamp(max=3.141592653);
	sin=torch.sin(angle);
	cos=torch.cos(angle);
	axis=(u+1e-9*0.57735026919)/(u.norm(2,dim=1,keepdim=True)+1e-9);
	ux=torch.zeros(batch,3,3);
	ux[:,0,1]=-axis[:,2];
	ux[:,0,2]=axis[:,1];
	ux[:,1,0]=axis[:,2];
	ux[:,1,2]=-axis[:,0];
	ux[:,2,0]=-axis[:,1];
	ux[:,2,1]=axis[:,0];
	R=torch.eye(3,3).view(1,3,3).repeat(batch,1,1);
	R=R+sin.view(batch,1,1).repeat(1,3,3)*ux+(1-cos.view(batch,1,1).repeat(1,3,3))*torch.bmm(ux,ux);
	#rotation matrix to euler angles
	rx=torch.zeros(batch);
	ry=torch.zeros(batch);
	rz=torch.zeros(batch);
	if order=='xyz':
		for i in range(0,batch):
			if -R[i,2,0]>=1-1e-3:  #pi/2 Add a small dead zone. Quite pointless decoding such a small angle
				ry[i:i+1]=3.1415926/2;
				rx[i:i+1]=torch.atan2(R[i:i+1,0,1],R[i:i+1,0,2])[0];
				rz[i:i+1]=0;
			elif -R[i,2,0]<=-1+1e-3: #-pi/2
				ry[i:i+1]=-3.1415926/2;
				rx[i:i+1]=torch.atan2(-R[i:i+1,0,1],-R[i:i+1,0,2])[0];
				rz[i:i+1]=0;
			else:
				ry[i:i+1]=torch.asin(-R[i:i+1,2,0]);
				rycos=torch.cos(ry[i:i+1]);
				rx[i:i+1]=torch.atan2(R[i:i+1,2,1]/rycos,R[i:i+1,2,2]/rycos);
				rz[i:i+1]=torch.atan2(R[i:i+1,1,0]/rycos,R[i:i+1,0,0]/rycos);
	rx=rx/3.14159265358*180;
	ry=ry/3.14159265358*180;
	rz=rz/3.14159265358*180;
	return rx,ry,rz;

def encode_quaternion_bvh(x):
	#Translation, don't touch
	out=[x[:,0:3]];
	for i in range(0,int((x.size(1)-3)/3)):
		#zxy order?
		rz=x[:,3*i+3:3*i+4];
		rx=x[:,3*i+4:3*i+5];
		ry=x[:,3*i+5:3*i+6];
		out+=euler_to_quaternion(rx,ry,rz);
	out=torch.cat(out,1);
	return out;

def decode_quaternion_bvh(x):
	#Translation, don't touch
	out=[x[:,0:3]];
	for ind in range(0,int((x.size(1)-3)/4)):
		q=x[:,4*ind+3:4*ind+7];
		q=q/q.norm(2,dim=1,keepdim=True);
		s=q[:,0:1];
		i=q[:,1:2];
		j=q[:,2:3];
		k=q[:,3:4];
		rx,ry,rz=quaternion_to_euler(s,i,j,k);
		out+=[rz,rx,ry];
	out=torch.cat(out,1);
	return out

def encode_axis_angle_bvh(x,cuda=True):
	#Translation, don't touch
	out=[x[:,0:3]];
	for i in range(3,x.size(1),3):
		out+=[euler_to_axis_angle(x[:,i+2],x[:,i+1],x[:,i],cuda=cuda)];
	out=torch.cat(out,1);
	return out;

def decode_axis_angle_bvh(data):
	#Translation, don't touch
	out=[data[:,0:3]];
	for i in range(3,data.size(1),3):
		x,y,z=axis_angle_to_euler(data[:,i:i+3]);
		out+=[z.view(-1,1),y.view(-1,1),x.view(-1,1)];
	out=torch.cat(out,1);
	return out

