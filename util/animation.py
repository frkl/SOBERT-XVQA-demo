
#require conda install -c menpo ffmpeg for it to work

import torch
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy

import sys

#Write a bvh file used by the animation software
def write_bvh(fname,animation,meta,fps=120):
	N=animation.size(0);
	dof=animation.size(1);
	f=open(fname,'w');
	f.write(meta);
	f.write('MOTION\n');
	f.write('Frames: %d\n'%N);
	f.write('Frame Time: %f\n'%(1.0/fps));
	for i in range(0,N):
		for j in range(0,dof):
			f.write('%f'%animation[i,j]);
			if j<dof-1:
				f.write(' ');
		f.write('\n');
	#
	f.close();
