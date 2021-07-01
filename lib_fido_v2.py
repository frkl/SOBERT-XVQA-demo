
#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
import math
import time
import argparse
import sys
sys.path.append('./FIDO-saliency/exp/vbd_imagenet')
import re
import importlib
from collections import namedtuple

import torchvision.models
import torchvision.datasets.folder
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from PIL import Image, ImageOps, ImageEnhance

import os
import sys
import argparse
import json
import lru_cache

class FIDO:
    def __init__(self,classifier='resnet50',inpainter='CAInpainter',batch=8):
        #Load ImageNet class names
        f=open('class_names.json');
        class_names=json.load(f);
        f.close();
        self.class_names=[class_names[str(i)][1] for i in range(1000)];
        
        #Load ResNet model
        #model=getattr(torchvision.models,classifier)
        #self.classifier=model(pretrained=True)
        #self.classifier.eval()
        #self.classifier.cuda()
        
        #Load inpainting model
        sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
        sys.path.append(os.path.join(os.getcwd(),'generative_inpainting'))
        import utils_model
        self.batch=batch;
        self.inpaint_model = utils_model.get_impant_model(inpainter,batch,cuda_enabled=False);
        self.inpaint_model.eval();
        self.inpaint_model#.cuda();
        
        self.lru_im_in=lru_cache.new(10);
        return;
    
    
    def denormalize(self,im,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
        mean=torch.Tensor(mean).view(3,1,1);
        std=torch.Tensor(std).view(3,1,1);
        im=im*std+mean
        return im;
    
    
    def infill(self,image,mask):
        background=self.inpaint_model.generate_background(image.view(1,3,224,224).repeat(self.batch,1,1,1).cuda(),mask.view(1,3,224,224).repeat(self.batch,1,1,1).cuda());
        return background[0].cpu().data.clone();
    
    def infill_batch(self,image,mask):
        image=image#.cuda();
        mask=mask#.cuda();
        background=self.inpaint_model.generate_background(image,mask);
        return background.cpu().data.clone();
    
    def sanitize_box(self,x,y,w,h,imw,imh):
        x0=round(x*imw);
        y0=round(y*imh);
        x1=round((x+w)*imw);
        y1=round((y+h)*imh);
        #
        x0=min(max(x0,0),imw-1);
        x1=min(max(x1,0),imw);
        y0=min(max(y0,0),imh-1);
        y1=min(max(y1,0),imh);
        #
        if x0>x1:
            x0=x1;
        
        if y0>y1:
            y0=y1;
        
        return x0,y0,x1-x0,y1-y0;
    
    #handles 8-ish at a time
    def batch_remove_box(self,im_ins,xs,ys,ws,hs,overlay=False):
        assert(len(im_ins)==self.batch);
        im_fullres=[];
        im_fullres_infilled=[]
        im_224=[];
        mask_224=[];
        roi_fullres=[]
        roi_224=[];
        for id,im_in in enumerate(im_ins):
            im=im_in;
            
            #Full res ones
            imsz=im.size;
            x=xs[id];
            y=ys[id];
            w=ws[id];
            h=hs[id];
            x,y,w,h=self.sanitize_box(x,y,w,h,imsz[0],imsz[1]);
            
            im_fullres_i=Ft.normalize(Ft.to_tensor(im),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]);
            roi_fullres_i=[x,y,w,h];
            im_fullres.append(im_fullres_i);
            roi_fullres.append(roi_fullres_i);
            
            #224x224 ones
            x=xs[id];
            y=ys[id];
            w=ws[id];
            h=hs[id];
            x,y,w,h=self.sanitize_box(x,y,w,h,224,224);
            im_224_i=Ft.resize(im,(224,224));
            im_224_i=Ft.normalize(Ft.to_tensor(im_224_i),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]);
            roi_224_i=[x,y,w,h];
            im_224.append(im_224_i);
            roi_224.append(roi_224_i);
            
            #Produce mask for 224
            mask_224_i=im_224_i.clone().fill_(1);
            mask_224_i[:,y:y+h,x:x+w]=0;
            mask_224.append(mask_224_i);
        
        #Do infilling on 224
        im_224_infill=self.infill_batch(torch.stack(im_224,dim=0),torch.stack(mask_224,dim=0));
        
        #Copy and resize
        for id,_ in enumerate(im_ins):
            im_224_infill_i=im_224_infill[id];
            roi_224_i=roi_224[id];
            x=roi_224_i[0];
            y=roi_224_i[1];
            w=roi_224_i[2];
            h=roi_224_i[3];
            
            im_fullres_i=im_fullres[id];
            im_fullres_infilled_i=im_fullres_i.clone();
            roi_fullres_i=roi_fullres[id];
            x2=roi_fullres_i[0];
            y2=roi_fullres_i[1];
            w2=roi_fullres_i[2];
            h2=roi_fullres_i[3];
            
            #Copy, resize and paste
            im_infill=im_224_infill_i[:,y:y+h,x:x+w];
            im_infill=F.adaptive_avg_pool2d(im_infill,(h2,w2));
            im_fullres_infilled_i[:,y2:y2+h2,x2:x2+w2]=im_infill;
            im_fullres_infilled_i=self.denormalize(im_fullres_infilled_i)
            
            if overlay:
                im_fullres_infilled_i[0,y2:y2+1,x2:x2+w2]=1;
                im_fullres_infilled_i[1,y2:y2+1,x2:x2+w2]=0;
                im_fullres_infilled_i[2,y2:y2+1,x2:x2+w2]=0;
                
                im_fullres_infilled_i[0,y2+h2-1:y2+h2,x2:x2+w2]=1;
                im_fullres_infilled_i[1,y2+h2-1:y2+h2,x2:x2+w2]=0;
                im_fullres_infilled_i[2,y2+h2-1:y2+h2,x2:x2+w2]=0;
                
                im_fullres_infilled_i[0,y2:y2+h2,x2:x2+1]=1;
                im_fullres_infilled_i[1,y2:y2+h2,x2:x2+1]=0;
                im_fullres_infilled_i[2,y2:y2+h2,x2:x2+1]=0;
                
                im_fullres_infilled_i[0,y2:y2+h2,x2+w2-1:x2+w2]=1;
                im_fullres_infilled_i[1,y2:y2+h2,x2+w2-1:x2+w2]=0;
                im_fullres_infilled_i[2,y2:y2+h2,x2+w2-1:x2+w2]=0;
            
            im_fullres_infilled.append(im_fullres_infilled_i);
        
        return im_fullres_infilled;
    
    #handles 8-ish at a time
    def batch_remove_box_reverse(self,im_ins,xs,ys,ws,hs,overlay=False):
        assert(len(im_ins)==self.batch);
        im_fullres=[];
        im_fullres_infilled=[]
        im_224=[];
        mask_224=[];
        roi_fullres=[]
        roi_224=[];
        for id,im_in in enumerate(im_ins):
            im=im_in;
            #Full res ones
            imsz=im.size;
            x=xs[id];
            y=ys[id];
            w=ws[id];
            h=hs[id];
            x,y,w,h=self.sanitize_box(x,y,w,h,imsz[0],imsz[1]);
            
            im_fullres_i=Ft.normalize(Ft.to_tensor(im),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]);
            roi_fullres_i=[x,y,w,h];
            im_fullres.append(im_fullres_i);
            roi_fullres.append(roi_fullres_i);
            
            #224x224 ones
            x=xs[id];
            y=ys[id];
            w=ws[id];
            h=hs[id];
            x,y,w,h=self.sanitize_box(x,y,w,h,224,224);
            im_224_i=Ft.resize(im,(224,224));
            im_224_i=Ft.normalize(Ft.to_tensor(im_224_i),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]);
            roi_224_i=[x,y,w,h];
            im_224.append(im_224_i);
            roi_224.append(roi_224_i);
            
            #Produce mask for 224
            mask_224_i=im_224_i.clone().fill_(0);
            mask_224_i[:,y:y+h,x:x+w]=1;
            mask_224.append(mask_224_i);
        
        #Do infilling on 224
        im_224_infill=self.infill_batch(torch.stack(im_224,dim=0),torch.stack(mask_224,dim=0));
        
        #Copy and resize
        for id,_ in enumerate(im_ins):
            im_224_infill_i=im_224_infill[id];
            
            im_fullres_i=im_fullres[id];
            im_fullres_infilled_i=im_fullres_i.clone();
            roi_fullres_i=roi_fullres[id];
            x2=roi_fullres_i[0];
            y2=roi_fullres_i[1];
            w2=roi_fullres_i[2];
            h2=roi_fullres_i[3];
            
            imh=im_fullres_i.shape[1];
            imw=im_fullres_i.shape[2];
            
            #Copy, resize and paste
            im_fullres_infilled_i=F.adaptive_avg_pool2d(im_224_infill_i,(imh,imw));
            im_fullres_infilled_i[:,y2:y2+h2,x2:x2+w2]=im_fullres_i[:,y2:y2+h2,x2:x2+w2];
            im_fullres_infilled_i=self.denormalize(im_fullres_infilled_i)
            
            if overlay:
                im_fullres_infilled_i[0,y2:y2+1,x2:x2+w2]=1;
                im_fullres_infilled_i[1,y2:y2+1,x2:x2+w2]=0;
                im_fullres_infilled_i[2,y2:y2+1,x2:x2+w2]=0;
                
                im_fullres_infilled_i[0,y2+h2-1:y2+h2,x2:x2+w2]=1;
                im_fullres_infilled_i[1,y2+h2-1:y2+h2,x2:x2+w2]=0;
                im_fullres_infilled_i[2,y2+h2-1:y2+h2,x2:x2+w2]=0;
                
                im_fullres_infilled_i[0,y2:y2+h2,x2:x2+1]=1;
                im_fullres_infilled_i[1,y2:y2+h2,x2:x2+1]=0;
                im_fullres_infilled_i[2,y2:y2+h2,x2:x2+1]=0;
                
                im_fullres_infilled_i[0,y2:y2+h2,x2+w2-1:x2+w2]=1;
                im_fullres_infilled_i[1,y2:y2+h2,x2+w2-1:x2+w2]=0;
                im_fullres_infilled_i[2,y2:y2+h2,x2+w2-1:x2+w2]=0;
            
            im_fullres_infilled.append(im_fullres_infilled_i);
        
        return im_fullres_infilled;
