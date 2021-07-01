import torch
from flask import Flask, request, send_from_directory
from flask import Response
import jsonrpcserver
from jsonrpcserver import methods
import json
from flask_cors import CORS
import time
import datetime
import math
import random
import os
import sys
import requests
import numpy as np
import torchvision
import torchvision.models
import torchvision.utils
import torchvision.datasets.folder
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
from io import BytesIO
import argparse
import string
import random 


#Command line options
parser=argparse.ArgumentParser(description='')
#	Model
parser.add_argument('--port', type=int, default=5001)
params=parser.parse_args();
params.argv=sys.argv;


app = Flask(__name__, static_url_path='')
CORS(app)

#Prepare vqa models
models=[{'root':'res/models/simple_bert_7x7_4/'},{'root':'res/models/0000036/'},{'root':'res/models/0000038/'},{'root':'res/models/0000040/'},{'root':'res/models/0000044/'}];
robot_names=['Robot X', 'Robot A', 'Robot O', 'Robot C', 'Robot N'];
import xvqa2 as xvqa
m=xvqa.xvqa(models);

#Prepare inpainting model
import lib_fido_v2 as lib_fido
fido=lib_fido.FIDO(inpainter='CAInpainter',batch=1);

def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


import lru_cache
lru_image=lru_cache.new(300);
lru_vqa=lru_cache.new(300);

p_use_cache=0.5;
def load_image(imurl):
    if imurl in lru_image and float(torch.rand(1))<p_use_cache:
        image=lru_image[imurl].copy();
    else:
        response=requests.get(imurl);
        image=Image.open(BytesIO(response.content));
        lru_image[imurl]=image;
        image=image.copy();
    
    return image;
    


@methods.add
def remove_box(imurl,misc):
    image=load_image(imurl);
    x=misc[0];
    y=misc[1];
    w=misc[2];
    h=misc[3];
    imout=fido.batch_remove_box([image],[x],[y],[w],[h])[0];
    id=random_string(12);
    fname='counterfactual/%s.jpg'%(id);
    torchvision.utils.save_image(imout,fname);
    return {'imurl':fname};

@methods.add
def remove_background(imurl,misc):
    image=load_image(imurl);
    
    x=misc[0];
    y=misc[1];
    w=misc[2];
    h=misc[3];
    imout=fido.batch_remove_box_reverse([image],[x],[y],[w],[h])[0];
    id=random_string(12);
    fname='counterfactual/%s.jpg'%(id);
    torchvision.utils.save_image(imout,fname);
    return {'imurl':fname};

@methods.add
def black_and_white(imurl,misc=None):
    image=load_image(imurl);
    
    image=Ft.to_tensor(image);
    imavg=image.mean(0,keepdim=True);
    imavg=imavg.repeat(3,1,1);
    id=random_string(12);
    fname='counterfactual/%s.jpg'%(id);
    torchvision.utils.save_image(imavg,fname);
    return {'imurl':fname};

@methods.add
def zoom_in(imurl,misc):
    image=load_image(imurl);
    
    x=misc[0];
    y=misc[1];
    w=misc[2];
    h=misc[3];
    image=Ft.to_tensor(image);
    x0=int(misc[0]*image.shape[2]);
    y0=int(misc[1]*image.shape[1]);
    x1=int((misc[0]+misc[2])*image.shape[2]);
    y1=int((misc[1]+misc[3])*image.shape[1]);
    x0=min(x0,image.shape[2]-2);
    y0=min(y0,image.shape[1]-2);
    x1=max(x1,x0+1);
    y1=max(y1,y0+1);
    
    imcrop=image[:,y0:y1,x0:x1];
    id=random_string(12);
    fname='counterfactual/%s.jpg'%(id);
    torchvision.utils.save_image(imcrop,fname);
    return {'imurl':fname};

@methods.add
def vqa(imurl,question):
    if not (imurl,question) in lru_vqa:
        image=load_image(imurl);
        
        data=[];
        for x in range(len(models)):
            table_result=m.vqa([Ft.to_tensor(image)],[question],x);
            table_result['id']=[random_string(12)];
            data.append(table_result);
        
        lru_vqa[(imurl,question)]=data;
    else:
        data=lru_vqa[(imurl,question)];
    
    answers=[d['A'][0] for d in data];
    return {'answers':answers,'question':question}

@methods.add
def average_attn(imurl,question):
    if not (imurl,question) in lru_vqa:
        _=vqa(imurl,question);
    
    data=lru_vqa[(imurl,question)];
    spatial_attn=[];#list of objects, average, by_token, by_7x7_zone, tokens, scores, url_template
    object_attn=[]; #list of objects, average, by_token, by_box, tokens, boxes, scores, url_template
    topk_answers=[]; #list of objects, answer, confidence 
    
    for i in range(len(models)):
        #Spatial average attention
        if not 'spatial_attn_average' in data[i].d.keys():
            fname=m.explain_attention_map_average(data[i]);
            data[i]['spatial_attn_average']=[fname];
        else:
            fname=data[i]['spatial_attn_average'][0];
        
        #Spatial average attention
        if not 'object_attn_average' in data[i].d.keys():
            fname=m.explain_object_attention_average(data[i]);
            data[i]['object_attn_average']=[fname];
        else:
            fname=data[i]['object_attn_average'][0];
        
        qtokens=data[i]['qtoken'][0];
        scores=data[i]['attention'][0].tolist();
        spatial_attn_avg=data[i]['spatial_attn_average'][0];
        object_attn_avg=data[i]['object_attn_average'][0];
        
        
        spatial_attn.append({'average':spatial_attn_avg,'tokens':qtokens,'scores':[],'by_token':[],'by_7x7_zone':[]});
        object_attn.append({'average':object_attn_avg,'tokens':qtokens,'boxes':[],'scores':[],'by_token':[],'by_box':[]});
        #Top k answers
        topk_answers.append(m.explain_top_answers(data[i])[0]);
    
    
    return {'spatial_attn':spatial_attn,'object_attn':object_attn,'topk_answers':topk_answers,'related_qas':[]};

@methods.add
def explain(imurl,question):
    if not (imurl,question) in lru_vqa:
        _=vqa(imurl,question);
    
    data=lru_vqa[(imurl,question)];
    spatial_attn=[];#list of objects, average, by_token, by_7x7_zone, tokens, scores, url_template
    object_attn=[]; #list of objects, average, by_token, by_box, tokens, boxes, scores, url_template
    errorcam=[]; #list of objects, average
    topk_answers=[]; #list of objects, answer, confidence 
    related_qas=[]; #list of objects, question, answer
    
    for i in range(len(models)):
        #Spatial average attention
        if not 'spatial_attn_average' in data[i].d.keys():
            fname=m.explain_attention_map_average(data[i]);
            data[i]['spatial_attn_average']=[fname];
        else:
            fname=data[i]['spatial_attn_average'][0];
        
        #Object average attention
        if not 'object_attn_average' in data[i].d.keys():
            fname=m.explain_object_attention_average(data[i]);
            data[i]['object_attn_average']=[fname];
        else:
            fname=data[i]['object_attn_average'][0];
        
        #Errorcam
        if not 'errorcam' in data[i].d.keys():
            fname=m.explain_errormap(data[0]);
            data[i]['errorcam']=[fname];
        else:
            fname=data[i]['errorcam'][0];
        
        qtokens=data[i]['qtoken'][0];
        scores=data[i]['attention'][0].tolist();
        spatial_attn_avg=data[i]['spatial_attn_average'][0];
        object_attn_avg=data[i]['object_attn_average'][0];
        errorcam_fname=data[i]['errorcam'][0];
        
        
        spatial_fnames=m.explain_attention_map_all(data[i]);
        object_fnames=m.explain_object_attention_all(data[i]);
        
        spatial_attn.append({'average':spatial_attn_avg,'tokens':qtokens,'scores':[],'by_token':spatial_fnames,'by_7x7_zone':[]});
        errorcam.append({'average':errorcam_fname})
        object_attn.append({'average':object_attn_avg,'tokens':qtokens,'boxes':[],'scores':[],'by_token':object_fnames,'by_box':[]});
        #Top k answers
        topk_answers.append(m.explain_top_answers(data[i])[0]);
        #Related QAs
        related_qas.append(m.explain_related_qas(data[i])[0]);
    
    
    return {'spatial_attn':spatial_attn,'object_attn':object_attn,'topk_answers':topk_answers,'related_qas':related_qas,'errorcam':errorcam};

@methods.add
def explain_full(imurl,question):
    #One attention map per word
    #One attention map per box
    if not (imurl,question) in lru_vqa:
        _=vqa(imurl,question);
    
    data=lru_vqa[(imurl,question)];
    #for i in range(len(models)):
    #    #Spatial average attention
    #    spatial_fnames=m.explain_attention_map_all(data[i]);
    #    object_fnames=m.explain_object_attention_all(data[i]);
    
    return {};

@methods.add
def list_ims():
    ims=os.listdir('./val/');
    ims=[os.path.join('val',x) for x in ims];
    return {'ims':ims};

@methods.add
def list_models():
    return {'models':robot_names};

#Serving files
@app.route('/', methods=['GET','POST'])
def root():
	return send_from_directory('./','index.html')

@app.route('/handbook/', methods=['GET','POST'])
def hb():
	return send_from_directory('./','handbook.html')

@app.route('/handbook/res/<path:path>')
def send_hbres(path):
    return send_from_directory('./res/', path)

@app.route('/res/<path:path>')
def send_res(path):
    return send_from_directory('./res/', path)



@app.route('/val/<path:path>')
def send_png(path):
    return send_from_directory('./val/', path)

@app.route('/attn/<path:path>')
def send_png_attn(path):
    return send_from_directory('./attn/', path)


@app.route('/counterfactual/<path:path>')
def send_png_counterfactual(path):
    return send_from_directory('./counterfactual/', path)



#Serving functions
@app.route('/api/', methods=['POST'])
def api():
	req = request.get_data().decode()
	response = jsonrpcserver.dispatch(req)
	return Response(str(response), response.http_status,mimetype='application/json')



if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=True, port=params.port);
