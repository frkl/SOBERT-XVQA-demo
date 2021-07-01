import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.datasets.folder
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from pytorch_transformers import BertTokenizer
import os
import db
from PIL import Image
import cv2
import numpy
import time
import copy
import math

import sys
sys.path.insert(0, './bottom-up-attention/')
sys.path.insert(0, './bottom-up-attention/caffe/python/')
sys.path.insert(0, './bottom-up-attention/lib/')
sys.path.insert(0, './bottom-up-attention/tools/')
sys.path.append('./errorcam')

import caffe
caffe.set_mode_gpu()


from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import cv2

cfg_from_file('bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml')
weights = 'bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
self_fast_rcnn = caffe.Net(prototxt, caffe.TEST, weights=weights);

import errorcam.models.attention_refine.atten_refine_network as att_refine
from errorcam.scripts.pytorchgradcam.gradcam import GradCam
from scipy.stats import spearmanr as correlation_func_atten
from statsmodels.stats.weightstats import ztest
import numpy as np
import json

#t0=time.time();
#im_file = 'val/n01532829_2439.JPEG'
# Similar to get_detections_from_im

#import requests
#response=requests.get('http://diva-1:5001/val/n01532829_2439.JPEG');
#image=Image.open(BytesIO(response.content));
#image=image.copy();
#im=F.to_tensor(image);
#im=(im*255).permute(1,2,0);
#im=torch.stack((im[:,:,2],im[:,:,1],im[:,:,0]),dim=2);
#im=im.cpu();
#im=im.numpy();
#im = cv2.imread(im_file)
#scores, boxes, attr_scores, rel_scores = im_detect(net, im)
#print('Loaded %f'%(time.time()-t0));
#a=0/0;


#QA classifier
import qa_classifier as qa_classifier
qa_classifier=qa_classifier.qa_classifier;
qtypes=['object', 'color', 'action', 'count', 'time', 'weather']

import model_7x7 as base_model

import lru_cache
import time
lru_mask_rcnn=lru_cache.new(100);

class xvqa:
    def __init__(self,args_models):
        self.in_use=0;
        #Prepare ResNet152 for feature extraction
        with torch.no_grad():
            resnet152=torchvision.models.resnet152(pretrained=True)
            resnet152=nn.Sequential(*list(resnet152.children())[:-2]).cuda();
            resnet152=nn.DataParallel(resnet152).cuda()
            resnet152.eval();
            self.resnet152=resnet152;
        
        #Prepare BERT tokenizer for question
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased');
        self.tokenizer.max_qlength=30;
        
        #Prepare several BERT-VQA models for QA
        print('Loading model')
        models=[];
        qfvs=[];
        for m in args_models:
            args_m=torch.load(os.path.join(m['root'],'args.pt'));
            model=base_model.simple_vqa_model(args_m).cuda();
            model=nn.DataParallel(model).cuda()
            checkpoint=torch.load(os.path.join(m['root'],'model_checkpoint.pt'));
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            model.answer_dictionary=torch.load(os.path.join(m['root'],'answer_dictionary.pt'));
            model.args=args_m;
            models.append(model);
            
            qfv=torch.load(os.path.join(m['root'],'qfv.pt'))
            qfvs.append(qfv);
        
        self.models=models;
        self.qfvs=qfvs;
        
        self.qfvs_imkey=torch.load('res/models/qfv_imkey.pt');
        
        #Prepare fast-rcnn detector
        #cfg_from_file('bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml')
        #weights = 'bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
        #prototxt = 'bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
        #self.fast_rcnn = caffe.Net(prototxt, caffe.TEST, weights=weights);
        
        def loadGloveModel(gloveFile):
            print("Loading Glove Model")
            f = open(gloveFile,'r', encoding='utf8')
            model = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            print("Done.",len(model)," words loaded!")
            return model
        
        
        #Get w2v
        self.w2v = loadGloveModel("errorcam/glove.6B.300d.txt");
        
        atten_dim = (4,12,115,115)
        model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129, "ques_cam":False}
        self.attention_refine_model = att_refine.uncertainatt_refinedatt_net_cam_bigger(**model_init_args).cuda()
        model_suffix = "model_3_5501.pt"
        exp_name = "exp4_fullmodel_corrpred_refinedattn_uncertainCAM_bigger"
        self.attention_refine_model.load_state_dict(torch.load("errorcam/checkpoints/"+exp_name+"/"+model_suffix))
        self.gradcam = GradCam(self.attention_refine_model)
        
        return;
    
    def get_lock(self):
        while self.in_use>0:
            time.sleep(0.2);
            print('locked');
        
        self.in_use=1;
        return;
    
    def release_lock(self):
        self.in_use=0;
        return;
    
    def parse_question(self,qtext):
        if isinstance(qtext,list):
            qtokens=[];
            question=[];
            for qi in qtext:
                qtokens_i,question_i=self.parse_question(qi);
                qtokens.append(qtokens_i);
                question.append(question_i);
            
            with torch.no_grad():
                question=torch.stack(question,dim=0);
            
            return qtokens,question;
        else:
            qtokens=self.tokenizer.tokenize(qtext);
            if len(qtokens)>self.tokenizer.max_qlength-2:
                qtokens=qtokens[:self.tokenizer.max_qlength-2];
            
            qtokens=['[CLS]']+qtokens+['[SEP]'];
            question=self.tokenizer.convert_tokens_to_ids(qtokens);
            question=question+[0]*(self.tokenizer.max_qlength-len(question));
            question=torch.LongTensor(question);
            return qtokens,question;
    
    def get_7x7_features(self,Is):
        #Resize & Normalize
        with torch.no_grad():
            It=[]
            for I in Is:
                I=F.adaptive_avg_pool2d(I,(224,224));
                I=Ft.normalize(I,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]);
                It.append(I);
            
            It=torch.stack(It,dim=0);
            
            #Extract features
            fvs=[];
            batch=8;
            for i in range(0,len(It),batch):
                r=min(i+batch,len(It));
                fv=self.resnet152(It[i:r]);
                fvs.append(fv);
            
            fvs=torch.cat(fvs,dim=0);
        
        return fvs;
    
    def get_maskrcnn_features(self,Is):
        try:
            self.get_lock();
            caffe.set_mode_gpu()
            conf_thresh=0.2
            min_boxes=36
            max_boxes=36
            net=self_fast_rcnn;
            
            fv=[];
            boxes_=[];
            for iid in range(len(Is)):
                I=Is[iid]
                k=I.numpy().tostring();
                if k in lru_mask_rcnn:
                    fv_i=lru_mask_rcnn[k]['fv'].clone();
                    boxes_i=lru_mask_rcnn[k]['boxes'].clone();
                    fv.append(fv_i);
                    boxes_.append(boxes_i);
                else:
                    t0=time.time();
                    I=I.cuda();
                    im=(I*255).permute(1,2,0);
                    im=torch.stack((im[:,:,2],im[:,:,1],im[:,:,0]),dim=2);
                    im=im.cpu();
                    print(im.shape,im.max(),im.min())
                    im=im.numpy();
                    print('chpt1 %f'%float(time.time()-t0));
                    scores, boxes, attr_scores, rel_scores = im_detect(net, im)
                    print('chpt2 %f'%float(time.time()-t0));
                    
                    # Keep the original boxes, don't worry about the regression bbox outputs
                    rois = net.blobs['rois'].data.copy()
                    # unscale back to raw image space
                    blobs, im_scales = _get_blobs(im, None)
                    print('chpt3 %f'%float(time.time()-t0));
                    
                    cls_boxes = rois[:, 1:5] / im_scales[0]
                    cls_prob = net.blobs['cls_prob'].data
                    attr_prob = net.blobs['attr_prob'].data
                    pool5 = net.blobs['pool5_flat'].data
                    
                    # Keep only the best detections
                    max_conf = numpy.zeros((rois.shape[0]))
                    for cls_ind in range(1,cls_prob.shape[1]):
                        cls_scores = scores[:, cls_ind]
                        try:
                            dets = numpy.hstack((cls_boxes, cls_scores[:, numpy.newaxis])).astype(numpy.float32)
                        except:
                            print(cls_boxes.shape);
                            print(cls_scores.shape);
                            dets = numpy.hstack((cls_boxes, cls_scores[:, numpy.newaxis])).astype(numpy.float32)
                        
                        keep = numpy.array(nms(dets, cfg.TEST.NMS))
                        max_conf[keep] = numpy.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
                    
                    keep_boxes = numpy.where(max_conf >= conf_thresh)[0]
                    if len(keep_boxes) < min_boxes:
                        keep_boxes = numpy.argsort(max_conf)[::-1][:min_boxes]
                    elif len(keep_boxes) > max_boxes:
                        keep_boxes = numpy.argsort(max_conf)[::-1][:max_boxes]
                    
                    print('chpt4 %f'%float(time.time()-t0));
                    imh=I.shape[1];
                    imw=I.shape[2];
                    boxes_i=torch.from_numpy(cls_boxes[keep_boxes]).view(1,36,4);
                    boxes_i=boxes_i/torch.Tensor([imw,imh,imw,imh]).view(1,1,4);
                    fv_i=torch.from_numpy(pool5[keep_boxes]).view(1,36,2048);
                    print(fv_i.shape,boxes_i.shape);
                    lru_mask_rcnn[k]={'fv':fv_i.clone().cpu(),'boxes':boxes_i.clone().cpu()};
                    print('chpt5 %f'%float(time.time()-t0));
                    fv.append(fv_i);
                    boxes_.append(boxes_i);
            
            fv=torch.cat(fv,dim=0);
            boxes_=torch.cat(boxes_,dim=0);
            self.release_lock();
        except:
            self.release_lock();
            a=0/0;
            
        return fv,boxes_;
    
    def vqa(self,Is,Qs,use_model=''):
        qtokens,q=self.parse_question(Qs);
        print(qtokens)
        fv7x7=self.get_7x7_features(Is);
        fv36,boxes=self.get_maskrcnn_features(Is);
        with torch.no_grad():
            print(fv7x7.shape,fv36.shape,q.shape);
            scores,attn=self.models[use_model](fv36,fv7x7.permute(0,2,3,1),q);
            scores=scores.data.cpu();
            attn=torch.stack(attn,dim=1).data.cpu();
            top1_conf,pred=scores.max(dim=1);
            As=[self.models[use_model].answer_dictionary[i] for i in pred.tolist()];
        
        return db.Table({'I':Is,'Q':Qs,'A':As,'scores':scores,'attention':attn,'qtoken':qtokens,'qtensor':q,'features_7x7':fv7x7,'features_fv36':fv36,'bbox':boxes,'model':[use_model for q in Qs]});
    
    #attn: 7x7 matrix
    #imurl: image url
    #output_fname: fname wrt root
    def write_spatial_attention(self,I,attn,output_fname):
        eps=1e-4
        I=Ft.to_pil_image(I);
        I=I.resize((224, 224))
        I=numpy.asarray(I).astype(numpy.float32)
        attn=attn.view(7,7).numpy()
        attn=cv2.resize(attn, (224, 224))
        attn=(attn-numpy.min(attn)+eps)/(numpy.max(attn)-numpy.min(attn)+eps)
        att_heatmap=cv2.applyColorMap(numpy.uint8(255*attn), cv2.COLORMAP_JET)
        alpha = 0.5
        output_image=(1-alpha)*att_heatmap+alpha*I;
        cv2.imwrite(output_fname,output_image)
        return;
    
    def write_object_attention(self,I,attn_rpn,bbox,attn_fname,token_ind=-1):
        def apply_mask(image, mask, color, alpha=0.7):
            for c in range(3):
                image[:, :, c] = numpy.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          image[:, :, c])
            return image
        
        def apply_obj_mask(masked_image, mask, actual_image, weight):
            mask = numpy.repeat(mask[:,:,numpy.newaxis], 3, axis=2)
            obj_image = numpy.ones(actual_image.shape)*255
            
            numpy.copyto(obj_image, actual_image, where=(mask==1))
            
            white_image = numpy.ones(actual_image.shape)*255
            
            if weight< 0.3:
                weight=weight+0.15
            obj_img_weighted = weight*obj_image + (1-weight)*white_image
            
            numpy.copyto(masked_image, obj_img_weighted, where=(mask==1))
            
            return masked_image

        def computeIOU(box1, box2):
            #boxes should be in (y1, x1, y2, x2)
            box1 = numpy.asarray(box1).astype(numpy.float32)
            box2 = numpy.asarray(box2).astype(numpy.float32)
            
            iou_box_x1 = max(box1[1], box2[1])
            iou_box_y1 = max(box1[0], box2[0])
            iou_box_x2 = min(box1[3], box2[3])
            iou_box_y2 = min(box1[2], box2[2])
            
            iou_h = max(0, iou_box_y2-iou_box_y1)
            iou_w = max(0, iou_box_x2 - iou_box_x1)
            
            roi_area = (iou_h * iou_w)
            
            box1_area = numpy.absolute((box1[3] - box1[1]) * (box1[2] - box1[0]))
            box2_area = numpy.absolute((box2[3] - box2[1]) * (box2[2] - box2[0]))
            
            iou = roi_area/float(box1_area + box2_area - roi_area)
            
            return iou
        
        def compute_box_distance(box1, box2):
            #boxes in (y1, x1, y2, x2)
            box1 = numpy.asarray(box1).astype(numpy.float32)
            box2 = numpy.asarray(box2).astype(numpy.float32)
            
            cntr_box1_x = int((box1[1] + box1[3])/2)
            cntr_box1_y = int((box1[0] + box1[2])/2)
            
            cntr_box2_x = int((box2[1] + box2[3])/2)
            cntr_box2_y = int((box2[0] + box2[2])/2)
            
            dist = numpy.sqrt((cntr_box1_x - cntr_box2_x)**2 + (cntr_box1_y - cntr_box2_y)**2)
            
            return dist
        
        def computeWeights(mrcnn_boxes, rpn_boxes, box_weights):
            epsilon = 1e-5
            rcnn_box_weights = []
            for ind, rcnn_box in enumerate(mrcnn_boxes):
                max_area = 0
                all_iou = []
                all_weights = []
                for rpn_ind, rpn_box in enumerate(rpn_boxes):
                    iou_area = computeIOU(rcnn_box, rpn_box)
                    all_iou.append(iou_area)
                    all_weights.append(box_weights[rpn_ind])
                
                if len(all_iou) >= 1 and numpy.sum(all_iou)>0:
                    final_weight = numpy.exp(numpy.log(numpy.sum(numpy.exp(numpy.log(numpy.asarray(all_iou)) + numpy.log(numpy.asarray(all_weights))))) -(numpy.log(float(numpy.sum(all_iou)+ epsilon))))
                    rcnn_box_weights.append(final_weight)
                else:
                    rcnn_box_weights.append(0)
            
            return rcnn_box_weights
        
        def make_rpn_attention_im(actual_image,attention_rpn,bboxes,attn_fname,token_ind=-1):
            im_boxes=(bboxes.numpy()*256).astype(numpy.int32)
            final_obj_weights = attention_rpn.numpy()
            actual_image = Ft.to_pil_image(actual_image).resize((256,  256))
            
            if len(final_obj_weights) != 0:
                if numpy.max(final_obj_weights) > 0:
                    final_obj_weights = numpy.exp(numpy.log(final_obj_weights) - numpy.log(numpy.max(final_obj_weights)))
            

            img_arr = numpy.asarray(actual_image).astype(numpy.float32)
            masked_image = numpy.ones(img_arr.shape) * 255
            masked_image = img_arr * 0.1 + masked_image * 0.9
            
            if len(final_obj_weights) != 0:
                obj_atten_inds = numpy.argsort(final_obj_weights)
            else:
                obj_atten_inds = []
            obj_atten_inds = obj_atten_inds[::-1]
            top_N = 5  # int(N * float(3) / 4)
            for i in obj_atten_inds[:top_N][::-1]:
                if final_obj_weights[i] > 0:
                    mask = numpy.zeros((256,256))
                    x0, y0, x1, y1 = im_boxes[i]
                    mask[y0:y1, x0:x1]=1
                    masked_image=apply_obj_mask(masked_image,mask,img_arr,float(final_obj_weights[i]))
            
            ## draw origin box (clicked box and draw arrows from that box to attended boxes)
            ## will only work for cases where we have such box to box attention, think about generalizing this later
            if token_ind>29 and token_ind<66:
                origin_box = im_boxes[token_ind-30]
                ox0, oy0, ox1, oy1 = origin_box
                cv2.rectangle(masked_image,(origin_box[0],origin_box[1]),(origin_box[2],origin_box[3]),(100,100,100),5)
                for i in obj_atten_inds[:top_N]:
                    x0, y0, x1, y1 = im_boxes[i]
                    cv2.rectangle(masked_image, (x0, y0), (x1, y1), (50, 50, 50), 1)
                    pt1, pt2 = compute_closest_corner(origin_box, im_boxes[i])
                    cv2.arrowedLine(masked_image, pt1, pt2, (100,100,100), 2,8,0,0.05)
            
            #masked_im = Image.fromarray(masked_image.astype(numpy.float32))
            cv2.imwrite(attn_fname,masked_image[:,:,::-1])
            return;
        
        def compute_closest_corner(box1, box2):
            ax0, ay0, ax1, ay1 = box1
            bx0, by0, bx1, by1 = box2
            min_d = float("inf")
            for ax in [ax0, ax1]:
                for bx in [bx0, bx1]:
                    d = abs(ax-bx)
                    if d<min_d:
                        ax_c = ax
                        bx_c = bx
                        min_d = d
            
            min_d = float("inf")
            for ay in [ay0, ay1]:
                for by in [by0, by1]:
                    d = abs(ay-by)
                    if d<min_d:
                        ay_c = ay
                        by_c = by
                        min_d = d
            
            return (ax_c, ay_c), (bx_c, by_c)
        
        make_rpn_attention_im(I,attn_rpn,bbox,attn_fname,token_ind);
        return;
    
    def explain_errormap(self,table_vqa):
        key=table_vqa['id'][0];
        
        I=table_vqa['I'][0]
        Q=table_vqa['Q'][0]
        fv7x7=table_vqa['features_7x7'][0:1].clone()#.permute(0,2,3,1).view(1,49,2048);
        attn=table_vqa['attention'][0:1];
        answer_prob=F.softmax(table_vqa['scores'][0:1],dim=1);
        
        
        def get_avg_w2v(question, w2v):
            q_w = question.lower().split("?")[0].split(" ")
            avg_feats = []
            for w in q_w:
                if w in w2v:
                    avg_feats.append(w2v[w])
            
            return np.average(avg_feats, axis=0)
        
        def get_err_weight(p):
            weight = (p/0.175)**4  # empirically defined by what looks good on the matplotlib colormap.
            if weight>1:
                weight=1.0

            return weight
        
        #get question features 
        ques_feats = torch.from_numpy(get_avg_w2v(Q,self.w2v))
        ques_feats = ques_feats.cuda().float().unsqueeze(0)
        
        #get failure prediction probability. Using this to weigh the error maps results in better visualization. 
        model_out = self.attention_refine_model(attn.cuda().view(1,-1), fv7x7.cuda(), ques_feats, answer_prob.cuda());
        fail_pred = model_out['wrong_pred']
        fail_pred = float(fail_pred.squeeze().detach().cpu())
        weight = get_err_weight(fail_pred)
        
        print(attn.shape,fv7x7.shape,ques_feats.shape,answer_prob.shape)
        att_map, _ = self.gradcam([attn.cuda().view(1,-1), fv7x7.cuda(), ques_feats, answer_prob.cuda()])
        actual_image = Ft.to_pil_image(I).resize((224,224))
        actual_image=numpy.asarray(actual_image).astype(numpy.float32)
        processed_img = cv2.resize(actual_image, (224,224))
        att_map = att_map.reshape((7,7))
        att_map = cv2.resize(att_map, (224,224)) 
        
        epsilon = 1e-3
        att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
        alpha = 0.5
        output_image = (1 - alpha) * att_heatmap *weight + alpha * processed_img
        
        errmap_im_file_name='./attn/%s_errormap.jpg'%key;
        cv2.imwrite(errmap_im_file_name, output_image)
        return errmap_im_file_name;
    
    
    def explain_attention_map_average(self,table_vqa):
        key=table_vqa['id'][0];
        attn=table_vqa['attention'][0];
        qtoken=table_vqa['qtoken'][0];
        L=len(qtoken);
        attn_sp=attn[-1,:,:L, 66:].mean(0).mean(0).view(7,7);
        attn_fname='./attn/%s_spatial_average.jpg'%key;
        self.write_spatial_attention(table_vqa['I'][0],attn_sp,attn_fname);
        return attn_fname;
    
    def explain_attention_map_all(self,table_vqa):
        key=table_vqa['id'][0];
        attn=table_vqa['attention'][0];
        qtoken=table_vqa['qtoken'][0];
        L=len(qtoken);
        attn_fname=[];
        for i in range(L):
            attn_sp=attn[-1,:,i, 66:].mean(0).view(7,7);
            attn_fname_i='./attn/%s_spatial_w%d.jpg'%(key,i);
            self.write_spatial_attention(table_vqa['I'][0],attn_sp,attn_fname_i);
            attn_fname.append(attn_fname_i);
        
        return attn_fname;
    
    def explain_object_attention_average(self,table_vqa):
        key=table_vqa['id'][0];
        attn=table_vqa['attention'][0];
        bbox=table_vqa['bbox'][0];
        qtoken=table_vqa['qtoken'][0];
        L=len(qtoken);
        
        attn_rpn=attn[-1,-1,:L,30:66].mean(0);
        attn_fname='./attn/%s_object_average.jpg'%key;
        self.write_object_attention(table_vqa['I'][0],attn_rpn,bbox,attn_fname)
        return attn_fname;
    
    def explain_object_attention_all(self,table_vqa):
        key=table_vqa['id'][0];
        attn=table_vqa['attention'][0];
        bbox=table_vqa['bbox'][0];
        qtoken=table_vqa['qtoken'][0];
        L=len(qtoken);
        attn_fname=[];
        for i in range(L):
            attn_rpn=attn[-1,-1,i,30:66];
            attn_fname_i='./attn/%s_object_w%d.jpg'%(key,i);
            self.write_object_attention(table_vqa['I'][0],attn_rpn,bbox,attn_fname_i)
            attn_fname.append(attn_fname_i);
        
        return attn_fname;
    
    #def explain_attention_map_pairs(self,table_vqa):
    
    
    def explain_top_answers(self,table_vqa,k=5):
        n=len(table_vqa);
        topk_answers=[];
        topk_confidence=[];
        for i in range(n):
            use_model=table_vqa['model'][i];
            s=table_vqa['scores'][i];
            p=F.softmax(s,dim=0);
            p,ind=p.sort(dim=0,descending=True);
            p=p[:k].tolist();
            ind=ind[:k].tolist();
            a=[self.models[use_model].answer_dictionary[j] for j in ind];
            topk_answers_i=[];
            for j in range(len(a)):
                topk_answers_i.append({'answer':a[j],'confidence':p[j]});
            
            topk_answers.append(topk_answers_i);
        
        return topk_answers;
    
    def explain_related_qas(self,table_vqa,k=5):
        n=len(table_vqa);
        topk_qas=[];
        for i in range(n):
            #Compute vector for question
            use_model=table_vqa['model'][i];
            I=table_vqa['I'][i];
            qtext=table_vqa['Q'][i]
            q=self.question_vector_v0(qtext,batch=50,model=use_model);
            
            #Query related question
            precomputed_qfv=self.qfvs[use_model]['qfv'];
            precomputed_q=self.qfvs[use_model]['q'];
            s=torch.mm(precomputed_qfv,q.view(-1,1)).view(-1);
            s,ind=s.sort(dim=0,descending=True);
            ind=ind.tolist();
            s=s.tolist();
            
            #Read questions and call VQA
            topk_qas_i=[];
            for j in range(k):
                topk_qas_i.append({'question':precomputed_q[ind[j]],'r':s[j]});
            
            result=self.vqa([I]*k,[x['question'] for x in topk_qas_i],use_model=use_model);
            for j in range(k):
                topk_qas_i[j]['answer']=result['A'][j];
            
            topk_qas.append(topk_qas_i);
        
        #Call VQA in batch mode
        
        
        return topk_qas;
    
    #Question type as perceived by the model
    def explain_qtype(self,table_vqa):
        qac=qa_classifier();
        qtype=[];
        n=len(table_vqa);
        for i in range(n):
            question=table_vqa['Q'][i];
            answer=table_vqa['A'][i];
            qtype.append(qac.classify_qa(question=question,answer=answer))
        
        return qtype;
    
    def question_vector_v0(self,qtext,T=15,std=1e-3,batch=4,model=0):
        def logmeanexp(inputs,dim=None,keepdim=False):
            return (inputs-F.log_softmax(inputs,dim=dim).data).mean(dim,keepdim=keepdim)-math.log(inputs.size(dim));
        
        seeds=[t*1000 for t in range(T)]; #Fix seeds across runs
        
        #Preprocess question
        _,q=self.parse_question(qtext);
        q=q.view(1,-1);
        
        feature=self.qfvs_imkey['fv36'].cuda();
        feature_7x7=self.qfvs_imkey['fv49'].cuda();
        
        model2=copy.deepcopy(self.models[model]);
        model2.train();
        s=[];
        for t in range(T):
            st=[];
            rng_state=torch.random.get_rng_state();
            torch.random.manual_seed(seeds[t]);
            #Run the model, pairing the q with each images
            with torch.no_grad():
                for j in range(0,feature.shape[0],batch):
                    r=min(j+batch,feature.shape[0]);
                    scores,_=model2(feature[j:r],feature_7x7[j:r],q.repeat(r-j,1));
                    scores=F.log_softmax(scores,dim=1).data;
                    st.append(scores);
            
            torch.random.set_rng_state(rng_state);
            st=torch.cat(st,dim=0);
            s.append(st.data);
        
        s=torch.stack(s,dim=0); #TxKx3129
        savg=logmeanexp(s,dim=0,keepdim=True);
        sdiff=s-savg;
        s=s.permute(1,0,2);
        sdiff=sdiff.permute(1,2,0);
        v=torch.bmm(torch.exp(s),torch.exp(sdiff))/T;
        return v.view(-1).cpu();
    
    
    
