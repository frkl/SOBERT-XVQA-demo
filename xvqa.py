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

#QA classifier
import qa_classifier as qa_classifier
qa_classifier=qa_classifier.qa_classifier;
qtypes=['object', 'color', 'action', 'count', 'time', 'weather']

import model_7x7 as base_model

class xvqa:
    def __init__(self,args_models):
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
        
        self.models=models;
    
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
        nim=len(Is);
        fv36=torch.Tensor(nim,36,2048).fill_(0)
        boxes=torch.Tensor(nim,36,6).fill_(0);
        return fv36,boxes;
    
    def vqa(self,Is,Qs,use_model=''):
        with torch.no_grad():
            qtokens,q=self.parse_question(Qs);
            print(qtokens)
            fv7x7=self.get_7x7_features(Is);
            fv36,boxes=self.get_maskrcnn_features(Is);
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

    
    def explain_attention_map_average(self,table_vqa):
        key=table_vqa['id'][0];
        attn=table_vqa['attention'][0];
        qtoken=table_vqa['qtoken'][0];
        L=len(qtoken);
        attn_sp=attn[-1,:,:L, 66:].mean(0).mean(0).view(7,7);
        attn_fname='./attn/%s_spatial_average.jpg'%key;
        self.write_spatial_attention(table_vqa['I'][0],attn_sp,attn_fname);
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
    
    