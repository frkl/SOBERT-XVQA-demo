import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical 
import sys
import os
sys.path.append("models/VQA")
import numpy as np
import tqdm
import db as db
import pdb
from PIL import Image
import cv2
import random
import math
from vqa_bert_interface import BertVQA
import inflect

ifl = inflect.engine()

default_device = torch.device("cpu")

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


def get_avg_w2v(question, w2v):
    q_w = question.lower().split("?")[0].split(" ")

    avg_feats = []
    for w in q_w:
        if w in w2v:
            avg_feats.append(w2v[w])

    return np.average(avg_feats, axis=0)


def norm_feats(feats):
    feats = torch.tensor(feats)
    if torch.max(feats)!=0 and torch.max(feats)!=torch.min(feats):
        feats = 2*(feats - torch.min(feats))/(torch.max(feats)-torch.min(feats)) - 1.0
    return feats

def format_answers(answers):
    new_ans = []
    for ans in answers:
        format_a = ans.strip().lower().split(".")[0]
        if ifl.singular_noun(format_a)!=False:
            format_a = ifl.singular_noun(format_a)
        new_ans.append(format_a)
    return new_ans


class attention_refine_data(Dataset):

    def __init__(self, data_val=None, data_choice="human_att", split="train", model_choice="simple_bert_7x7_4", num_vals=20, w2v = None, im_feature_choice="bbox", im_feat_flatten=True, nonlinchoice='softmax', att_dim=(14,14), caption=False, device=default_device, cam=False):
        
        if data_choice=="human_att":
            self.base_path = "data/precomputed_attention_"+model_choice+"/"+split
        elif data_choice=="convqa":
            self.base_path = "data/consistentVQASets/"+split # only works with val or val train because there is no human attention for these to train on
        
        self.data_val = db.DB.load(data_val)
        print("loaded vqa data")
        self.answer_dictionary=torch.load('models/VQA/'+model_choice+'/answer_dictionary.pt')
        self.ans2id = dict()
        for i, key in enumerate(self.answer_dictionary):
            self.ans2id[key]=i
        if split in ["val_train", "val"]:
            if num_vals!=-1:
                self.atten_data = random.sample(os.listdir(self.base_path), num_vals)
            else:
                self.atten_data = os.listdir(self.base_path)
        else:    
            self.atten_data = os.listdir(self.base_path)
        
        self.split=split
        if w2v is None:
            self.w2v = loadGloveModel("/data/ARIJIT/glove.6B.300d.txt")
        else:
            self.w2v = w2v
        
        if caption:
            self.cocoid2vqacap_data = dict()
            if split in ['val_train', 'val']:
                cap_data = json.load(open("data/vqa_ques_captions/vqaval.json"))
            else:
                cap_data = json.load(open('data/vqa_ques_captions/vqatrain.json'))
            for entry in cap_data:
                cocoid = int(entry['image_filepath'].split('_')[-1].split(".")[0])
                self.cocoid2vqacap_data[cocoid] = entry['caption']

        if cam and split in ["val", "val_train"]:
            self.vqa_cams = json.load(open("data/"+model_choice+"_VQAval_cams.json"))

        self.cam = cam
        self.im_feature_choice = im_feature_choice
        self.nonlinchoice = nonlinchoice
        self.model_choice = model_choice
        self.att_dim = att_dim
        self.im_feat_flatten = im_feat_flatten
        self.caption = caption
        self.data_choice = data_choice
        self.device = device

    def __getitem__(self, idx):
        qid = self.atten_data[idx].split(".")[0]
        act_qid = qid.split("_")[0]
        coco_id, question, answer, gt_a, max_s, attn = json.load(open(os.path.join(self.base_path, self.atten_data[idx])))
        attn = torch.tensor(attn).flatten() 
        if self.data_choice == "convqa":
            con_qs = question[1:]
            con_as = answer[1:]
            con_gtas = gt_a[1:]
            question = question[0]
            answer = answer[0]
            gt_a = gt_a[0]

        id=self.data_val['table_iid']['coco_id'].index(coco_id)
        iid=self.data_val['table_iid']['iid'][id]
        
        #Get im feature and question feats
        if self.im_feature_choice=="bbox":
            im_feature=self.data_val['table_ifvs']['features_7x7'][iid:iid+1,:,:, :]
        elif self.im_feature_choice=="spatial":
            im_feature=self.data_val['table_ifvs']['features_7x7'][iid:iid+1,:,:]
        
        if self.im_feat_flatten:
            im_feature = im_feature.squeeze().flatten()
        else:
            im_feature = im_feature.squeeze()
        ques_feats = torch.from_numpy(get_avg_w2v(question, self.w2v))
        
        ques_cap_feats = None
        if self.caption: # get saved question as caption if using cap sim models
            ques_cap = self.cocoid2vqacap_data[int(coco_id)]
            ques_cap_feats = torch.from_numpy(get_avg_w2v(ques_cap, self.w2v)).to(self.device).float()

        #get VQA cam for the vqa answer
        vqa_cam = None
        if self.cam==True and self.split in ["val", "val_train"]:
            vqa_cam = self.vqa_cams[str(coco_id)+"_"+str(question)]

        #get human attention if present
        try:
            if self.split in ["val_train", "val"]:
                qid = qid.split("_x")[0]
                human_att = cv2.imread("/data/DataSets/VQA/HumanAttention/vqahat_val/"+str(qid)+".png", cv2.IMREAD_GRAYSCALE)
            else:
                qid = qid.split("_")[0]
                #human_att = cv2.imread("/data/DataSets/VQA/HumanAttention/vqahat_train/"+str(qid)+"_1.png", cv2.IMREAD_GRAYSCALE)
                human_att = cv2.imread("/data/DataSets/VQA/HumanAttention/vqahat_train/"+str(qid)+"_1.png", cv2.IMREAD_GRAYSCALE)
                
            human_att = cv2.resize(human_att, self.att_dim).flatten()

            if self.nonlinchoice=='sigmoid':
                num = (human_att - np.min(human_att))
                den = (np.max(human_att) - np.min(human_att))
                if den<1:
                    den+=1
                    num+=1
                human_att = num/den
                human_att = torch.from_numpy(human_att)
            elif self.nonlinchoice=='softmax':
                human_att = F.softmax(torch.tensor(human_att).float())
        except:
            human_att=None

        data_return = {'coco_id':coco_id, 
                'question':question, 
                'answer':answer, 
                'im_feature':im_feature.to(self.device).float(), 
                'ques_feats':ques_feats.to(self.device).float(),
                'gt_a':gt_a, 
                'max_s':torch.tensor(max_s).to(self.device).float(), 
                'attn':attn.to(self.device).float(), 
                'act_qid': act_qid}

        if vqa_cam is not None:
            data_return['vqa_cam'] = torch.tensor(vqa_cam).flatten().float()

        if ques_cap_feats!=None:
            data_return['ques_cap_feats'] = ques_cap_feats
        if human_att!=None:
            data_return['human_att'] = human_att.to(self.device).float()
        
        if self.data_choice== 'convqa':
            data_return['con_qs'] = con_qs 
            data_return['con_as'] = con_as
            data_return['con_gtas'] = con_gtas
            
                
        return data_return

    def __len__(self):

        return len(self.atten_data)




class human_att_data(Dataset):

    def __init__(self, split="train"):

        ques_data = json.load(open("/data/DataSets/VQA/OpenEnded_mscoco_"+split+"2014_questions.json"))

        ans_data = json.load(open("/data/DataSets/VQA/mscoco_"+split+"2014_annotations.json")) # load answer annotations

        hat_ims = os.listdir("/data/DataSets/VQA/HumanAttention/vqahat_"+split)

        qid2iq= dict()
        for entry in ques_data['questions']:
            qid = entry['question_id']
            im_id = entry['image_id']
            qid2iq[qid]= [im_id, entry['question']]

        
        qid2ans = dict()
        for entry in ans_data['annotations']:
            qid = entry['question_id']
            answer = entry['multiple_choice_answer']
            qid2ans[qid] = answer

        self.hatqid2iq = []
        for imf in hat_ims:
            qid = int(imf.split("_")[0])
            self.hatqid2iq.append([qid,]+ qid2iq[qid] +[qid2ans[qid]])

    
    def __getitem__(self, idx):

        return self.hatqid2iq[idx]

    def __len__(self):
        return len(self.hatqid2iq)


class attention_refine_data_live(Dataset):

    def __init__(self, data_val, vqa_model_file="simple_bert_7x7_4", val=False, split="train", batch_size=100, nonlinchoice='softmax'):

        vqa_hat = human_att_data(split=split)

        self.vqa_hat_data = iter(DataLoader(vqa_hat, batch_size=batch_size))

        self.vqa_model = BertVQA(vqa_model_file, data_val=data_val, gpu_id=0)

        self.data_val = db.DB.load(data_val)

        self.val = val
        self.w2v = loadGloveModel("/data/ARIJIT/glove.6B.300d.txt")

    
    def __getitem__(self, idx):

        qids, imids, questions, gt_ans = self.vqa_hat_data.next()

        a_ids, answers, attns, max_scores = self.vqa_model.getVQAAns_batch(imids, questions, score=True)

        attns = attns.detach()
        max_scores = max_scores.detach().cpu()
        qids = qids.tolist()
        #pdb.set_trace()
        all_attns = []
        all_human_atts = []
        all_ques_feats = []
        all_im_features = []
        for coco_id, question, attn, qid in  zip(imids, questions, attns, qids):

            attn = torch.flatten(attn)
            #pdb.set_trace()

            id=self.data_val['table_iid']['coco_id'].index(coco_id)
            iid=self.data_val['table_iid']['iid'][id]
            
            #Get feature and question
            im_feature=self.data_val['table_ifvs']['features'][iid:iid+1,:,:]
            im_feature = im_feature.squeeze().flatten()
            ques_feats = get_avg_w2v(question, self.w2v)

            #get human attention
            if self.val:
                human_att = cv2.imread("/data/DataSets/VQA/HumanAttention/vqahat_val/"+str(qid)+"_1.png", cv2.IMREAD_GRAYSCALE)
            else:
                human_att = cv2.imread("/data/DataSets/VQA/HumanAttention/vqahat_train/"+str(qid)+"_1.png", cv2.IMREAD_GRAYSCALE)
            human_att = cv2.resize(human_att, (14,14)).flatten()

            '''
            num = (human_att - np.min(human_att))
            den = (np.max(human_att) - np.min(human_att))
            if den<1:
                den+=1
                num+=1
            human_att = num/den
            '''
            human_att = F.softmax(torch.tensor(human_att).float())

            all_attns.append(attn)
            all_human_atts.append(human_att)
            all_ques_feats.append(torch.tensor(ques_feats))
            all_im_features.append(torch.tensor(im_feature))

        all_attns = torch.stack(all_attns)
        all_human_atts = torch.stack(all_human_atts)
        all_ques_feats = torch.stack(all_ques_feats)
        all_im_features = torch.stack(all_im_features)

        return imids, questions, answers, all_im_features, all_ques_feats, gt_ans, max_scores, all_attns, all_human_atts

    def __len__(self):

        return len(self.vqa_hat_data)



