import numpy as np
import json
import os
from scipy.stats import spearmanr as correlation_func_atten
from statsmodels.stats.weightstats import ztest
import math
import cv2

import sys
sys.path.append("models/VQA")
import db as db
from torch.utils.data import Dataset, DataLoader
import torch
import models.attention_refine.atten_refine_network as att_refine
from scripts.pytorchgradcam.gradcam import GradCam
from datasets import attention_refine_data


class atten_error_helpfulness:

    def __init__(self):

        ques_data = json.load(open("/data/DataSets/VQA/OpenEnded_mscoco_val2014_questions.json"))

        ans_data = json.load(open("/data/DataSets/VQA/mscoco_val2014_annotations.json")) # load answer annotations

        hat_ims = os.listdir("data/vqahat_val")

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

        self.hatqid2iqa = []
        for imf in hat_ims:
            qid = int(imf.split("_")[0])
            human_att = cv2.imread("data/vqahat_val/"+str(imf), cv2.IMREAD_GRAYSCALE)
            human_att = cv2.resize(human_att, (7,7)).flatten()

            num = (human_att - np.min(human_att))
            den = (np.max(human_att) - np.min(human_att))
            if den<1:
                den+=1
                num+=1
            human_att = num/den
            
            self.hatqid2iqa.append([qid, human_att]+ qid2iq[qid] +[qid2ans[qid]])

    
    def calc_helpfulness(self, heatmap_data):
        #assumes atten_map data as a dictionary of qid: ['answer': answer_text, 'atten': raw attention grid data, 'err': raw error grid data]

        all_corr_correct_atten = []
        all_corr_wrong_atten = []

        all_corr_correct_err = []
        all_corr_wrong_err = []

        all_corr_correct_attenerr = []
        all_corr_wrong_attenerr = []

        eval_count=0
        
        for qid, h_att, imid, ques, gt_ans in self.hatqid2iqa:
            qid = str(qid)
            if qid in heatmap_data:
                ans = heatmap_data[qid]['answer']
                atten = np.array(heatmap_data[qid]['atten']).flatten()
                c=0
                if 'err' in heatmap_data[qid]:
                    err = np.array(heatmap_data[qid]['err']).flatten()
                    #print(atten, h_att)
                    try:
                        corr, p = correlation_func_atten(err, h_att)
                    
                        if not math.isnan(corr):
                            if ans==gt_ans:
                                all_corr_correct_err.append(corr)
                            else:
                                all_corr_wrong_err.append(corr)
                        c=1
                    except:
                        _=1
                
                if 'atten' in heatmap_data[qid]:
                    atten = np.array(heatmap_data[qid]['atten']).flatten()
                    #print(atten, h_att)
                    try:
                        corr, p = correlation_func_atten(atten, h_att)
                    
                        if not math.isnan(corr):
                            if ans==gt_ans:
                                all_corr_correct_atten.append(corr)
                            else:
                                all_corr_wrong_atten.append(corr)
                        c=1
                    except:
                        _=1

                if 'err' in heatmap_data[qid] and 'atten' in heatmap_data[qid]:
                    
                    try:
                        corr, p = correlation_func_atten(err, atten)
                    
                        if not math.isnan(corr):
                            if ans==gt_ans:
                                all_corr_correct_attenerr.append(corr)
                            else:
                                all_corr_wrong_attenerr.append(corr)
                        c=1
                    except:
                        _=1
                
                if c==1:
                    eval_count+=1

        eval_scores = dict()
        if all_corr_correct_atten!=[]:
            z,p = ztest(all_corr_correct_atten, all_corr_wrong_atten)
            eval_scores['z_help_atten'], eval_scores['help_pval_atten'] = round(z,4), round(p, 4)
            eval_scores['rel_correct_atten'] = round(np.average(all_corr_correct_atten), 4)
            eval_scores['rel_wrong_atten'] = round(np.average(all_corr_wrong_atten), 4)

        if all_corr_correct_err!=[]:
            z, p = ztest(all_corr_correct_err, all_corr_wrong_err)
            eval_scores['z_help_err'], eval_scores['help_pval_err'] = round(-z, 4), round(-p, 4)
            eval_scores['rel_correct_err'] = round(np.average(all_corr_correct_err), 4)
            eval_scores['rel_wrong_err'] = round(np.average(all_corr_wrong_err), 4)

        if all_corr_correct_attenerr!=[]:
            z,p = ztest(all_corr_correct_attenerr, all_corr_wrong_attenerr)
            eval_scores['z_help_attenerr'], eval_scores['help_pval_attenerr'] = round(-z, 4),round(-p, 4)
            eval_scores['rel_correct_attenerr'] = round(np.average(all_corr_correct_attenerr), 4)
            eval_scores['rel_wrong_attenerr'] = round(np.average(all_corr_wrong_attenerr), 4)

        if eval_count<1:
            return None
        
        return eval_scores


def get_avg_w2v(question, w2v):
    q_w = question.lower().split("?")[0].split(" ")

    avg_feats = []
    for w in q_w:
        if w in w2v:
            avg_feats.append(w2v[w])

    return np.average(avg_feats, axis=0)

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

class ErrorMaps:

    def __init__(self, device = None, w2v=None, data_val = None):

        #load data
        if data_val==None:
            self.data_val = db.DB.load("models/VQA/data_vqa_val.pt")
        else:
            self.data_val = data_val

        if w2v==None:
            self.w2v = loadGloveModel("/data/ARIJIT/glove.6B.300d.txt")
        else:
            self.w2v = w2v

        # LOAD MODEL
        if device==None:
            self.device = torch.device("cuda")
        else:
            self.device = device
        atten_dim = (4,12,115,115)
        model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129, "ques_cam":True}
        self.attention_refine_model = att_refine.uncertainatt_refinedatt_net_cam_bigger(**model_init_args).to(self.device)
        model_suffix = "model_3_5501.pt"
        exp_name = "exp4_fullmodel_corrpred_refinedattn_uncertainCAM_bigger"
        self.attention_refine_model.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+model_suffix))
        self.gradcam = GradCam(self.attention_refine_model)

    def computeErrorMap(self, atten_weights, coco_id, question, answer_prob):

        #get image features
        id=self.data_val['table_iid']['coco_id'].index(coco_id)
        iid=self.data_val['table_iid']['iid'][id]
        
        #Get im feature and question feats
        im_feature=self.data_val['table_ifvs']['features_7x7'][iid:iid+1,:,:]
        #im_feature = im_feature.squeeze()
        im_feature = im_feature.to(self.device).float()


        #get question features 
        ques_feats = torch.from_numpy(get_avg_w2v(question, self.w2v))
        ques_feats = ques_feats.to(self.device).float().unsqueeze(0)

        #get answer probabilities
        atten_weights = atten_weights.to(self.device).float().unsqueeze(0)
        answer_prob = torch.tensor(answer_prob).to(self.device).float().unsqueeze(0)

        #compute errmap: gradcam for error prediction output
        err_cam, grad_q_feats = self.gradcam([atten_weights, im_feature, ques_feats, answer_prob])

        return err_cam

    def compute_errmap_image(self, atten_weights, coco_id, question, answer_prob, errmap_im_file_name):
        #can only visualize for val images

        att_map = self.computeErrorMap(atten_weights, coco_id, question, answer_prob) #att_map is the err map here
        
        split="val"
        im_file = "/data/DataSets/COCO/"+split+"2014/COCO_"+split+"2014_"+str(int(coco_id)).zfill(12)+".jpg"
        im = cv2.imread(im_file)
        #pdb.set_trace()
        processed_img = cv2.resize(im, (224,224))
        att_map = att_map.reshape((7,7))
        att_map = cv2.resize(att_map, (224,224)) 

        epsilon = 1e-3
        #att_map = (att_map - np.min(att_map) + epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
        
        att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
        alpha = 0.5
        output_image = (1 - alpha) * att_heatmap + alpha * processed_img

        cv2.imwrite(errmap_im_file_name, output_image)



if __name__=="__main__":
    ############# tutorial of how to generate error maps. ######################

    ########### Optional data loading ###############
    #if you dont load data, error map model will intenally load all data. 
    # I am providing them as arguments to reduce redundant copies of data
    w2v = loadGloveModel("/data/ARIJIT/glove.6B.300d.txt")

    #load precomputed attention weights, in reality, you will feed in the attention weights from the VQA model. 
    #skip this step if you are using a live VQA model
    
    device = torch.device("cuda")
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'simple_bert_7x7_4', 
                          'im_feat_flatten':False,
                          'device': device}
    atten_dataset_val = attention_refine_data(w2v=w2v, split="val", **val_dataset_args)
    ##################################################


    ##################### ERROR MAP MODEL ########################
    #load error map generating model
    errmap_model = ErrorMaps(data_val=atten_dataset_val.data_val, w2v=w2v, device=device) 
    # all arguments to ErrorMaps are optional. it will create its own copies of data if you dont provide them. 
    # you can provide them as arguments to avoid redundant loading in case you already have them loaded beforhand. 

    #get atten weights and answer_probs from precomputed VQA model, again, use a live VQA model if required.
    data_dictionary = next(iter(atten_dataset_val)) 

    atten_weights = data_dictionary['attn'] # this should come from the live VQA model, but we are using a precomputed one here.
    #atten_weights is a 4 x 12 x 115 x 115 BERT attention weights. 

    coco_id = data_dictionary['coco_id']  #something like "12345"
    question = data_dictionary['question'] #something like "what is girl doing?"
    answer_probs = data_dictionary['max_s'] # this should also come from live VQA model. This is a 1 x 3129 answer probabilities

    #save the errmap image
    errmap_model.compute_errmap_image(atten_weights, coco_id, question, answer_probs, errmap_im_file_name = "test_errmap.png")



    


    