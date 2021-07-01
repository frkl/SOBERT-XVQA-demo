from collections import defaultdict
import pprint
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical 
import sys
import os
sys.path.append("models/VQA")
import numpy as np
from models.attention_refine.atten_refine_network import attention_refine_net
import tqdm
import db as db
import pdb
from PIL import Image
import cv2
import random
import math
from scipy.stats import spearmanr as correlation_func_atten
from scipy.stats import pearsonr as correlation_func_quality
from scipy.stats import ttest_ind
#from scipy.stats import spearmanr as correlation_func
from datasets import get_avg_w2v, attention_refine_data
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
from scripts.qa_classifier import qa_classifier
from statsmodels.stats.weightstats import ztest

qa_classify = qa_classifier()


############# eval output postprocess routines ####################
# process outputs and store to variables for use in evaluation later. 

def gen_cams(batch_size, gradcam, dataouts, input_choice, w2v = None, index=None):
    cams = [] 
    batch_size = dataouts['im_feature'].shape[0]
    for ev_i in range(batch_size):
        #try:
        cam, grads_q_feat = gradcam([dataouts[inp][ev_i:ev_i+1] for inp in input_choice], index=index)
            #pdb.set_trace()
        #except:
        #    pdb.set_trace()
        cams.append(cam.flatten())
    
    return cams

def gen_word_cams(batch_size, gradcam, dataouts, input_choice, w2v, index=None):
    word_weights = [] 
    batch_size = dataouts['im_feature'].shape[0]
    for ev_i in range(batch_size):
        cam, grads_q_feat = gradcam([dataouts[inp][ev_i:ev_i+1] for inp in input_choice], index=index)
        
        q_w = dataouts['question'][ev_i].lower().split("?")[0].split(" ")
        avg_feats = []
        for w in q_w:
            if w in w2v:
                avg_feats.append(w2v[w])
        
        avg_feats = np.array(avg_feats).T

        word_weight = np.matmul(grads_q_feat.squeeze(), avg_feats)
        word_weight = np.maximum(word_weight, 0)
        word_weight = (word_weight - np.min(word_weight))/(np.max(word_weight) - np.min(word_weight) + 1e-4)
        word_weights.append(word_weight)
    
    return word_weights

def gen_bert_cams(batch_size, gradcam, dataouts, input_choice, w2v = None, index=None):
    cams = [] 
    batch_size = dataouts['im_feature'].shape[0]
    for ev_i in range(batch_size):
        #try:
        cam, grads_q_feat, bert_cams = gradcam([dataouts[inp][ev_i:ev_i+1] for inp in input_choice], index=index)
            #pdb.set_trace()
        #except:
        #    pdb.set_trace()
        cams.append(bert_cam.squeeze())
    
    return cams

def calc_errorcam_bert_correlations(batch_size, gradcam, dataouts, input_choice):
    bert_cams = gen_bert_cams(batch_size, gradcam, dataouts, input_choice)
    all_bert_weights = dataouts['attn']
    questions= dataouts['question']

    #for 



def make_baseline_attention(all_dense_att_weights, questions, atten_shape):
    #pdb.set_trace()
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")

        attention_sp = dense_att_weights[-1, :, :len(q_w), 66:].mean(0).mean(0).numpy()
        attention_sp = np.reshape(attention_sp, (7,7))
        all_spatial_attens.append(attention_sp.flatten())

    return all_spatial_attens

def make_bestBERT_attention(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[0, 3, :len(q_w), 66:].mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens

def make_bestBERT_attention_fullmodel(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[2, :, :len(q_w), 66:].mean(0).mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens

def make_bestBERT_attention_action(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[0, 6, :len(q_w), 66:].mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens

def make_bestBERT_attention_layeronly(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[1, :, :len(q_w), 66:].mean(0).mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens

def make_bestBERT_attention_headonly(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[:, 4, :len(q_w), 66:].mean(0).mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens


def make_bestBERT_errorcam(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_spatial_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        sp_atten = dense_att_weights[3, 7, :len(q_w), 66:].mean(0).numpy()
        all_spatial_attens.append(sp_atten.flatten())
    
    return all_spatial_attens

def log_all_bert_correlations(all_dense_att_weights, questions, atten_shape, attn2):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    attn2 = attn2.cpu().detach().numpy()
    #atten_shape is in the form of num_layers x num_heads x dim x dim
    num_layers = atten_shape[0]
    num_heads = atten_shape[1]
    all_corrl = np.zeros((all_dense_att_weights.shape[0], num_layers, num_heads))
    batch_cnt = 0
    for dense_att_weights, ques, a2 in zip(all_dense_att_weights, questions, attn2):
        q_w = ques.lower().split("?")[0].split(" ")
        for layer in range(num_layers):
            for head in range(num_heads):
                sp_atten = dense_att_weights[layer, head, :len(q_w), 66:].mean(0).numpy()
                corrl, p = correlation_func_atten(sp_atten, a2)
                if not math.isnan(corrl):
                    all_corrl[batch_cnt, layer, head] = corrl
        batch_cnt+=1
    
    return all_corrl

def log_all_bertquesatten_correlations_queserrorcam(all_dense_att_weights, questions, atten_shape, attn2):
    attn2 = attn2[-len(questions):]
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    #attn2 = attn2.cpu().detach().numpy()
    #atten_shape is in the form of num_layers x num_heads x dim x dim
    num_layers = atten_shape[0]
    num_heads = atten_shape[1]
    all_corrl = np.zeros((all_dense_att_weights.shape[0], num_layers, num_heads))
    batch_cnt = 0
    for dense_att_weights, ques, a2 in zip(all_dense_att_weights, questions, attn2):
        q_w = ques.lower().split("?")[0].split(" ")
        for layer in range(num_layers):
            for head in range(num_heads):
                ques_atten = dense_att_weights[layer, head, 30:, :len(q_w)].mean(0).numpy()
                try:
                    corrl, p = correlation_func_atten(ques_atten, a2)
                except:
                    continue
                if not math.isnan(corrl):
                    all_corrl[batch_cnt, layer, head] = corrl
        batch_cnt+=1
    
    return all_corrl

def log_all_bert_correlations_errorcam(all_dense_att_weights, questions, atten_shape, attn2):
    attn2 = attn2[-len(questions):]
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    #attn2 = attn2.cpu().detach().numpy()
    #atten_shape is in the form of num_layers x num_heads x dim x dim
    num_layers = atten_shape[0]
    num_heads = atten_shape[1]
    all_corrl = np.zeros((all_dense_att_weights.shape[0], num_layers, num_heads))
    batch_cnt = 0
    
    for dense_att_weights, ques, a2 in zip(all_dense_att_weights, questions, attn2):
        q_w = ques.lower().split("?")[0].split(" ")
        for layer in range(num_layers):
            for head in range(num_heads):
                sp_atten = dense_att_weights[layer, head, :len(q_w), 66:].mean(0).numpy()
                corrl, p = correlation_func_atten(sp_atten, a2)
                if not math.isnan(corrl):
                    all_corrl[batch_cnt, layer, head] = corrl
        batch_cnt+=1
    
    return all_corrl




def detach_output(o):
    return o.cpu().detach().numpy()

def calc_avg_imvstext(bert_attns, questions, atten_shape):
    # bert atten is 4 x 8 (or 12) x 115 x 115 . The 115 is ordered as 30 words, 36 bboxes, 49 image sp feats.

    bert_attns = bert_attns.view(-1, *atten_shape).cpu().detach()
    all_metrics = []
    for bert_attn, question in zip(bert_attns, questions):
        q_w = question.lower().split("?")[0].split(" ")
        avg_word_atten = float(bert_attn[-1, :, :, :len(q_w)].mean(0).mean(0).mean())
        avg_im_atten = float(bert_attn[-1, :, :, 30:].mean(0).mean(0).mean())

        max_word_atten = float(bert_attn[-1, :, :, :len(q_w)].mean(0).mean(0).max())
        max_im_atten = float(bert_attn[-1, :, :, 30:].mean(0).mean(0).mean(0).max())
        all_metrics.append((avg_word_atten, avg_im_atten, max_word_atten, max_im_atten))

    return all_metrics

def make_bert_wordatten(all_dense_att_weights, questions, atten_shape):
    all_dense_att_weights = all_dense_att_weights.view(-1, *atten_shape).cpu().detach()
    all_word_attens = []
    for dense_att_weights, question in zip(all_dense_att_weights, questions):
        q_w = question.lower().split("?")[0].split(" ")
        avg_word_attens = dense_att_weights[0, 4, 30:, :len(q_w)].mean(0)
        all_word_attens.append(avg_word_attens.flatten().numpy())
    
    return all_word_attens




################ Evaluation Functions ####################
# functions to calculate accuracies, correlations etc.
def calc_correlations(attn1, attn2, gt_a, answer):
    all_corr_correct = []
    all_corr_wrong = []
    eval_count=0
    for a1, a2, a, gt in zip(attn1, attn2, answer, gt_a):
            try:
                corr, p = correlation_func_atten(a1, a2)
            except:
                continue
            if math.isnan(corr):
                #pdb.set_trace()
                #print("continue")
                continue
            if a==gt:
                all_corr_correct.append(corr)
            else:
                all_corr_wrong.append(corr)
            eval_count+=1

    all_corrs = all_corr_correct + all_corr_wrong
    try:
        z_score, p_z = ztest(all_corr_correct, all_corr_wrong)
    except:
        z_score = "None"
        p_z = "None"
    #pdb.set_trace()
    #print("Num Eval : "+str(eval_count))
    #t, p = ttest_rel(all_corr_correct, all_corr_wrong)
    #qcorr, p_c = correlation_func_quality([1, 0], [np.average(all_corr_correct), np.average(all_corr_wrong)])
    txt = "Corr Correct: "+str(np.average(all_corr_correct))+" Corr Wrong: "+str(np.average(all_corr_wrong))+" Corr avg: "+str(np.average(all_corrs))
    txt += "\n Z-test z_score: %s , p: %s"%(str(z_score), str(p_z))
    #txt += "\n T-test t: %s , p: %s"%(str(t), str(p))
    #txt += "\n Acc corr: %s, p: %s"%(str(qcorr), str(p_c))

    return txt

def calc_accuracy_soft(preds, labels, best_thresh=-1):
    preds = np.asarray(preds)
    if type(labels) is torch.Tensor:
        labels = labels.cpu().detach().numpy()
    avg_prec = average_precision_score(labels, preds)
    best_acc = 0
    if best_thresh==-1:
        best_thresh = 0
        for thresh in np.arange(0.1,0.9,0.025):
            acc = (recall_score(labels, preds>thresh) + recall_score(labels<0.5, preds<thresh))/2.0
            if acc >best_acc:
                best_acc=acc
                best_thresh=thresh
    else:
        best_acc = (recall_score(labels, preds>best_thresh) + recall_score(labels<0.5, preds<best_thresh))/2.0
    
    norm_recall = best_acc

    best_acc = accuracy_score(labels, preds>best_thresh)
    
    return avg_prec, best_acc, best_thresh, norm_recall

def calc_accuracy_wrongpred(pred, answer, gt_a, best_thresh=-1):
    wrong_labels = np.array(gt_a)!=np.array(answer)

    avg_prec, best_acc, best_thresh, norm_recall = calc_accuracy_soft(pred, wrong_labels, best_thresh)

    return "NormRecall: %.3f, AP: %.3f, BestAcc: %.3f, BestThresh: %.3f"%(norm_recall, avg_prec, best_acc, best_thresh)

def calc_accuracyclass_wrongpred(pred, answer, gt_a):
    wrong_labels = np.array(gt_a)!=np.array(answer)
    wrong_preds = torch.argmax(torch.stack(pred), axis=1).cpu().detach().numpy()

    return "Accuracy: %.4f"%(accuracy_score(wrong_labels, wrong_preds))


def calc_acc_correlation_histogram(attn1, attn2, gt_a, answer, name=""):
    if type(attn1) is torch.Tensor:
        attn1 = attn1.cpu().detach().numpy()
    if type(attn2) is torch.Tensor:
        attn2 = attn2.cpu().detach().numpy()

    corr_acc = defaultdict(list)
    corr_acc_list = []
    acc_corr = []
    for a1, a2, a, gt in zip(attn1, attn2, answer, gt_a):
            try:
                corr, p = correlation_func_atten(a1, a2)
            except:
                continue
            if math.isnan(corr):
                #pdb.set_trace()
                #print("continue")
                continue
            bin_c = float("%.1f"%(corr))
            corr_acc[bin_c].append(a==gt)

            corr_acc_list.append((corr, a==gt))

            acc_corr.append([a==gt, corr])

    for key in corr_acc:
        corr_acc[key] = (np.average(corr_acc[key]), len(corr_acc[key]))
        

    items = sorted(list(corr_acc.items()), key=lambda x:x[0])

    if len(items)<2:
        return "0"
    
    if False:
        corrs_bin, accs_ns_bin = tuple(zip(*items))
        accs_bin, ns_bin = tuple(zip(*accs_ns_bin))
        correlation, p = correlation_func_quality(corrs_bin, accs_bin)

    if False:
        acc_corr.sort(key=lambda x:x[0])
        accs, corrs = list(zip(*acc_corr))
        correlation, p = correlation_func_quality(corrs, accs)

    if True:
        corr_acc_list.sort(key=lambda x:x[0])
        n_bin = 10
        bin_len = len(corr_acc_list)//n_bin
        all_avgs = []
        bin_avgs = []
        for c, a in corr_acc_list:
            bin_avgs.append([c,a])
            if len(bin_avgs)==bin_len:
                all_avgs.append(np.average(bin_avgs, axis=0))
                bin_avgs=[]
        if len(all_avgs)<2:
            return "0"
        corrs, accs = list(zip(*all_avgs))
        correlation, p = correlation_func_quality(corrs, accs)
            

    
    #plt.figure()
    #plt.scatter(ind, accs)
    #plt.scatter(ind, corrs)
    #plt.savefig(name+".png")
    #plt.close()

    txt = "Accuracy-AttenQuality Correlation: "+str(correlation)+", p value: "+str(p)

    return txt


def calc_baseline_centeredatt_corrs(human_att, gt_as, answers, preds):
    preds = np.array(preds)

    arr = np.zeros((256,256), dtype=np.uint8)
    imgsize = arr.shape[:2]
    innerColor = (255, 255, 255)
    outerColor = (0, 0, 0)
    for y in range(imgsize[1]):
        for x in range(imgsize[0]):
            #Find the distance to the center
            distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)

            #Make it on a scale from 0 to 1innerColor
            distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)

            #Calculate r, g, and b values
            r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
            #g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
            #b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)
            # print r, g, b
            arr[y, x] = int(r) #, int(g), int(b))
    
    center_att = cv2.resize(arr, (7,7))
    center_att = center_att.flatten()

    center_att_all = np.tile(center_att, (len(gt_as), 1))
    random_att = np.random.rand(49)
    random_att_all = np.random.rand(len(gt_as), 49)

    txt = "=====Center Att all===== \n"
    txt += calc_correlations(human_att, center_att_all, gt_as, answers)
    txt += "\n"
    txt += calc_acc_correlation_histogram(human_att, center_att_all, gt_as, answers)

    txt += "\n=====Random Att all===== \n"
    txt += calc_correlations(human_att, random_att_all, gt_as, answers)
    txt += "\n"
    txt += calc_acc_correlation_histogram(human_att, random_att_all, gt_as, answers)

    all_center_random_exp = []
    for pred in preds:
        if float(pred)<0.175:
            all_center_random_exp.append(center_att)
        else:
            all_center_random_exp.append(random_att)

    txt += "\n=====Center_random Att===== \n"
    txt += calc_correlations(human_att, all_center_random_exp, gt_as, answers)
    txt += "\n"
    txt += calc_acc_correlation_histogram(human_att, all_center_random_exp, gt_as, answers)

    return txt


def calc_consistency_predacc(preds, answers, gt_as, all_con_as, all_con_gtas, best_thresh=0.2):
    #check when error pred is True, but vqa ans is correct, is consistency lower than when error pred is False and vqa ans is correct
    #check when error pred is False, b ut vqa ans is wrong, is consistency higher than when error pred is True and vqa ans is wrong.
    prederror_vqacorr_conaccs = []
    predcorr_vqacorr_conaccs = []
    predcorr_vqaerror_conaccs = []
    prederror_vqaerror_conaccs = []

    for pred, ans, gt_a, con_as, con_gtas in zip(preds, answers, gt_as, all_con_as, all_con_gtas):
        con_acc = accuracy_score(con_gtas, con_as)

        if pred>best_thresh and ans==gt_a:
            prederror_vqacorr_conaccs.append(con_acc)
        if pred<best_thresh and ans==gt_a:
            predcorr_vqacorr_conaccs.append(con_acc)
        if pred<best_thresh and ans!=gt_a:
            predcorr_vqaerror_conaccs.append(con_acc)
        if pred>best_thresh and ans!=gt_a:
            prederror_vqaerror_conaccs.append(con_acc)

    txt = "VQA CORRECT but error-pred ERROR consistency : %.4f (%d) vs error-pred CORRECT consistency : %.4f (%d) \n\
           VQA ERROR but error-pred CORRECT consistency : %.4f (%d) vs error-pred ERROR consistency : %.4f (%d)"%(np.average(prederror_vqacorr_conaccs), 
                                                                                                        len(prederror_vqacorr_conaccs),
                                                                                                        np.average(predcorr_vqacorr_conaccs),
                                                                                                        len(predcorr_vqacorr_conaccs),
                                                                                                        np.average(predcorr_vqaerror_conaccs),
                                                                                                        len(predcorr_vqaerror_conaccs),
                                                                                                        np.average(prederror_vqaerror_conaccs),
                                                                                                        len(prederror_vqaerror_conaccs))

    return txt


def calc_camattncorr_consistency_hist(cams, attens, answers, gt_as, all_con_as, all_con_gtas, exp_name):
    #plot histogram of consistency and accuracies for different correlations for error-CAM and attention
    corr_dict = defaultdict(list)
    for cam, atten, answer, gt_a, con_as, con_gtas in zip(cams, attens, answers, gt_as, all_con_as, all_con_gtas):
        con_acc = accuracy_score(con_gtas, con_as)
        vqa_acc = float(gt_a==answer)
        cam_corr, _ = correlation_func_atten(cam, atten)

        if not math.isnan(cam_corr):
            corr_bin = float("%.1f"%(cam_corr)) #a terrible hack, but I dont know how else to do this.
            corr_dict[corr_bin].append((con_acc, vqa_acc))

    corr_dict_items = sorted(list(corr_dict.items()), key=lambda x:x[0])
    all_conaccs, all_vqaaccs = np.array([np.average(item, axis=0) for key, item in corr_dict_items]).transpose()
    corr_bins = sorted(list(corr_dict.keys()))



    #plot histogram
    if False:
        plt.plot(corr_bins, all_conaccs, 'r')
        plt.plot(corr_bins, all_vqaaccs, 'b')
        plt.savefig("vis/"+exp_name+"/camattencorr_consistency_hist.png")

    return ""

def calc_predbased_correlations(attn1, attn2, gt_a, answer, wrong_pred):
    predright_vqaright_corr = []
    predright_vqawrong_corr = []
    predwrong_vqawrong_corr = []
    predwrong_vqaright_corr = []

    for a1, a2, gt, ans, pred in zip(attn1, attn2, gt_a, answer, wrong_pred):
        pred = pred<0.2 #True is vqaright 
        vqaright = gt==ans

        correlation, p = correlation_func_atten(a1, a2)

        if math.isnan(correlation):
            continue

        if pred and vqaright:
            predright_vqaright_corr.append(correlation)
        elif pred and not vqaright:
            predright_vqawrong_corr.append(correlation)
        elif not pred and not vqaright:
            predwrong_vqawrong_corr.append(correlation)
        elif not pred and vqaright:
            predwrong_vqaright_corr.append(correlation)

    txt = "Correlations \n VQA CORRECT, Pred CORRECT: %.4f (%d) vs Pred ERROR: %.4f (%d) \n\
           VQA ERROR, Pred CORRECT: %.4f (%d) vs Pred ERROR: %.4f (%d)"%(np.average(predright_vqaright_corr), len(predright_vqaright_corr),
                                                                        np.average(predwrong_vqaright_corr), len(predwrong_vqaright_corr),
                                                                        np.average(predright_vqawrong_corr), len(predright_vqawrong_corr),
                                                                        np.average(predwrong_vqawrong_corr), len(predwrong_vqawrong_corr))

    return txt

def question_type_attention_quality(attn1, attn2, gt_a, answer, question):
    qtype_acc = defaultdict(list)
    qtype_corr = defaultdict(list)
    for q, ans, gt, a1, a2 in zip(question, answer, gt_a, attn1, attn2):
        q_class = qa_classify.classify_qa(q, gt)
        corr, _ = correlation_func_atten(a1, a2)
        if q_class!=None:
            qtype_acc[q_class].append(gt==ans)
        if not math.isnan(corr):
            qtype_corr[q_class].append(corr)

    for key in qtype_acc:
        qtype_acc[key] = np.average(qtype_acc[key])

    for key in qtype_corr:   
        qtype_corr[key] = np.average(qtype_corr[key])
    
    txt = "Qtype Accuracies: \n"
    txt+= pprint.pformat(qtype_acc)
    txt+= "\n Qtype Att Quality \n"
    txt+= pprint.pformat(qtype_corr)

    return txt


def calc_all_bert_correlations(all_corrs, answers, gt_as):
    #all_corrs is in the form of batch_size x num_layers x num_heads
    all_corrs = np.array(all_corrs)
    num_heads = all_corrs.shape[2]
    num_layers = all_corrs.shape[1]
    batch = all_corrs.shape[0]
    #print(all_corrs.mean(0))
    all_acc_corr = np.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        for head in range(num_heads):
            corr_acc = defaultdict(list)
            acc_corr = []
            corr_acc_list = []
            corr_rels = []
            for i in range(batch):
                corr = all_corrs[i, layer, head]
                corr_acc[float("%.1f"%(corr))].append(answers[i]==gt_as[i])
                acc_corr.append((answers[i]==gt_as[i], corr))
                corr_acc_list.append((corr, answers[i]==gt_as[i]))

                if answers[i]==gt_as[i]:
                    corr_rels.append(corr)
            
            if False:
                for key in corr_acc:
                    corr_acc[key] = (np.average(corr_acc[key]), len(corr_acc[key]))
                items = sorted(list(corr_acc.items()), key=lambda x:x[0])
                corrs, accs_ns = tuple(zip(*items))            
                accs, ns = tuple(zip(*accs_ns))

            if False:
                acc_corr.sort(key=lambda x:x[0])
                accs, corrs = list(zip(*acc_corr))

            if True:
                corr_acc_list.sort(key=lambda x:x[0])
                n_bin = 10
                bin_len = len(corr_acc_list)//n_bin
                all_avgs = []
                bin_avgs = []
                for c, a in corr_acc_list:
                    bin_avgs.append([c,a])
                    if len(bin_avgs)==bin_len:
                        all_avgs.append(np.average(bin_avgs, axis=0))
                        bin_avgs=[]
                
                corrs, accs = list(zip(*all_avgs))
                #correlation, p = correlation_func_quality(corrs, accs)

            
            att_acc_corr, p = correlation_func_quality(corrs, accs)
            all_acc_corr[layer, head] = att_acc_corr
    
    return str(all_acc_corr)

def calc_strength_acc(attn, errormap, vqa_a, gt_a, wrong_pred):
    if type(attn) is torch.Tensor:
        attn = attn.cpu().detach().numpy()
    if type(errormap) is torch.Tensor:
        errormap = errormap.cpu().detach().numpy()
    
    count_corr = 0
    all_count = 0
    for att, err, ans, gt, pred in zip(attn, errormap, vqa_a, gt_a, wrong_pred):
        max_attn = np.max(att)
        max_errormap = np.max(err*pred)

        if max_attn>max_errormap:
            if ans==gt:
                count_corr+=1
        else:
            if ans!=gt:
                count_corr+=1
        all_count+=1
    
    txt = "Strength-based acc: "+str(float(count_corr)/all_count)

    return txt

def calc_alignbased_acc(attn, errormap, human_att, vqa_a, gt_a, wrong_pred):
    if type(attn) is torch.Tensor:
        attn = attn.cpu().detach().numpy()
    if type(errormap) is torch.Tensor:
        errormap = errormap.cpu().detach().numpy()
    if type(human_att) is torch.Tensor:
        human_att = human_att.cpu().detach().numpy()
    
    all_cnt=0
    hyp1_corrcnt=0
    best_acc = 0
    for align_thresh in range(2,7):
        align_thresh = float(align_thresh)/10.0
        for att, err, h_att, ans, gt, pred in zip(attn, errormap, human_att, vqa_a, gt_a, wrong_pred):
            att_humatt_corr, p = correlation_func_atten(att, h_att)
            err_att_corr, p = correlation_func_atten(err, att)
            #err_hum_corr, p = correlation_func_atten(err, h_att)

            #hypothesis 1
            if float(att_humatt_corr)>align_thresh:
                if float(err_att_corr)>align_thresh:
                    if ans!=gt:
                        hyp1_corrcnt+=1
                else:
                    if ans==gt:
                        hyp1_corrcnt+=1
            else:
                if ans!=gt:
                    hyp1_corrcnt+=1
        
            all_cnt +=1

        acc = hyp1_corrcnt/float(all_cnt)

        if acc>best_acc:
            best_acc=acc
            best_thresh = align_thresh
            print(align_thresh)
    
    txt = "Hyp1 Best Acc: %f, Best Thresh: %f"%(best_acc, best_thresh)

    return txt


########unused
def evaluate_avgprec(human_att, refined_attn, answer, gt_a):
    human_att = human_att.cpu().detach().numpy()
    refined_attn = refined_attn.cpu().detach().numpy()
    precs_corr = []
    precs_wrong = []
    for human_a, refined_a, a, gt in zip(human_att, refined_attn, answer, gt_a):
        human_max = np.max(human_att)
        refined_max = np.max(refined_attn)

        thresh_human = np.linspace(0, human_max, 10)
        thresh_refined = np.linspace(0, refined_max, 10)

        all_precs = 0
        for t_h, t_r in zip(thresh_human, thresh_refined):
            human_salient = human_a>t_h
            refined_salient = refined_a>t_r
            all_precs += precision_score(human_salient, refined_salient)
        if a==gt:
            precs_corr.append(all_precs/10.0)
        else:
            precs_wrong.append(all_precs/10.0)    
    
    return precs_corr, precs_wrong 


########## For online eval script 

def generate_eval_json(exp_name):
    eval_lists = torch.load("checkpoints/"+exp_name+"/saved_examples_val.pt")

    qids = eval_lists['qid']
    answers = eval_lists['answer']
    atten_map = eval_lists['all_ref_attn']
    err_map = eval_lists['all_cams']

    eval_json = dict()
    for ind, qid in enumerate(qids):
        eval_json[qid]={'atten':atten_map[ind].tolist(), 'err':err_map[ind].tolist(), 'answer': answers[ind]}

    with open("checkpoints/"+exp_name+"/online_eval_jatt.json", "w") as f:
        json.dump(eval_json, f)

if __name__=="__main__":
    exp_name = "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger_recheck"
    generate_eval_json(exp_name)