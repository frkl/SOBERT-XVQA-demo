from collections import defaultdict
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
from scipy.stats import pearsonr, spearmanr
from datasets import get_avg_w2v, attention_refine_data
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt


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

def detach_output(o):
    return o.cpu().detach().numpy()

def calc_avg_imvstext(bert_attns, questions, atten_shape):
    # bert atten is 4 x 8 (or 12) x 115 x 115

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


################ Evaluation Functions ####################
# functions to calculate accuracies, correlations etc.
def calc_correlations(attn1, attn2, gt_a, answer):
    all_corr_correct = []
    all_corr_wrong = []
    eval_count=0
    for a1, a2, a, gt in zip(attn1, attn2, answer, gt_a):
            corr, p = spearmanr(a1, a2)
            if math.isnan(corr):
                #pdb.set_trace()
                #print("continue")
                continue
            if a==gt:
                all_corr_correct.append(corr)
            else:
                all_corr_wrong.append(corr)
            eval_count+=1
    #pdb.set_trace()
    #print("Num Eval : "+str(eval_count))
    return "Corr Correct: "+str(np.average(all_corr_correct))+" Corr Wrong: "+str(np.average(all_corr_wrong))

def calc_accuracy_soft(preds, labels):
    if type(preds) is torch.Tensor:
        preds = preds.cpu().detach().numpy()
    if type(labels) is torch.Tensor:
        labels = labels.cpu().detach().numpy()
    avg_prec = average_precision_score(labels, preds)
    best_acc = 0
    best_thresh = 0
    for thresh in np.arange(0.1,0.9,0.025):
        acc = accuracy_score(labels, preds>thresh)
        if acc >best_acc:
            best_acc=acc
            best_thresh=thresh
    
    norm_recall = (recall_score(labels, preds>best_thresh) + recall_score(labels<0.5, preds<best_thresh))/2.0
    
    return avg_prec, best_acc, best_thresh, norm_recall

def calc_accuracy_wrongpred(pred, answer, gt_a):
    wrong_labels = np.array(gt_a)!=np.array(answer)

    avg_prec, best_acc, best_thresh, norm_recall = calc_accuracy_soft(pred, wrong_labels)

    return "NormRecall: %.3f, AP: %.3f, BestAcc: %.3f, BestThresh: %.3f"%(norm_recall, avg_prec, best_acc, best_thresh)

def calc_accuracyclass_wrongpred(pred, answer, gt_a):
    wrong_labels = np.array(gt_a)!=np.array(answer)
    wrong_preds = torch.argmax(torch.stack(pred), axis=1).cpu().detach().numpy()

    return "Accuracy: %.4f"%(accuracy_score(wrong_labels, wrong_preds))


def calc_acc_correlation_histogram(attn1, attn2, gt_a, answer):
    if type(attn1) is torch.Tensor:
        attn1 = attn1.cpu().detach().numpy()
    if type(attn2) is torch.Tensor:
        attn2 = attn2.cpu().detach().numpy()

    corr_acc = defaultdict(list)
    for a1, a2, a, gt in zip(attn1, attn2, answer, gt_a):
            corr, p = spearmanr(a1, a2)
            if math.isnan(corr):
                #pdb.set_trace()
                #print("continue")
                continue
            bin_c = float("%.1f"%(corr))
            corr_acc[bin_c].append(a==gt)

    for key in corr_acc:
        corr_acc[key] = (np.average(corr_acc[key]), len(corr_acc[key]))

    #barplt = list(zip(*list(corr_acc.items())))
    #plt.bar(barplt[0], barplt[1])
    #plt.savefig("vis/corracchist.png")
    
    txt = str(sorted(list(corr_acc.items()), key=lambda x:x[0]))

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
        cam_corr, _ = spearmanr(cam, atten)

        if not math.isnan(cam_corr):
            corr_bin = float("%.1f"%(cam_corr)) #a terrible hack, but I dont know how else to do this.
            corr_dict[corr_bin].append((con_acc, vqa_acc))

    corr_dict_items = sorted(list(corr_dict.items()), key=lambda x:x[0])
    all_conaccs, all_vqaaccs = np.array([np.average(item, axis=0) for key, item in corr_dict_items]).transpose()
    corr_bins = sorted(list(corr_dict.keys()))

    #plot histogram
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

        correlation, p = spearmanr(a1, a2)

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