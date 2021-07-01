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
import models.attention_refine.atten_refine_network as att_refine
import tqdm
import db as db
import pdb
from PIL import Image
import cv2
import random
import math
import eval
import datasets

from scripts.pytorchgradcam.gradcam import GradCam

curr = sys.modules[__name__]

def get_value(s):
    if "[" in s:
        dict_name = s.split("[")[0]
        key = s.split("[")[1].split("]")[0]
        try:
            return getattr(curr, dict_name)[key].squeeze()
        except:
            return getattr(curr, dict_name)[key]
    else:
        return getattr(curr, s)

def get_processed_value(entry, module):
    if entry[0]=='': # just store var name
        var_value = get_value(entry[1])
    else: #get value from function
        func = entry[0]
        f_args = entry[1]
        var_value = getattr(module, func)(*[get_value(a) for a in f_args])
    return var_value

def accumulate(model1, model2, decay=0):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


if __name__=="__main__":
    from experiment_configs import *
    print("configs loaded...")
    ####################################################################

    if not os.path.exists("checkpoints/"+exp_name):
        os.mkdir("checkpoints/"+exp_name)
    with open("checkpoints/"+exp_name+"/logs.txt", "a") as f:
        f.write("\n====================================\n")
        f.write(str([model_choice, model_init_args, train_dataset_choice, train_dataset_args, val_dataset_choice, val_dataset_args, input_choice,
                output_choice, losses_right, losses_wrong, other_losses, log_eval_vars]))
        f.write("\n")

    ################# load data #####################################
    atten_dataset = getattr(datasets, train_dataset_choice)(**train_dataset_args)
    w2v = atten_dataset.w2v
    atten_dataloader = DataLoader(atten_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if eval_only:
        if eval_all_checkpoints:
            atten_dataset_val = getattr(datasets, val_dataset_choice)(w2v=w2v, split="val_train", **val_dataset_args)    
        else:
            atten_dataset_val = getattr(datasets, val_dataset_choice)(w2v=w2v, split="val", **val_dataset_args)
    else:
        atten_dataset_val = getattr(datasets, val_dataset_choice)(w2v=w2v, split="val_train", **val_dataset_args)
    atten_dataloader_val = DataLoader(atten_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    print("data loaded...")
    #################################################################

    ############## define model, losses and checkpoints ################
    attention_refine_model = getattr(att_refine, model_choice)(**model_init_args).to(device)
    if 'CAM' in exp_name:
        attention_refine_model_cam = getattr(att_refine, model_choice)(ques_cam=True, **model_init_args).to(device)
        
    att_refine_optim = optim.Adam(attention_refine_model.parameters(), lr=0.00001, betas=(0.0, 0.99))
    if load_checkpoint and not eval_all_checkpoints:
        attention_refine_model.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+model_suffix))
        if 'CAM' in exp_name:
            attention_refine_model_cam.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+model_suffix))
        if not eval_only:
            att_refine_optim.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+optim_suffix))
    
    mse = lambda x,y,m: torch.sum(torch.sum((x - y)**2, dim=1)*m)/sum(m) if sum(m)!=0 else 0
    neg_mse = lambda x,y,m: -torch.sum(torch.sum((x - y)**2, dim=1)*m)/sum(m) if sum(m)!=0 else 0
    entropy = lambda x,m: (torch.sum(Categorical(probs = x).entropy()*m)/sum(m))/7.0 if sum(m)!=0 else 0
    neg_entropy = lambda x,m: (torch.sum(-(Categorical(probs = x).entropy()*m))/sum(m))/7.0 if sum(m)!=0 else 0
    crossentropyf = nn.CrossEntropyLoss()
    crossentropy = lambda p,l: crossentropyf(p, l.long())
    bce = nn.BCELoss()
    meanatt = lambda x, m: torch.sum(torch.mean(x**2, dim=1)*m)/sum(m) if sum(m)!=0 else 0

    gradcam = GradCam(attention_refine_model_cam)
    print("models loaded...")
    ###########################################################


    ###init bookkeeping lists
    saved_examples = []
    avg_loss_att = []
    avg_loss_wrong = []
    avg_loss_corr = []
    avg_loss_other = []
    #########################

    for ep in range(num_epochs):
        if eval_only:
            pbar = range(1)
        else:
            pbar = tqdm.tqdm(atten_dataloader)
        for i, entry in enumerate(pbar):
            if not eval_only:
                pbar.set_description('Ep:'+str(ep))
            
            if not eval_only:
                dataset_outs = entry
                
                outputs = attention_refine_model(*[dataset_outs[inp] for inp in input_choice])
                
                ################### loss and train #####################################

                ##### calculate losses ########
                mask_corr = torch.tensor((np.array(dataset_outs['gt_a'])==np.array(dataset_outs['answer']))*1).float().to(device)
                mask_wrong = torch.tensor(((np.array(dataset_outs['gt_a'])==np.array(dataset_outs['answer']))==False)*1).float().to(device)
                corr_labels = mask_corr
                wrong_labels = mask_wrong
                ans_labels = torch.argmax(dataset_outs['max_s'], dim=1).to(device)

                loss_att_corr = 0
                for l_entry in losses_right:
                    loss = getattr(curr, l_entry[0])
                    loss_args = [get_value(l_a) for l_a in l_entry[1]]
                    loss_att_corr += loss(*loss_args)
                
                loss_att_wrong = 0
                for l_entry in losses_wrong:
                    loss = getattr(curr, l_entry[0])
                    loss_args = [get_value(l_a) for l_a in l_entry[1]]
                    loss_att_wrong += loss(*loss_args)

                loss_other = 0
                for l_entry in other_losses:
                    loss = getattr(curr, l_entry[0])
                    loss_args = [get_value(l_a) for l_a in l_entry[1]]
                    loss_other += loss(*loss_args)
                
                
                loss_att = loss_att_corr*sum(mask_corr)/len(mask_corr) + loss_att_wrong*sum(mask_wrong)/len(mask_wrong) + loss_other

                if math.isnan(loss_att):
                    pdb.set_trace()

                ### average losses over a few iterations and print ###
                avg_loss_att.append(float(loss_att))
                if loss_att_corr!=0:
                    avg_loss_corr.append(float(loss_att_corr))
                if loss_att_wrong!=0:
                    avg_loss_wrong.append(float(loss_att_wrong))
                if loss_other!=0:
                    avg_loss_other.append(float(loss_other))

                if (i+1)%20==0:           
                    print("L_C:%.4f L_W:%.4f L_O:%.4f L:%.4f"%(np.average(avg_loss_corr), np.average(avg_loss_wrong), np.average(avg_loss_other), np.average(avg_loss_att)))
                    avg_loss_att = []
                    avg_loss_corr=[]
                    avg_loss_wrong=[]
                    avg_loss_other = []
                ####################################################

                ############### train ##############################
                att_refine_optim.zero_grad()
                attention_refine_model.zero_grad()
                loss_att.backward()
                att_refine_optim.step()
                if "CAM" in exp_name:
                    accumulate(attention_refine_model_cam, attention_refine_model)
                #####################################################
            
            
            ######### Validation #################################################
            if (i-1)%500==0 or eval_only:
                
                if eval_all_checkpoints:
                    #get all checkpoints
                    all_checkpoints = []
                    for files in os.listdir(os.path.join("checkpoints", exp_name)):
                        if "model_" in files:
                            all_checkpoints.append(files)
                    
                    for checkpoint in all_checkpoints:
                        eval_lists = defaultdict(list)
                        attention_refine_model.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+checkpoint))
                        if 'CAM' in exp_name:
                            attention_refine_model_cam.load_state_dict(torch.load("checkpoints/"+exp_name+"/"+checkpoint))
                        
                        for val_i, val_data_outs in enumerate(tqdm.tqdm(atten_dataloader_val)):
                            outputs = attention_refine_model(*[val_data_outs[inp] for inp in input_choice])

                            # process output and accumulate stuff required for visualization and evaluation.
                            for acc_entry in accumulate_eval_vars:
                                var_name = acc_entry[0]
                                eval_lists[var_name].extend(get_processed_value(acc_entry[1], eval))
                        
                        #run all required log routines in eval.py that log to text for printing or saving
                        log_text = "\n Epoch %d  Iter %d \n"%(ep, i)
                        for log_entry in log_eval_vars:
                            log_text += log_entry[0]+" : "
                            log_text += get_processed_value(log_entry[1], eval)
                            log_text+="\n"
                        print(exp_name)
                        print(checkpoint)
                        print(log_text)

                else:
                    eval_lists = defaultdict(list)
                    for val_i, val_data_outs in enumerate(tqdm.tqdm(atten_dataloader_val)):
                        outputs = attention_refine_model(*[val_data_outs[inp] for inp in input_choice])

                        # process output and accumulate stuff required for visualization and evaluation.
                        for acc_entry in accumulate_eval_vars:
                            var_name = acc_entry[0]
                            eval_lists[var_name].extend(get_processed_value(acc_entry[1], eval))
                    
                    #run all required log routines in eval.py that log to text for printing or saving
                    log_text = "\n Epoch %d  Iter %d \n"%(ep, i)
                    for log_entry in log_eval_vars:
                        log_text += log_entry[0]+" : "
                        log_text += get_processed_value(log_entry[1], eval)
                        log_text+="\n"

                    print(exp_name)
                    print(log_text)

                    #pdb.set_trace()
                    if not eval_only:
                        #torch.save(saved_examples, "checkpoints/"+exp_name+"/saved_examples.pt")
                        torch.save(attention_refine_model.state_dict(), "checkpoints/"+exp_name+"/model_"+str(ep)+"_"+str(i)+".pt")
                        torch.save(att_refine_optim.state_dict(), "checkpoints/"+exp_name+"/modeloptim_"+str(ep)+"_"+str(i)+".pt" )
                        with open("checkpoints/"+exp_name+"/logs.txt", "a") as f:
                            f.write(log_text)
                    else:
                        torch.save(eval_lists, "checkpoints/"+exp_name+"/saved_examples_val.pt")
