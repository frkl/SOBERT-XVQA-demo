import torch
import os
import cv2
import numpy as np
from writeToHTML import HTMLPage
import pdb
import tqdm
import sys
import math
from matplotlib import cm
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

def get_err_weight(p):
    weight = (p/0.175)**4  # empirically defined by what looks good on the matplotlib colormap.
    if weight>1:
        weight=1.0

    return weight

def make_attention_image(att_map, im_file, att_map_size=(14,14), weight=1):
    im = cv2.imread(im_file)
    #pdb.set_trace()
    processed_img = cv2.resize(im, (224,224))
    att_map = att_map.reshape(att_map_size)
    att_map = cv2.resize(att_map, (224,224)) 

    epsilon = 1e-3
    #att_map = (att_map - np.min(att_map) + epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
    
    att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
    alpha = 0.5
    output_image = (1 - alpha) * att_heatmap * weight + alpha * processed_img

    return output_image


def make_error_image(att_map, im_file, att_map_size=(14,14), weight=1):
    im = cv2.imread(im_file)
    #pdb.set_trace()
    processed_img = cv2.resize(im, (224,224))
    att_map = att_map.reshape(att_map_size)
    att_map = cv2.resize(att_map, (224,224)) 

    epsilon = 1e-3
    #att_map = (att_map - np.min(att_map) + epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
    
    att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
    alpha = 0.5
    output_image = (1 - alpha) * att_heatmap + alpha * processed_img

    return output_image


def make_bertattention_image(att_map, im_file, att_map_size=(14,14), weight=1):
    im = cv2.imread(im_file)
    #pdb.set_trace()
    processed_img = cv2.resize(im, (224,224))
    att_map = att_map.reshape(att_map_size)
    att_map = cv2.resize(att_map, (224,224)) 

    epsilon = 1e-3
    att_map = (att_map - np.min(att_map) + epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
    
    att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map * weight), cv2.COLORMAP_JET)
    alpha = 0.5
    output_image = (1 - alpha) * att_heatmap + alpha * processed_img

    return output_image

def RainBowColor(length, maxLength): #https://stackoverflow.com/questions/5137831/map-a-range-of-values-e-g-0-255-to-a-range-of-colours-e-g-rainbow-red-b
    i = (length * 255.0 / maxLength)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    return 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'

def make_question_attention(question, atten_weights):
    q_ws = question.lower().split("?")[0].split(" ")
    html_text = ""
    for q_w, atten in zip(q_ws, atten_weights):
        #r,g,b,al = cm.jet(atten)
        #r = int(r*255)
        #g = int(g*255)
        #b = int(b*255)
        #html_text+='<span style="color:'+'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'+';">'+q_w+'</span> '
        html_text+=q_w+" "
    return html_text


def save_attention_image(question, coco_id, refined_attn_im, prefix=""):
    q_format = "_".join(question.split("?")[0].lower().split(" "))
    atten_im_file = "vis/"+exp_name+"/attention_images/"+str(int(coco_id))+"_"+str(q_format)+"_"+str(prefix)+".png"
    cv2.imwrite(atten_im_file, refined_attn_im)
    attn_im_html_file = "attention_images/"+str(int(coco_id))+"_"+str(q_format)+"_"+str(prefix)+".png"
    return attn_im_html_file

def print_avg_attn(avg_attens):
    avg_word_atten, avg_im_atten, max_word_atten, max_im_atten = avg_attens
    return "Atten Avg Word: %.4f, Avg Img: %.4f, Max Word: %.4f, Max Img: %.4f"%(avg_word_atten, avg_im_atten, max_word_atten, max_im_atten)

def get_image(coco_id):
    return "/data/DataSets/COCO/"+split+"2014/COCO_"+split+"2014_"+str(int(coco_id)).zfill(12)+".jpg"

def get_online_im(coco_id):
    return 'https://vqa_mscoco_images.s3.amazonaws.com/'+split+'2014/COCO_'+split+'2014_'+str(int(coco_id)).zfill(12)+'.jpg'

def torchtofloat(s):
    return float(s.detach().cpu())

def torchtonumpy(t):
    return t.detach().cpu().numpy()

def get_corr_pred(p):
    try:
        return torchtofloat(p)>=0.175
    except:
        return p>=0.175
    
def get_class_pred(p):
    return np.argmax(torchtonumpy(p))


if __name__=='__main__':

    from experiment_configs import *
    split = "val"

    if not os.path.exists("vis/"+exp_name):
        os.mkdir("vis/"+exp_name)
        os.mkdir("vis/"+exp_name+"/attention_images")

    ########## format data to be a list of keyed entries #############
    eval_lists = torch.load("checkpoints/"+exp_name+"/saved_examples_val.pt")
    format_data = []
    keys = []
    for key in eval_lists:
        format_data.append(eval_lists[key][:100])
        keys.append(key) # just to make sure key are always aligned not sure if key order is randomized in python check this later. 

    format_data = list(zip(*format_data)) # transpose to make each entry a row of all the keys

    for i, entry in enumerate(format_data):
        entry_dict = dict()
        for k_i, k in enumerate(keys):
            entry_dict[k] = entry[k_i]
        format_data[i] = entry_dict
    ###################################################################
    
    ############ VISUALIZE TO HTML FILE ###############################
    vis_vars = dict()
    html = HTMLPage(filename = "vis/"+exp_name+"/attention_refine.html", title=exp_name)
    html.startTable()
    html.startRow()
    html.startCol()
    html.writeText("Index")
    for ops in visualize_ops:
        if ops[-1]!='':
            html.startCol()
            html.writeText(ops[0])

    save_indices = [] #optional

    for i, vis_entry in enumerate(format_data):
        html.startRow()
        html.startCol()
        html.writeText(str(i))
        for ops in visualize_ops:
            key, process_function, args, type = ops
            if process_function == '':
                vis_vars[key] = get_value(args)
            else:
                vis_vars[key] = getattr(curr, process_function)(*[get_value(a) for a in args])

            if type=='image':
                html.startCol()
                html.writeImage(vis_vars[key])

            if type=='text':
                html.startCol()
                html.writeText(vis_vars[key])

    html.endCol()
    html.endRow()
    html.closeHTMLFile()   

    
        






    

