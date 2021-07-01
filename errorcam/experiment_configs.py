import torch
import numpy as np
################ choices ###########
exp_name= "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger_recheck"
device = torch.device("cuda") #cuda
eval_only = True
eval_all_checkpoints=False
use_precomputed= True
load_checkpoint=False
#####################################

######### get architecture, training and loss choices #############
num_epochs = 4
num_vals = -1
batch_size=10


if exp_name == "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM":
    model_choice = 'uncertainatt_refinedatt_net_cam'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))]]


    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camcorrel_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camattncorr_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['baseline_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqacam_prefix="vqacamim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      ['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_cam_im', 'make_attention_image', ('vis_entry[vqa_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_answer_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_cam_im]', 'vqacam_prefix'), 'image'],
                      ['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]

    #eval only checkpoint
    model_suffix = "model_3_4001.pt"

if False: #exp_name == "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM": #evaluates consistency
    model_choice = 'uncertainatt_refinedatt_net_cam'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          'data_choice': 'convqa',
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['all_con_as', ('', 'val_data_outs[con_as]')],
                            ['all_con_gtas', ('', 'val_data_outs[con_gtas]')],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))]]


    log_eval_vars = [['cam_attn_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['consistency_predacc', ('calc_consistency_predacc', ('eval_lists[val_wrong_pred]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[all_con_as]', 'eval_lists[all_con_gtas]'))],
                     ['baseline_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camattn_consistent_hist', ('calc_camattncorr_consistency_hist', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[all_con_as]', 'eval_lists[all_con_gtas]', 'exp_name'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['im_html', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', '', 'vis_entry[question]', 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['cam_im_file', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refatt_im_file', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['hum_att_file', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['corr_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]


if exp_name == "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att', ('make_bestBERT_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            ['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     ['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     ['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     ['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint
    model_suffix = "model_2_5501.pt"

if exp_name == "exp4_actioncrippledmodel_corrpred_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'actioncrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'actioncrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att', ('make_bestBERT_attention_action', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            #['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            #['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            #['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))]
                            ]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_bestbertatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_bestbertattlayer_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_bestbertatthead_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestbertattlayer_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att_layer]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestbertatthead_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att_head]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertatthead_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertattlayer_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_humatt_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     ['baselineerrorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     ['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertlayeratthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_layer]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertheadatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_head]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_humanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertattlayer_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatthead_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     ['bestbertatt_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertattlayer_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_layer]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthead_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_head]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     #['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     #['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     #['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))],
                     #['quesatten_errorquescam_corrl', ('calc_correlations', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))],
                     #['strength_pred_refatts', ('calc_strength_acc', ('eval_lists[all_ref_attn]', 'eval_lists[all_cams]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['align_acc_refatts', ('calc_alignbased_acc', ('eval_lists[all_ref_attn]', 'eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['strength_pred_bestbert', ('calc_strength_acc', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['align_acc_bestbert', ('calc_alignbased_acc', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['baseline_centeredatt_corrs', ('calc_baseline_centeredatt_corrs', ('eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint
    model_suffix="model_3_4001.pt"

if exp_name == "exp4_actioncolorcombinedmodel_corrpred_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'combined', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'combined', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_att', ('make_bestBERT_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            ['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     ['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint


if exp_name == "exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger_recheck":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'cam':True,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att', ('make_bestBERT_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att_layer', ('make_bestBERT_attention_layeronly', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att_head', ('make_bestBERT_attention_headonly', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            ['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))],
                            ['qid', ('', 'val_data_outs[act_qid]')]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_bestbertatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_bestbertattlayer_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_bestbertatthead_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertattlayer_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att_layer]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertatthead_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att_head]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_bertatthead_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestberterr_bertattlayer_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_humatt_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     #['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     #['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     #['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['baselineerrorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertlayeratthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_layer]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertheadatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_head]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_humanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertattlayer_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_layer]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatthead_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att_head]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatt_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertattlayer_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_layer]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthead_errorcam_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att_head]', 'eval_lists[all_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     #['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     #['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))],
                     ['quesatten_errorquescam_corrl', ('calc_correlations', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))],
                     #['strength_pred_refatts', ('calc_strength_acc', ('eval_lists[all_ref_attn]', 'eval_lists[all_cams]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['align_acc_refatts', ('calc_alignbased_acc', ('eval_lists[all_ref_attn]', 'eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['strength_pred_bestbert', ('calc_strength_acc', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     #['align_acc_bestbert', ('calc_alignbased_acc', ('eval_lists[best_bert_att]', 'eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[answer]', 'eval_lists[gt_a]', 'eval_lists[val_wrong_pred]'))],
                     ['baseline_centeredatt_corrs', ('calc_baseline_centeredatt_corrs', ('eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    bestbert_prefix = 'bestbertatt'
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      ['err_weight', 'get_err_weight', ('vis_entry[val_wrong_pred]',), ''],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_error_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim', 'vis_vars[err_weight]'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['bestbert_im', 'make_bertattention_image', ('vis_entry[best_bert_att]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['bestbert_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[bestbert_im]', 'bestbert_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_bertattention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      ['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint
    model_suffix="model_3_2001.pt"
    


if exp_name == "exp4_crippledmodel_corrpredcondrefatt_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger_errorcondatt'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     #['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     ['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint

if exp_name == "exp4_fullsmallmodel_corrpred_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'fullsmall', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'fullsmall', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_att', ('make_bestBERT_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            ['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     ['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint


if exp_name == "exp4_fullmodel_corrpred_refinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,12,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'simple_bert_7x7_4', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'simple_bert_7x7_4', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['qid', ('', 'val_data_outs[act_qid]')],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['all_bert_ques', ('make_bert_wordatten', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['best_bert_att', ('make_bestBERT_attention_fullmodel', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            #['best_bert_err', ('make_bestBERT_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))],
                            ['all_bert_corrs_errorcam', ('log_all_bert_correlations_errorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams]'))],
                            ['all_quesbert_corrs_queserror', ('log_all_bertquesatten_correlations_queserrorcam', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'eval_lists[all_cams_ques]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1

    plot_camhuman = "plot_errorcam_human"
    plot_camatt = "plot_errorcam_refatt"
    plot_refatthuman = "plot_refatt_human"
    plot_baselinehuman = "plot_baseline_human"
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['bestbertatt_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_humatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['bestberterr_bertatt_corrl', ('calc_correlations', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camhuman'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_camatt'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_refatthuman'))],
                     #['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestbertatthumatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_att]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['bestberterr_bertatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[best_bert_err]', 'eval_lists[best_bert_att]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'plot_baselinehuman'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     ['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_errorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs_errorcam]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['best_queserrorbert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_quesbert_corrs_queserror]', 'eval_lists[answer]', 'eval_lists[gt_a]'))],
                     ['quesatten_errorquescam_acc_corrl', ('calc_acc_correlation_histogram', ('eval_lists[all_cams_ques]', 'eval_lists[all_bert_ques]', 'eval_lists[gt_a]', 'eval_lists[answer]',))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]  

    #eval only checkpoint
    model_suffix = "model_3_5501.pt"



if exp_name == "exp4_crippledmodel_corrpred_refinedattn_negwrong_uncertainCAM_bigger_pearson":
    model_choice = 'uncertainatt_refinedatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = [['neg_mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_wrong")],]
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('detach_output', ('outputs[wrong_pred]',))],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))],
                            ['all_bert_corrs', ('log_all_bert_correlations', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim', 'val_data_outs[human_att]'))]]
    if eval_only:
        best_thresh=0.175
    else:
        best_thresh=-1
    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'best_thresh'))],
                     #['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['errorcam_refatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     #['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camhumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['camrefatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatthumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baselinehumanatt_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camrefatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['camhumatt_qtype', ('question_type_attention_quality', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['refatthumatt_qtype', ('question_type_attention_quality', ('eval_lists[human_attn]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['baseline_qtype', ('question_type_attention_quality', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[question]'))],
                     #['best_bert_layerhead', ('calc_all_bert_correlations', ('eval_lists[all_bert_corrs]', 'eval_lists[answer]', 'eval_lists[gt_a]'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]     




if exp_name == "exp4_crippledmodel_corrpred_norefinedattn_uncertainCAM_bigger":
    model_choice = 'uncertainatt_net_cam_bigger'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred']
    losses_right = []
    losses_wrong = []
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('', 'outputs[wrong_pred]')],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')]]


    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_baselineatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_baselineatt_predbased_corrl', ('calc_predbased_correlations', ('eval_lists[all_cams]', 'eval_lists[all_vqa_attns]', 'eval_lists[gt_a]', 'eval_lists[answer]', 'eval_lists[val_wrong_pred]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqabaselineatt_prefix="vqabaselineattim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      #['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_baselineatt_im', 'make_attention_image', ('vis_entry[all_vqa_attns]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_baselineatt', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_baselineatt_im]', 'vqabaselineatt_prefix'), 'image'],
                      #['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      #['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]    



if exp_name == "exp4_crippledmodel_corrpred_norefwrong_refinedattn_uncertainCAM":
    model_choice = 'uncertainatt_refinedatt_net_cam'
    atten_dim = (4,8,115,115)
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": np.prod(atten_dim), "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          #'split':"val_train", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'colorcrippled', 
                          'im_feat_flatten':False,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'max_s']
    output_choice = ['wrong_pred', 'refined_attn']
    losses_right = [['mse', ("outputs[refined_attn]", "dataset_outs[human_att]", "mask_corr")],]
    losses_wrong = []
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')],]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('', 'outputs[wrong_pred]')],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['all_cams_ques', ('gen_word_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice', 'w2v'))],
                            ['vqa_cams', ('detach_output', ('val_data_outs[vqa_cam]',))],
                            ['all_vqa_attns', ('make_baseline_attention', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['avg_text_im_attn', ('calc_avg_imvstext', ('val_data_outs[attn]', 'val_data_outs[question]', 'atten_dim'))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')],
                            ['all_ref_attn', ('detach_output', ('outputs[refined_attn]',))]]


    log_eval_vars = [['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_humatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_refatt_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['errorcam_vqacam_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[vqa_cams]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['vqacam_humaatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['vqacam_refatt_corrl', ('calc_correlations', ('eval_lists[vqa_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['refatt_humatt_corrl', ('calc_correlations', ('eval_lists[all_ref_attn]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camcorrel_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['camattncorr_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['baseline_corr', ('calc_correlations', ('eval_lists[all_vqa_attns]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     #['baseline_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_vqa_attns]', 'eval_lists[all_ref_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))]
                     ]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    cam_prefix="cam"
    hum_prefix="human"
    refatt_prefix="refatt"
    vqacam_prefix="vqacamim"
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['image', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', 'make_question_attention', ('vis_entry[question]', 'vis_entry[all_cams_ques]'), 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      ['avg_attentions', 'print_avg_attn', ('vis_entry[avg_text_im_attn]',), 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['error_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]', 'cam_prefix'), 'image'],
                      ['refatt_im', 'make_attention_image', ('vis_entry[all_ref_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['refined_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[refatt_im]', 'refatt_prefix'), 'image'],
                      ['vqa_cam_im', 'make_attention_image', ('vis_entry[vqa_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['VQA_answer_cam', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[vqa_cam_im]', 'vqacam_prefix'), 'image'],
                      ['hum_att', 'make_attention_image', ('vis_entry[human_attn]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['human_attention', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]', 'hum_prefix'), 'image'],
                      ['failure_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]    



if exp_name == "exp4_quescapsim_corrpred_uncertainCAM":
    model_choice = 'quescaptionmatch_failurepred'
    atten_dim = 4*12*115*115
    model_init_args = {"im_feat_dim": (7,7,2048), "hidden_feat_size": 96, "atten_dim": atten_dim, "ans_dim":3129}
    train_dataset_choice = 'attention_refine_data'
    val_dataset_choice = 'attention_refine_data'
    train_dataset_args = {'data_val': "models/VQA/data_vqa_train.pt", 
                          'split':"train", 
                          'nonlinchoice':'sigmoid', 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'simple_bert_7x7_4', 
                          'im_feat_flatten':False,
                          'caption': True,
                          'device': device}
    val_dataset_args = {'data_val': "models/VQA/data_vqa_val.pt", 
                          'nonlinchoice':'sigmoid',
                          'num_vals': -1, 
                          'im_feature_choice':'spatial', 
                          'att_dim':(7,7), 
                          'model_choice':'simple_bert_7x7_4', 
                          'im_feat_flatten':False,
                          'caption': True,
                          'device': device}

    input_choice = ['attn', 'im_feature', 'ques_feats', 'ques_cap_feats', 'max_s']
    output_choice = ['wrong_pred']
    losses_right = []
    losses_wrong = []
    other_losses = [['bce', ('outputs[wrong_pred]', 'wrong_labels')]]

    #specifiy the quantities you want to accumulate in eval_lists[] dict as ['key', ('', 'variable_name_to_be_accumulated')] or
    # or ['key', ('postprocessfunction', ('args', 'of', 'function'))] 
    accumulate_eval_vars = [['val_wrong_pred', ('', 'outputs[wrong_pred]')],
                            ['gt_a', ('', 'val_data_outs[gt_a]')],
                            ['answer', ('', 'val_data_outs[answer]')],
                            ['all_cams', ('gen_cams', ('batch_size', 'gradcam', 'val_data_outs', 'input_choice'))],
                            ['attn', ('detach_output', ('val_data_outs[attn]',))],
                            ['human_attn', ('detach_output', ('val_data_outs[human_att]',))],
                            ['coco_id', ('', 'val_data_outs[coco_id]')],
                            ['question', ('', 'val_data_outs[question]')]]


    log_eval_vars = [['cam_hum_corrl', ('calc_correlations', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['corr_pred_acc', ('calc_accuracy_wrongpred', ('eval_lists[val_wrong_pred]', 'eval_lists[gt_a]', 'eval_lists[answer]'))],
                     ['correl_acc_hist', ('calc_acc_correlation_histogram', ('eval_lists[all_cams]', 'eval_lists[human_attn]', 'eval_lists[gt_a]', 'eval_lists[answer]'))]]

    # save to key in vis_vars[], processing function, function args (keys in eval_lists are saved in vis_entry[]), image/text out
    att_dim = (7,7)
    visualize_ops = [['im_file', 'get_image', ('vis_entry[coco_id]',), ''],
                      ['im_html', 'get_online_im', ('vis_entry[coco_id]',), 'image'],
                      ['question', '', 'vis_entry[question]', 'text'],
                      ['answer', '', 'vis_entry[answer]', 'text'],
                      ['gt_a', '', 'vis_entry[gt_a]', 'text'],
                      ['cam_im', 'make_attention_image', ('vis_entry[all_cams]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['cam_im_file', 'save_attention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[cam_im]'), 'image'],
                      ['hum_att_np', 'torchtonumpy', ('vis_entry[human_attn]',), ''],
                      ['hum_att', 'make_attention_image', ('vis_vars[hum_att_np]', 'vis_vars[im_file]', 'att_dim'), ''],
                      ['hum_att_file', 'save_humanattention_image', ('vis_entry[question]', 'vis_entry[coco_id]', 'vis_vars[hum_att]'), 'image'],
                      ['corr_pred', 'get_corr_pred', ('vis_entry[val_wrong_pred]',), 'text']]



if eval_only:
    num_epochs=1
    num_vals=-1
    load_checkpoint = True

  
