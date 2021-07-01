import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt

import pdb

# a custom loss, not used
def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]


class attention_refine_net(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=128, q_feat_dim=300, im_feat_dim=36*2048, out_atten_dim=14*14):
        super().__init__()

        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        
        self.im_feat_in = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        
        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*3, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size,14*14), 
            nn.Softmax()
        )

    def forward(self, attention, im_feat, q_feat):

        att_feats = self.atten_input(attention)
        im_feats = self.im_feat_in(im_feat)
        q_feats = self.question_feat_in(q_feat)

        concat_feat = torch.cat((att_feats, im_feats, q_feats), dim=1)

        att_out = self.linear_block_attention_out(concat_feat)

        return {'refined_attn':att_out}


class attention_refine_net_ansconf(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=64, q_feat_dim=300, im_feat_dim=36*2048, ans_dim=3000, out_atten_dim=14*14):
        super().__init__()

        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        
        self.im_feat_in = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        
        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size,14*14), 
            nn.Softmax()
        )

        

    def forward(self, attention, im_feat, q_feat, ans_scores):

        att_feats = self.atten_input(attention)
        im_feats = self.im_feat_in(im_feat)
        q_feats = self.question_feat_in(q_feat)
        ans_score_feat = self.answer_score_in(ans_scores)

        concat_feat = torch.cat((att_feats, im_feats, q_feats, ans_score_feat), dim=1)

        att_out = self.linear_block_attention_out(concat_feat)

        return {'refined_attn':att_out}


class attention_refine_net_corrpred(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=64, q_feat_dim=300, im_feat_dim=36*2048, ans_dim=3000, out_atten_dim=14*14, att_out_nonlin="Softmax"):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )
        
        
        self.im_feat_in = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )

        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )

        self.concat2feat = nn.Sequential(
            nn.Linear(hidden_feat_size*8, hidden_feat_size*4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*4, hidden_feat_size*3),
            nn.LeakyReLU(0.1),
        )
        
        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*3, hidden_feat_size*3), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*3,14*14), 
            getattr(nn, att_out_nonlin)()
        )
            

        self.linear_block_corr_pred = nn.Sequential(
            nn.Linear(hidden_feat_size*3, 1),
            nn.Sigmoid()
        )

    def forward(self, attention, im_feat, q_feat, ans_scores):

        att_feats = self.atten_input(attention)
        im_feats = self.im_feat_in(im_feat)
        q_feats = self.question_feat_in(q_feat)
        ans_score_feat = self.answer_score_in(ans_scores)

        concat_feat = torch.cat((att_feats, im_feats, q_feats, ans_score_feat), dim=1)

        feats = self.concat2feat(concat_feat)

        refined_attention = self.linear_block_attention_out(feats)
        corr_pred = self.linear_block_corr_pred(feats)

        return {'refined_attn':refined_attention, 'corr_pred':corr_pred}


class attention_refine_net_anspred(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=64, q_feat_dim=300, im_feat_dim=36*2048, ans_dim=3000, out_atten_dim=14*14, att_out_nonlin="Softmax"):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )

        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size*2),
            nn.LeakyReLU(0.1)
        )

        self.concat2feat = nn.Sequential(
            nn.Linear(hidden_feat_size*8, hidden_feat_size*4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*4, hidden_feat_size*3),
            nn.LeakyReLU(0.1),
        )
        
        
        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*3, hidden_feat_size*3), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*3,14*14), 
            getattr(nn, att_out_nonlin)()
        )

        self.linear_block_ans_pred = nn.Linear(hidden_feat_size*3, ans_dim)

        self.att_out_nonlin = att_out_nonlin

    def forward(self, attention, im_feat, q_feat, ans_scores):

        att_feats = self.atten_input(attention)
        im_feats = self.im_feat_in(im_feat)
        q_feats = self.question_feat_in(q_feat)
        ans_score_feat = self.answer_score_in(ans_scores)

        concat_feat = torch.cat((att_feats, im_feats, q_feats, ans_score_feat), dim=1)

        feats = self.concat2feat(concat_feat)

        refined_attention = self.linear_block_attention_out(feats)
        ans_pred = self.linear_block_ans_pred(feats)

        return {'refined_attn':refined_attention, 'ans_pred':ans_pred}


class uncertain_attention_net_cam(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, num_class=1):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU()
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        if num_class==1:
            self.corr_pred_layer = nn.Sequential(
                nn.Linear(hidden_feat_size*4, num_class),
                nn.Sigmoid()
            )
        else:
            self.corr_pred_layer = nn.Sequential(
                nn.Linear(hidden_feat_size*4, hidden_feat_size),
                nn.Linear(hidden_feat_size, hidden_feat_size),
                nn.Linear(hidden_feat_size, hidden_feat_size//2),
                nn.Linear(hidden_feat_size//2, num_class)
            )

        self.gradients = []

    def save_gradients(self, grad):
        self.gradients = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        
        att_feats = self.atten_input(attention)
        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':[im_feats_feature]}

class uncertainatt_refinedatt_net_cam(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU()
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_feat_size, 1),
            nn.Sigmoid()
        )

        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size*3), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*3,hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size,7*7), 
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        
        att_feats = self.atten_input(attention)
        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        refined_attn = self.linear_block_attention_out(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'refined_attn': refined_attn, 'q_feats':q_feat}


class uncertainatt_refinedatt_net_cam_bigger(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size//2, 1),
            nn.Sigmoid()
        )

        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size*3), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*3,hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size,7*7), 
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        
        att_feats = self.atten_input(attention)

        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        refined_attn = self.linear_block_attention_out(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'refined_attn': refined_attn, 'q_feats':q_feat}


class uncertainatt_net_cam_bigger(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size//2, 1),
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        
        att_feats = self.atten_input(attention)

        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'q_feats':q_feat}


class uncertainatt_refinedatt_agnosticnet_cam_bigger(nn.Module):

    def __init__(self, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*3, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size//2, 1),
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):

        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'q_feats':q_feat}



class uncertainatt_refinedatt_net_cam_bigger_errorcondatt(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.im_feat_linear2 = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size//2, 1),
            nn.Sigmoid()
        )

        self.linear_block_attention_out = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size*3), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size*3,hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size,7*7), 
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.gradients_bert = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def save_gradients_bert(self, grad):
        self.gradients_bert = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        #if self.ques_cam:
        #    attention.register_hook(self.save_gradients_bert)
        att_feats = self.atten_input(attention)

        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        refined_attn = self.linear_block_attention_out(concat_feats)

        ref_att_format = refined_attn.view(-1, 7,7).unsqueeze(1).repeat(1,1024,1,1)
        
        im_feats_feature_weighted = im_feats_feature*ref_att_format

        im_feats_weighted = im_feats_feature_weighted.view(im_feats_feature_weighted.size(0), -1)

        im_feats_weighted = self.im_feat_linear2(im_feats_weighted)

        concat_feats_weighted = torch.cat((att_feats, im_feats_weighted, q_feats, ans_feat), dim=1)
        
        corr_pred = self.corr_pred_layer(concat_feats_weighted)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'refined_attn': refined_attn, 'q_feats':q_feat}



class uncertainatt_net_cam_bigger(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129, ques_cam=False):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*4, hidden_feat_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size, hidden_feat_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_feat_size//2, 1),
            nn.Sigmoid()
        )

        self.gradients = []
        self.gradients_qfeat = []
        self.ques_cam = ques_cam

    def save_gradients(self, grad):
        self.gradients = [grad]

    def save_gradients_qfeat(self, grad):
        self.gradients_qfeat = [grad]

    def forward(self, attention, im_feat, q_feat, ans_scores):
        
        att_feats = self.atten_input(attention)
        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)

        if self.ques_cam:
            q_feat.register_hook(self.save_gradients_qfeat)
        q_feats = self.question_feat_in(q_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':im_feats_feature, 'q_feats':q_feat}




class uncertain_attention_net_noans_cam(nn.Module):

    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048)):

        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
         
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)
        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU()
        )


        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*3, 1),
            nn.Sigmoid()
        )

        self.gradients = []

    def save_gradients(self, grad):
        self.gradients = [grad]

    def forward(self, attention, im_feat, q_feat):
        
        att_feats = self.atten_input(attention)
        im_feats_feature = self.im_feat_in(im_feat)

        im_feats_feature.register_hook(self.save_gradients)

        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)
        q_feats = self.question_feat_in(q_feat)

        concat_feats = torch.cat((att_feats, im_feats, q_feats), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':[im_feats_feature]}


class uncertain_attention_net(nn.Module):

    def __init__(self, hidden_feat_size=64, q_feat_dim=300, im_feat_dim=36*2048):

        super().__init__()

        
        self.im_feat_in = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        
        self.attention_layer = nn.Linear(hidden_feat_size*2, 7*7*2048)

        self.weighted_im_layer = nn.Sequential(
            nn.Linear(im_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )

        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*2, 1),
            nn.Sigmoid()
        )

    def forward(self, im_feat, q_feat):

        im_feats = self.im_feat_in(im_feat)
        q_feats = self.question_feat_in(q_feat)
        
        concat_feats = torch.cat((im_feats, q_feats), dim=1)

        attention = self.attention_layer(concat_feats)
        attention = attention.view(-1, 7*7,2048)
        attention = F.softmax(attention, dim=1)
        attention = attention.view(-1, 7*7*2048)

        weighted_im = attention*im_feat
        weighted_im_feats = self.weighted_im_layer(weighted_im)

        concat_weighted_feats = torch.cat((weighted_im_feats, q_feats), dim=1)

        corr_pred = self.corr_pred_layer(concat_weighted_feats)

        return {'wrong_pred':corr_pred, 'wrong_att':attention}


class quescaptionmatch_failurepred(nn.Module):
    def __init__(self, atten_dim = 4*12*115*115, hidden_feat_size=96, q_feat_dim=300, im_feat_dim=(7,7,2048), ans_dim=3129):
        super().__init__()
        self.atten_input = nn.Sequential(
            nn.Linear(atten_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )  
        self.im_feat_in = nn.Conv2d(im_feat_dim[-1], 1024, 1)

        self.im_feat_linear = nn.Sequential(
            nn.Linear(im_feat_dim[0]*im_feat_dim[1]*1024, hidden_feat_size),
            nn.LeakyReLU()
        )
        self.question_feat_in  = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        self.quescap_feat_in = nn.Sequential(
            nn.Linear(q_feat_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        ) 
        self.answer_score_in = nn.Sequential(
            nn.Linear(ans_dim, hidden_feat_size),
            nn.LeakyReLU(0.1)
        )
        self.corr_pred_layer = nn.Sequential(
            nn.Linear(hidden_feat_size*5, 1),
            nn.Sigmoid()
        )
        self.gradients = []
    
    def save_gradients(self, grad):
        self.gradients = [grad]

    def forward(self, attention, im_feat, q_feat, q_cap_feat, ans_scores):
        att_feats = self.atten_input(attention)
        im_feats_feature = self.im_feat_in(im_feat)
        im_feats_feature.register_hook(self.save_gradients)
        im_feats = im_feats_feature.view(im_feats_feature.size(0), -1)

        im_feats = self.im_feat_linear(im_feats)
        q_feats = self.question_feat_in(q_feat)
        q_cap_feats = self.quescap_feat_in(q_cap_feat)
        ans_feat = self.answer_score_in(ans_scores)

        concat_feats = torch.cat((att_feats, im_feats, q_feats, q_cap_feats, ans_feat), dim=1)

        corr_pred = self.corr_pred_layer(concat_feats)

        return {'wrong_pred':corr_pred, 'im_feature':[im_feats_feature]}

