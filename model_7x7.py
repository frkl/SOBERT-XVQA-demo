"""
Hirerachical Modular Attention Networks 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from pytorch_transformers import cached_path # pytorch_transformers pytorch_pretrained_bert

from modeling import BertLayerNorm, BertLayer
import copy


class mlp(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers,p=0.5):
        super().__init__()
        self.layers=nn.ModuleList();
        self.dropout=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
            self.dropout.append(nn.Dropout(p));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            self.dropout.append(nn.Dropout(p));
            for i in range(1,nlayers-1):
                self.layers.append(nn.Linear(nh,nh));
                self.dropout.append(nn.Dropout(p));
            self.layers.append(nn.Linear(nh,noutput));
            self.dropout.append(nn.Dropout(p));
        return;
    #
    def forward(self,x):
        batch=x.shape[0];
        x=x.view(batch,-1);
        x=self.layers[0](x);
        for i in range(1,len(self.layers)):
            x=F.relu(x);
            x=self.dropout[i](x);
            x=self.layers[i](x);
        return x;


class simple_bert_encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        #config requires
        #   hidden_size recommend=768
        #   intermediate_size recommend=3072
        #   num_hidden_layers recommend=12
        #   num_attention_heads recommend=12
        #  I default this part of the config for good
        #   hidden_dropout_prob
        #   attention_probs_dropout_prob
        #   hidden_act
        #   layer_norm_eps
        config.layer_norm_eps=1e-12;
        config.hidden_dropout_prob=0.1;
        config.attention_probs_dropout_prob=0.1;
        config.hidden_act="gelu";
        
        self.layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        layer=BertLayer(config,output_attentions=True,keep_multihead_output=False)
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    
    def forward(self,x,avail):
        x=self.layernorm(x);
        x=self.dropout(x);
        #x: embeddings, one at each word
        #avail: used to mask out unused words
        h=[];
        att=[];
        for layer in self.layers:
            a,x=layer(x,avail);
            att.append(a);
            h.append(x);
        return h,att;

class simple_vqa_model(nn.Module):
    def __init__(self,config):
        super().__init__()
        #config requires
        #   hidden_size recommend=768
        #   intermediate_size recommend=3072
        #   num_hidden_layers recommend=12
        #   num_attention_heads recommend=12
        nhimage=config.nhimage;
        nh=config.nh;
        nhword=config.hidden_size;
        nwords=config.nwords;
        nlayers=config.nlayers;
        nanswers=config.nanswers;
        
        self.bert=simple_bert_encoder(config);
        
        self.it=mlp(nhimage,nh,nhword,nlayers);
        self.it2=mlp(nhimage,nh,nhword,nlayers);
        self.qembed=nn.Embedding(nwords+1,nhword,padding_idx=0);
        self.embed_type=nn.Embedding(3,nhword);
        self.embed_qloc=nn.Embedding(config.max_qlength,nhword);
        self.embed_iloc_7x7=nn.Embedding(49,nhword);
        self.vqa=mlp(nhword,nh,nanswers,2);
        return
    
    def forward(self,ifv,ifv_7x7,qtokens):
        #ifv: [batch, nbox, nhimage]
        #ibox: [batch, 7,7,nhimage]
        #qtokens: [batch,length]
        bsz=ifv.shape[0];
        nbox=ifv.shape[1];
        qlength=qtokens.shape[1];
        
        #Produce avail: a mask indicating availability of words/images
        qmask=qtokens.data.le(0).float()*(-1e10); #Annihilate attention to those words, so BERT skips them
        imask=qmask.sum(1,keepdim=True)*0;   #Hack to get around device issues
        imask=imask.repeat(1,nbox+49);
        avail=torch.cat((qmask,imask),dim=1);
        avail=avail.view(bsz,1,1,nbox+qlength+49)+avail.view(bsz,1,nbox+qlength+49,1);
        
        #Produce vtype: a mask embedding marking words=0/images=1
        vtype_q=self.embed_type.weight[0,:].view(1,1,-1).repeat(1,qlength,1);
        vtype_i=self.embed_type.weight[1,:].view(1,1,-1).repeat(1,nbox,1);
        vtype_i_7x7=self.embed_type.weight[2,:].view(1,1,-1).repeat(1,49,1);
        vtype=torch.cat((vtype_q,vtype_i,vtype_i_7x7),dim=1);
        
        #Produce location embeddings of words
        qloc=self.embed_qloc.weight[:qlength,:].view(1,qlength,-1);
        
        #Process word embeddings
        q=self.qembed(qtokens);
        
        #Process image embeddings for boxes
        ifv=ifv.view(bsz*nbox,-1);
        i=self.it(ifv);
        i=i.view(bsz,nbox,-1);
        
        #Process image embeddings for 7x7
        ifv_7x7=ifv_7x7.contiguous().view(bsz*49,-1);
        i_7x7=self.it2(ifv_7x7);
        i_7x7=i_7x7.view(bsz,49,-1);
        iloc_7x7=self.embed_iloc_7x7.weight.view(1,49,-1);
        
        #Form input to bert
        h=torch.cat((q+qloc,i,i_7x7+iloc_7x7),dim=1)+vtype;
        
        #Run through bert
        hs,atts=self.bert(h,avail);
        
        #Make VQA prediction
        fv=hs[-1][:,0,:];
        a=self.vqa(fv);
        return a,atts;


