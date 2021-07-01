import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util.basic_layers as basic
import util.language
import nltk
#import nltk.word_tokenize

class new(nn.Module):
	def __init__(self,vqa_core,resnet,question_dictionary,answer_dictionary):
		super(new,self).__init__()
		self.resnet=resnet;
		self.vqa_core=vqa_core;
		self.question_dictionary=question_dictionary;
		self.answer_dictionary=answer_dictionary;
		return;
	
	def tokenize_questions(self,qs):
		tokenized_qs=[nltk.word_tokenize(q) for q in qs];
		qtokens,_=util.language.token2id(tokenized_qs,self.question_dictionary);
		return qtokens.long();
	
	def encode_image_center(self,I_normim):
		batch=I_normim.shape[0];
		Iembed=self.resnet(I_normim).view(batch,1,-1);
		return Iembed;
	
	def forward(self,I_normim,Q_text):
		Q=self.tokenize_questions(Q_text).cuda();
		I=self.encode_image_center(I_normim.cuda());
		pred=self.vqa_core(I,Q)
		return pred;
	
	def decode_answer(self,aid):
		answer=[self.answer_dictionary[i] for i in aid.tolist()];
		return answer;
	