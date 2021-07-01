import collections

class new:
	def __init__(self, capacity):
		self.dic = collections.OrderedDict()
		self.remain = capacity

	def __getitem__(self, key):
		if key not in self.dic:
			raise('Key Error')
			return -1;
		v = self.dic.pop(key) 
		self.dic[key] = v   # set key as the newest one
		return v
	
	def __contains__(self, key):
		if key in self.dic:
			return True;
		else:
			return False;
	
	def __setitem__(self, key, value):
		if key in self.dic:    
			self.dic.pop(key)
		else:
			if self.remain > 0:
				self.remain -= 1  
			else:  # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value
		return;