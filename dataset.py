import numpy as np
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from input import input_stl

class MeshData(Dataset):

	def __init__(self,root_dir):
		self.classes = set()
		self.X = []
		self.Y = []
		self.final_dataset = []
		self.classes_codec = LabelEncoder()

		lst_of_classes = os.listdir(root_dir)
		lst_of_classes.sort()

		self.classes_codec.fit(lst_of_classes)

		for x in lst_of_classes:
			
			path = root_dir + '/' + x + '/train'
			lst_of_objects = os.listdir(path)
			for y in lst_of_objects:
				file_path = path + '/' + y
				
				dict_para = input_stl(file_path)
				if dict_para == False:
					print('Input file',y,'has problems')
					continue
				else:	
					neigh = dict_para["neigh_index"]
					corner = dict_para["corners"]
					center = dict_para["centroids"]
					normal = dict_para["normals"]

					self.X.append(np.concatenate((center,corner,normal,neigh),axis=1))
					self.Y.append((self.one_hot_encode(self.classes_codec,[x])))


		self.final_dataset = [self.X,self.Y]

	def one_hot_encode(self,codec,values):
		value_idxs = codec.transform(values)
		val,idx = torch.max(torch.eye(len(codec.classes_))[value_idxs],1)
		return torch.LongTensor(idx)



	def __len__(self):
		return len(final_dataset)

	def __getitem__(self,idx):
		return torch.from_numpy(self.X[idx]),self.Y[idx]


#DataObject = MeshData('/home/prathmesh/Desktop/SoC-2020/stl/b')
#print(len(DataObject.final_dataset))
#x,y = DataObject[0]
#print(x.shape)
#print(y)
#x,y = DataObject[1]
#print(x.shape)
#print(y)
#print(DataObject[1].shape)

#print(y.shape)
#print(DataObject[1].shape)
