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
			
			path = root_dir + '/' + x + '/train/'
			lst_of_objects = os.listdir(path)
			for y in lst_of_objects:
				file_path = path + '/' + y

				dict_para = input_stl(file_path)
				neigh = dict_para["neigh_index"]
				corner = dict_para["corners"]
				center = dict_para["centroids"]
				normal = dict_para["normals"]

				self.X.append((center,corner,normal,neigh))
				self.Y.append((self.one_hot_encode(self.classes_codec,[x])))


		self.final_dataset = [self.X,self.Y]

	def one_hot_encode(self,codec,values):
		value_idxs = codec.transform(values)
		return torch.eye(len(codec.classes_))[value_idxs]



	def __len__(self):
		return len(final_dataset)

	def __getitem__(self,idx):
		return torch.tensor(final_dataset[idx])



DataObject = MeshData('ModelNet10')