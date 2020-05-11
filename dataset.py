import torch
import torch.utils.data.dataset


class MeshData(dataset):

	def __init__(self,root_dir):

		lst_of_classes = os.listdir(root_dir)
		lst_of_classes.sort()

		for x in lst_of_classes:
			path = root_dir + '/' + x
			lst_of_objects = os.listdir(path)
			for y in lst_of_objects:
				file_path = path + y

				dict_para = input_stl(file_path)
				



	def __len__(self):
		return len(final_dataset)

	def __getitem__(self,idx):
		return final_dataset[idx]

