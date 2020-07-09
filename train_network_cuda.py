import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from Model import Net
model = Net(10)  				#10 classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9)

if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()

#from dataset import MeshData
#DataObject = MeshData(path)
st0 = time.time()
DataObject_X = np.load('x10.npy', allow_pickle=True)
DataObject_Y = np.load('y10.npy', allow_pickle=True)
st1 = time.time()
print(st1 - st0)

max_epochs = 30
for epochs in range(max_epochs):
	#print('e =',epochs)z
	running_loss = 0.0
	st2 = time.time()
	for i in range(len(DataObject_X)):
		#x,y = DataObject[i]
		x,y = torch.from_numpy(DataObject_X[i]), torch.tensor([DataObject_Y[i]])
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		#print(y) 
		x = x.float()
		
		#y = y.float()
		optimizer.zero_grad()
		yhat = model(x)
		loss = criterion(yhat,y)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 200 == 199:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
			running_loss = 0.0
		st3 = time.time()
	print(st3 - st2)
	print(epochs, running_loss)

PATH = '/home/prathmesh/Desktop/Mesh_Neural_Network/new_models.pth'
torch.save(model.state_dict(), PATH)
