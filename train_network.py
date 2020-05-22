import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Model import Net
model = Net(10)  				#10 classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9)

from dataset import MeshData
DataObject = MeshData('ModelNet10')

max_epochs = 30
for epochs in range(max_epochs):
	for i in range(len(DataObject.final_dataset)):
		x,y = DataObject[i]
		optimizer.zero_grad()
		yhat = model(x)
		loss = criterion(y,yhat)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 200 == 199:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
			running_loss = 0.0

PATH = 'new_model.pth'
torch.save(net.state_dict(), PATH)
