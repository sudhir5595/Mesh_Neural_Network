import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from Model import Net
model = Net(10)  				#10 classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9)

from dataset import MeshData
DataObject = MeshData('/home/prathmesh/Desktop/SoC-2020/ModelNet10_stl/ModelNet10')
dataLoad = torch.utils.data.DataLoader(DataObject, batch_size=1, shuffle=True)
batch = next(iter(dataLoad))
print(len(batch))

max_epochs = 30
loss_list = []
for epochs in range(max_epochs):
	#print('e =',epochs)
	running_loss = 0.0
	for i,data in enumerate(dataLoad,0):
		x,y = data
		x = x[0].float().to(device)
		#y = y.float()
		y = y[0].to(device)
		optimizer.zero_grad()
		yhat = model(x)
		loss = criterion(yhat,y)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 200 == 199:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
			running_loss = 0.0
	#print(running_loss)
	import matplotlib.pyplot as plt
    	plt.plot(loss_list)
    	plt.title('Model_loss vs epochs')
    	plt.ylabel('Loss')
    	plt.xlabel('epochs')
    	s = '../working/epochwise_loss_' + str(epochs)
    	plt.savefig(s)
    	plt.show()
    	plt.close()
    	torch.save(model.state_dict(), '../working/new_mod.pth')

PATH = '/home/prathmesh/Desktop/Mesh_Neural_Network/new_models.pth'
torch.save(model.state_dict(), PATH)
