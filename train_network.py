import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim


from Outline import Net
model = Net(5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum = 0.9)


for epochs in range(10):
	for i,data in enumerate(dataset,0):
		x,y = data
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