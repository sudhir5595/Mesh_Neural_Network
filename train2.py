model = Net(10)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        
model.to(device)
max_epochs = 50
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.0008,betas=(0.9, 0.999))
loss_list = []
for epochs in range(max_epochs):
    #print('e =',epochs)
    running_loss = 0.0
    print(dataLoad)
    for i,data in enumerate(dataLoad,0):
        print(i)
        #print(data[0].shape)
        x,y = data
        x = x[0].float().to(device)
        #y = y.float()
        y = y[0].to(device)
        optimizer.zero_grad()
        yhat = model(x)
        print(yhat)
        #print(y)
        loss = criterion(yhat,y)
        print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epochs + 1, i + 1, running_loss / 200))
            loss_list.append(running_loss/200)
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
