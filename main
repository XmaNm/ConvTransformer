from model import *

net = Transformer()
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons)))
epochs = 2

# running_loss to record the losses
running_loss = 0.0

for epoch in range(epochs):
    for i,data in enumerate(trainloader,0):
        # get input images and their labels
        inputs, labels = data
        # set optimizer buffer to 0
        optimizer.zero_grad()
        # forwarding
        outputs = net(inputs)
        # computing loss
        loss = loss_function(outputs, labels)
        # loss backward
        loss.backward()
        # update parameters using optimizer
        optimizer.step()

        # printing some information
        running_loss += loss.item()
        # for every 1000 mini-batches, print the loss
        if i % 1000 == 0:
            print("epoch {} - iteration {}: average loss {:.3f}".format(epoch+1, i+1, running_loss/1000))
            running_loss = 0.0


print("Training Finished!")
