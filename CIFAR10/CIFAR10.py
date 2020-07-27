# encoding = utf-8

import torch
import torch.nn as nn
#import torch.utils.data.DataLoader as DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #input channel, output channel, kernel
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.liner1 = nn.Linear(16*5*5,120)
        self.liner2 = nn.Linear(120,84)
        self.liner3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        x = self.liner3(x)
        return x

def fit(batchSize=4,epochs = 2):

    # load data
    # 50000*32*32*3
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10('./data/', train=True, download= True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

    # model
    net= Net()
    net = net.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # fit
    for epoch in range(epochs):

        runningLoss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the optimizer gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            if i % 1000 == 999:
                print('[epoch: %d, batch: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, runningLoss / 2000))
                runningLoss = 0.0
    print('Finished Training')
    path = './cifar_net.pth'
    torch.save(net.state_dict(), path)
    return

def accuarcy(batchSize = 4):
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10('./data/', train=False,download= True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

    #load model
    net = Net()
    net = net.to(device)
    path = './cifar_net.pth'
    net.load_state_dict(torch.load(path))

    #test
    classCorrect = list(0. for i in range(10))
    classTotal = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            # dimension == 1 , predicted is  a indice
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(c.size()[0]):
                label = labels[i]
                classCorrect[label] += c[i].item()
                classTotal[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i],100*classCorrect[i]/classTotal[i]))
    return

def predict(imgPath):

    #read img
    img = cv2.imread(imgPath)
    img = Image.fromarray(img)
    transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    img =transform(img)
    img = img.to(device)
    # single figure has 3 dimensions, use torch.unsqueeze(0) expand(1,1,32,32)
    img = img.unsqueeze(0)

    # predict picture
    net = Net()
    net = net.to(device)
    path = './cifar_net.pth'
    net.load_state_dict(torch.load(path))

    output = net(img)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    batchSize = 4
    epochs = 50
    # fit(batchSize,epochs)
    # accuarcy(batchSize)
    label = predict('./pre/car2.jpg')
    print(label)

