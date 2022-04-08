
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size = 4

trainset     = torchvision.datasets.CIFAR10(root='./data' ,train=True,download=True,transform=transform)

trainloader   = torch.utils.data.DataLoader(trainset , batch_size=batch_size,shuffle=True , num_workers=2)

testset     = torchvision.datasets.CIFAR10(root='./data' ,train=True,download=True , transform=transform)

testloader =torch.utils.data.DataLoader(testset , batch_size=batch_size ,shuffle=False ,num_workers=2)

classes = ('plane' , 'car' ,'bird' , 'cat' , 'deer' ,'dog' ,'frog' ,'horse' ,'ship' , 'truck')



import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img= img/2 +0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
dataiter= iter(trainloader)
images , labels =dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(''.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):


    def __init__(self):
        super().__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*5*5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。                              
         
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
                     
    def forward(self,x):

        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

net=Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)






for epoch in range(10):
    
    running_loss= 0.0
    for i ,data in enumerate(trainloader, 0):
        
        inputs, labels =data
        
        optimizer.zero_grad()
        outputs=net(inputs)
        loss =criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 ==1999:
            print(f'[{epoch+1},{i+1:5d}] loss: {running_loss / 2000 :.3f}')
            running_loss =0.0
            
print('Finished Training')


PATH ='./clfar_net.pth'
torch.save(net.state_dict(),PATH)


net =Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)



dataiter =iter(testloader)
images , labels =dataiter.next()


imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ' ,''.join(f'{classes[labels[j]]:5s}' for j in range(4)))


correct_pred = {classname: 0 for classname in classes}
total_pred =  {classname: 0 for classname in classes}

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predictions =torch.max(outputs, 1)

    for label, prediction in zip(labels, predictions):
      if label ==prediction:
        correct_pred[classes[label]] += 1
      total_pred[classes[label]] += 1



for classname, correct_count in correct_pred.items():
  accuracy =100*float(correct_count) / total_pred[classname]
  print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

