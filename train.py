import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
from torchvision import models
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters 超参数设置
num_epochs = 10
num_classes = 90
batch_size = 128    
learning_rate = 0.0002
transform1 = transforms.Compose([ # 数据归一化,三通道 均值和方差为0.5
                                 transforms.ToTensor(),                             
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])
# 按文件夹读取数据(每个文件夹下为一类)
train_dataset = torchvision.datasets.ImageFolder(root='./data/train/',
                                            transform=transform1,
                                            )


test_dataset = torchvision.datasets.ImageFolder(root='./data/test/',
                                            transform=transform1,
                                            )

# Data loader 加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, #shuffle将训练数据打乱
                                           drop_last=True, #drop_last为True会将多出来不足一个batch的数据丢弃
                                           )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#加载模型
model=models.squeezenet1_0(pretrained=True)
#读取参数
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth")) #加载预训练模型

model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=num_classes,
                                kernel_size=1)
model.num_classes = num_classes
model.cuda()
#model = SqueezeNet(1.0,90).to(device) #调用网络模型

criterion = nn.CrossEntropyLoss() #交叉熵


# Train the model 训练模型

#收集loss和正确率用于后面画图
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

total_step = len(train_loader)  # every epoch has how many step

for epoch in range(num_epochs):  # how many epoch
    model.train()
    if epoch ==12:
        learning_rate=learning_rate*0.1
    if epoch ==30:
        learning_rate=learning_rate*0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# 优化器
    for i, (images, labels) in enumerate(train_loader): # load data
        images = images.to(device)  # image
        labels = labels.to(device)   # label

        correct = 0
        # Forward pass
        outputs = model(images)   # outputs 输出
        
        loss = criterion(outputs, labels)  #计算损失函数
        _, predicted = torch.max(outputs.data, 1)  # 预测的值（返回每一行中最大值的那个元素的索引）

        # Backward and optimize 反向传播梯度下降优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (predicted == labels).sum().item()
        train_acc = correct / labels.size(0)
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, train_accuracy: {}'
                     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), train_acc))
    train_loss.append(loss.item())
    train_accuracy.append(train_acc)

    #验证集
    model.eval()
    with torch.no_grad(): #不计算梯度，节约内存
        correct = 0
        total = 0
        step_test = len(test_loader)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            te_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_acc = correct / total
        test_loss.append(te_loss.item())
        test_accuracy.append(test_acc)
    # every epoch save the model 保存模型(注意这里使用state_dict只保存网络参数)
    torch.save(model.state_dict(), './checkpoints/'+'model_'+str(epoch)+'.ckpt')
    

# matpolite 画图
x1 = range(0, len(train_loss))

plt.subplot(2, 1, 1)
plt.plot(x1, train_accuracy, 'b-')
plt.plot(x1, test_accuracy, 'r-')
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.legend(('train_acc', 'test_acc'), loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(x1, train_loss, 'b-')
plt.plot(x1, test_loss, 'r-')
plt.title('Loss')
plt.ylabel('loss')
plt.legend(('train_loss', 'test_loss'), loc='upper right')

plt.savefig('./result.png')
plt.show()









