import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torch.utils.data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)  # input:(1,28,28) output:(10,24,24)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)  # input:(10,12,12) output:(20,10,10)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 先经过conv1卷积再激活函数
        x = F.max_pool2d(x, 2, 2)  # 池化
        x = F.relu(self.conv2(x))  # 先经过conv2卷积再激活函数
        x = F.max_pool2d(x, 2, 2)  # 池化
        x = x.view(-1, 5 * 5 * 16)  # 重新调整Tensor的形状，-1相当于自动调整保持数据总体不变的情况下调整为50通道的4*4大小的特征图
        x = F.relu(self.fc1(x))  # 先经过fc1的全连接层再使用激活函数
        x = F.relu(self.fc2(x))  # 先经过fc2的全连接层再使用激活函数
        x = F.log_softmax(self.fc3(x), dim=1)  # 先经过fc3的全连接层，再通过softmax激活函数
        return x


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()  # 当有batch normalization 和 dropout 会启用他们
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 有GPU把数据送到GPU上运算
        optimizer.zero_grad()  # 梯度置零，也就是把loss关于weight的导数变成0，防止一个batch的梯度影响下一个batch
        output = model(data)  # 实例化的网络就是我们上面定义的网络
        loss = F.nll_loss(output, target)  # 调用损失函数
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 使用优化策略来反更新参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()  # 关闭 batch normalization 和 dropout
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


lr = 0.01
momentum = 0.5
batch_size = test_batch_size = 32
torch.manual_seed(53113)  # 随机种子
epochs = 3
# 使用GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
# 在训练模型时使用此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据，直至所有的数据都抛出。
# 定义好batch——size等参数，数据加载器会一次抛出一个batch的数据进行处理


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# 同理生成测试程序的数据加载器
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# 进行训练和测试
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
save_model = True
# 保存模型
if save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
net = Net()  # 实例化一个网络此时里面参数是随机的
img_path = r'.\test_number\8.jpg'
path = r'.\mnist_cnn.pt'
transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])
net.load_state_dict(torch.load(path))  # 把训练的参数放入其中
net = net.to(device)  # 把网络放到GPU上
torch.no_grad()
img = Image.open(img_path)
# 灰度化
img = img.convert('L')
img = np.asarray(img)
img_ = torch.from_numpy(img)
img_ = abs(255 - img_)

img_ = img_.unsqueeze(0)
img_ = img_.unsqueeze(0)
img_ = img_.to(device)
img_ = img_.float()
outputs = net(img_)
_, predicted = torch.max(outputs, 1)
print(predicted)