import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, random
import numpy as np
from torch import optim

from AdaSGDM import AdaSGDM
from InverseAdam_AF import InverseAdam_AF
from InverseAdam_IF import InverseAdam_IF
from AR_InverseAdam import ARInverseAdam
from model import resnet, vgg

'''数据集准备'''


def load_data(dataset_name):
    trainloader = ""
    testloader = ""
    if dataset_name == "cifar10":
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # CIFAR-10 训练集
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        # CIFAR-10 测试集
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif dataset_name == "cifar100":
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # CIFAR-100 训练集
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        # CIFAR-100 测试集
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset_name == "tiny_imagenet":
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
        trainset = datasets.ImageFolder(root=os.path.join('./data/tiny_imagenet/tiny-imagenet-200', 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join('./data/tiny_imagenet/tiny-imagenet-200', 'val'), transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True)

    return trainloader, testloader


'''定义和训练测试函数'''


# 训练函数
def train(net, trainloader, optimizer, criterion, device="cuda"):
    net.train()
    running_loss = 0.0
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = net(data)  # 前向传播
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累加损失
    return running_loss / len(trainloader)  # 返回这个epoch的平均损失


# 测试函数
def test(net, testloader, device="cuda"):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = net(data)  # 前向传播
            _, predicted = torch.max(output.data, 1)  # 取预测值
            total += label.size(0)
            correct += predicted.eq(label).sum().item()  # 累加正确预测的数量
    return 100 * correct / total  # 返回准确率


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return total / (epoch_num + 0.0)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def select_model(model_name, device):
    if model_name == "resnet18":
        return resnet.ResNet18(num_classes=200).to(device)
    elif model_name == "resnet34":
        return resnet.ResNet34(num_classes=200).to(device)
    elif model_name == "vgg16":
        return vgg.VGG(vgg_name="VGG16", num_classes=100).to(device)


def select_optimizer(optimizer_name, model, lr):
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=0)
    elif optimizer_name == "InverseAdam":
        return InverseAdam_IF(params=model.parameters(), lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
                              switch_rate=8e-5, weight_decay=0)
    elif optimizer_name == "InverseAdam_AF":
        return InverseAdam_AF(params=model.parameters(), lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
                              switch_rate=2e-5, weight_decay=1e-2)
    elif optimizer_name == "SGDM":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=1e-2)
    elif optimizer_name == "RAdam":
        return optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-2, decoupled_weight_decay=True)
    elif optimizer_name == "NAdam":
        return optim.NAdam(model.parameters(), lr=lr, weight_decay=1e-2, decoupled_weight_decay=True, momentum_decay=4e-3)
    elif optimizer_name == "ARInverseAdam":
        return ARInverseAdam(model.parameters(), lr=lr, switch_rate=1e-6, weight_decay=1e-2, T=200*391)
    elif optimizer_name == "AdaSGDM":
        return AdaSGDM(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
