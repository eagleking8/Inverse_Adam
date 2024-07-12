import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import load_data, train, test, seed_torch
from InverseAdam import InverseAdam
from model import resnet
import warmup_cosine_scheduler

if __name__ == '__main__':
    seed_torch(42)
    epoch_num = 200
    accuracies = []
    losses = []
    lr = 1e-2

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset_name = "cifar10"

    # 加载数据
    trainloader, testloader = load_data(dataset_name)

    model_name = "resnet18"
    model = ""

    if model_name == "resnet18":
        model = resnet.ResNet18(num_classes=10).to(device)
    elif model_name == "resnet34":
        model = resnet.ResNet34(num_classes=100).to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer_name = "InverseAdam"
    optimizer = ""

    # 选择并实例化优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optimizer_name == "InverseAdam":
        optimizer = InverseAdam(params=model.parameters(), lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                switch_rate=8e-5, weight_decay=1e-2)
    elif optimizer_name == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)

    # 实例化损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print(f"Dataset:{dataset_name} Model: {model_name} Optimizer:{optimizer_name}")

    # 加入学习率调度
    # def lambda_lr(epoch):
    #     if epoch < epoch_num / 10:
    #         return (epoch + 1) / epoch_num * 10
    #     else:
    #         return 0.1 ** (epoch // 80)
    # def lambda_lr(epoch): return 0.1 ** (epoch // 80)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr/1000, verbose=True)
    scheduler = warmup_cosine_scheduler.WarmUpCosineAnnealingLR(optimizer=optimizer, warmup_epochs=(epoch_num/10), max_epochs=epoch_num, min_lr=lr/1000)

    # 记录训练时间
    # start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(model, trainloader, optimizer, criterion, device=device)
        accuracy = test(model, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy} LR:{ scheduler.get_last_lr()[0]}")
    # training_time = time.time() - start_time
    # print(f"training time:{training_time}")

    # 保存准确率到文件
    with open('InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_warm_up_cosine1000_resnet18_cifar10.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('InverseAdam_loss_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_warm_up_cosine1000_resnet18_cifar10.pkl', 'wb') as file:
        pickle.dump(losses, file)
