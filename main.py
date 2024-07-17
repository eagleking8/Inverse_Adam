import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
from InverseAdam_IF import InverseAdam_IF
from model import resnet
from WarmUpCosineAnnealingLR import WarmUpCosineAnnealingLR

if __name__ == '__main__':
    utils.seed_torch(42)

    epoch_num = 200
    accuracies = []
    losses = []
    lr = 1e-2

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset_name = "cifar100"

    # 加载数据
    trainloader, testloader = utils.load_data(dataset_name)

    model_name = "resnet18"
    model = utils.select_model(model_name, device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer_name = "InverseAdam_AF"
    optimizer = utils.select_optimizer(optimizer_name, model, lr)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr/1000, verbose=True)
    # scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, warmup_epochs=(epoch_num/10), max_epochs=epoch_num, min_lr=lr/100)

    # 记录训练时间
    # start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = utils.train(model, trainloader, optimizer, criterion, device=device)
        accuracy = utils.test(model, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy} LR:{ scheduler.get_last_lr()[0]}")
    # training_time = time.time() - start_time
    # print(f"training time:{training_time}")

    # 保存准确率到文件
    with open('InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr=1e-6_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('InverseAdam_AF_loss_200_epochs_lr=1e-2_sr=1e-6_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'wb') as file:
        pickle.dump(losses, file)
