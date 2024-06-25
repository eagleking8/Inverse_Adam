import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import load_data,train,test
from InverseAdam import InverseAdam
from model import resnet


if __name__ == '__main__':
    epoch_num = 200
    accuracies = []
    losses = []

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

    optimizer_name = "Adam"
    optimizer = ""

    # 选择并实例化优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    elif optimizer_name == "InverseAdam":
        optimizer = InverseAdam(params=model.parameters(), lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.01)
    elif optimizer_name == "SGDM":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=False)


    # 实例化损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 加入学习率调度
    def lambda_lr(epoch): return 0.1 ** (epoch // 80)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # 记录训练时间
    # start_time = time.time()
    #进行训练和测试
    for epoch in range(epoch_num):
        loss = train(model, trainloader, optimizer, criterion, device=device)
        accuracy = test(model, testloader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss: {loss}, Accuracy:{accuracy}")
    # adam_time = time.time() - start_time
    # print(f"adai_time:{adam_time}")

    # 保存准确率到文件
    with open('Adam_accuracy_200_epochs_lr=0.001.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # 保存损失到文件
    with open('Adam_loss_200_epochs_lr=0.001.pkl', 'wb') as file:
        pickle.dump(losses, file)

    # torch.save(net, 'adai_model')