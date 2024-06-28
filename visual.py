import matplotlib.pyplot as plt
import pickle
adam_file = open('Adam_accuracy_200_epochs_lr=0.001_resnet18_cifar10.pkl', 'rb')
inverse_adam001_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.01_resnet18_cifar10.pkl', 'rb')
inverse_adam0005_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.005_resnet18_cifar10.pkl', 'rb')
inverse_adam0001_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.001_resnet18_cifar10.pkl', 'rb')
inverse_adam00005_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0005_resnet18_cifar10.pkl', 'rb')
inverse_adam00001_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')
inverse_adam000005_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00005_resnet18_cifar10.pkl', 'rb')
inverse_adam000001_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00001_resnet18_cifar10.pkl', 'rb')
inverse_adam0000005_file = open('InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.000005_resnet18_cifar10.pkl', 'rb')
inverse_adamlr001_file = open('InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')
inverse_adamlr01_file = open('InverseAdam_accuracy_200_epochs_lr=0.1_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')
sgdm_file = open('SGDM_accuracy_200_epochs_lr=0.1_momentum=0.9_batchsize=128_weightdecay=5e-4_resnet18_cifar10.pkl', 'rb')

y_inverse_adam001_accuracy = pickle.load(inverse_adam001_file)[-100:]
y_adam_accuracy = pickle.load(adam_file)[-100:]
y_inverse_adam0005_accuracy = pickle.load(inverse_adam0005_file)[-100:]
y_inverse_adam0001_accuracy = pickle.load(inverse_adam0001_file)[-100:]
y_inverse_adam00005_accuracy = pickle.load(inverse_adam00005_file)[-100:]
y_inverse_adam00001_accuracy = pickle.load(inverse_adam00001_file)[-100:]
y_inverse_adam000005_accuracy = pickle.load(inverse_adam000005_file)[-100:]
y_inverse_adam000001_accuracy = pickle.load(inverse_adam000001_file)[-100:]
y_inverse_adam0000005_accuracy = pickle.load(inverse_adam0000005_file)[-100:]
y_inverse_adamlr001_accuracy = pickle.load(inverse_adamlr001_file)[-100:]
y_inverse_adamlr01_accuracy = pickle.load(inverse_adamlr01_file)[-100:]
y_sgdm_accuracy = pickle.load(sgdm_file)[-100:]



epoch = range(100, 200)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

plt.plot(epoch, y_adam_accuracy, linewidth=1, linestyle="solid", label="adam lr=0.001", color='red')
plt.plot(epoch, y_inverse_adam001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.01", color='black')
plt.plot(epoch, y_inverse_adam0005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.005", color='blue')
plt.plot(epoch, y_inverse_adam0001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.001", color='purple')
plt.plot(epoch, y_inverse_adam00005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.0005", color='orange')
plt.plot(epoch, y_inverse_adam00001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.0001", color='green')
plt.plot(epoch, y_inverse_adam000005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.00005", color='brown')
plt.plot(epoch, y_inverse_adam000001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.00001", color='grey')
plt.plot(epoch, y_inverse_adam0000005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam switch rate=0.000005", color='yellow')
plt.plot(epoch, y_inverse_adamlr001_accuracy, linewidth=2, linestyle="dotted", label="inverse_adam lr=0.01 switch rate=0.0001", color='red')
plt.plot(epoch, y_inverse_adamlr01_accuracy, linewidth=2, linestyle="dotted", label="inverse_adam lr=0.1 switch rate=0.0001", color='black')
plt.plot(epoch, y_sgdm_accuracy, linewidth=1, linestyle="solid", label="sgdm weight decay=5e-4", color='pink')




plt.legend()
plt.title('accuracy curve')
plt.show()

epoch = range(100, 200)

adam_loss_file = open('Adam_loss_200_epochs_lr=0.001_resnet18_cifar10.pkl', 'rb')
inverse_adam001_loss_file = open('InverseAdam_loss_200_epochs_lr=0.001_alpha=0.01_resnet18_cifar10.pkl', 'rb')
inverse_adam0005_loss_file = open('InverseAdam_loss_200_epochs_lr=0.001_alpha=0.005_resnet18_cifar10.pkl', 'rb')
inverse_adam0001_loss_file = open('InverseAdam_loss_200_epochs_lr=0.001_alpha=0.001_resnet18_cifar10.pkl', 'rb')

y_inverse_adam001_loss = pickle.load(inverse_adam001_loss_file)[-100:]
y_adam_loss = pickle.load(adam_loss_file)[-100:]
y_inverse_adam0005_loss = pickle.load(inverse_adam0005_loss_file)[-100:]
y_inverse_adam0001_loss = pickle.load(inverse_adam0001_loss_file)[-100:]

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('loss')     # y轴标签

plt.plot(epoch, y_adam_loss, linewidth=1, linestyle="solid", label="adam lr=0.001", color='red')
plt.plot(epoch, y_inverse_adam001_loss, linewidth=1, linestyle="solid", label="inverse_adam alpha=0.01", color='black')
plt.plot(epoch, y_inverse_adam0005_loss, linewidth=1, linestyle="solid", label="inverse_adam alpha=0.005", color='blue')
plt.plot(epoch, y_inverse_adam0001_loss, linewidth=1, linestyle="solid", label="inverse_adam alpha=0.001", color='purple')

plt.legend()
plt.title('loss curve')
plt.show()