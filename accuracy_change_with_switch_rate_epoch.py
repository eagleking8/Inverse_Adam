import matplotlib.pyplot as plt
import pickle
adam_file = open('./Adam/Adam_accuracy_200_epochs_lr=0.001_resnet18_cifar10.pkl', 'rb')
optimal_inverse_adam_file = open(
    './inverse_adam_sr0.0001/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')

inverse_adam_sr0_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr0000005_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.000005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr000001_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr000005_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr00001_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr00005_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr0001_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr0005_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr001_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.01_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en2_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=6e-4_resnet18_cifar10.pkl', 'rb')

sgdm_file = open('./SGDM/SGDM_accuracy_200_epochs_lr=0.1_momentum=0.9_batchsize=128_weightdecay=5e-4_resnet18_cifar10.pkl', 'rb')

y_adam_accuracy = pickle.load(adam_file)[-100:]

y_optimal_inverse_adam_accuracy = pickle.load(optimal_inverse_adam_file)[-100:]

y_inverse_adam_sr0_accuracy = pickle.load(inverse_adam_sr0_file)[-100:]
y_inverse_adam_sr0000005_accuracy = pickle.load(inverse_adam_sr0000005_file)[-100:]
y_inverse_adam_sr000001_accuracy = pickle.load(inverse_adam_sr000001_file)[-100:]
y_inverse_adam_sr000005_accuracy = pickle.load(inverse_adam_sr000005_file)[-100:]
y_inverse_adam_sr00001_accuracy = pickle.load(inverse_adam_sr00001_file)[-100:]
y_inverse_adam_sr00005_accuracy = pickle.load(inverse_adam_sr00005_file)[-100:]
y_inverse_adam_sr0001_accuracy = pickle.load(inverse_adam_sr0001_file)[-100:]
y_inverse_adam_sr0005_wd00000001_accuracy = pickle.load(inverse_adam_sr0005_file)[-100:]
y_inverse_adam_sr001_wd000000005_accuracy = pickle.load(inverse_adam_sr001_file)[-100:]
y_inverse_adam_lr1en2_accuracy = pickle.load(inverse_adam_lr1en2_file)[-100:]

y_sgdm_accuracy = pickle.load(sgdm_file)[-100:]

epoch = range(100, 200)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

# plt.plot(epoch, y_adam_accuracy, linewidth=1, linestyle="solid", label="adam lr=0.001", color='red')
# plt.plot(epoch, y_inverse_adam00001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.0001", color='green')
plt.plot(epoch, y_adam_accuracy, linewidth=1, linestyle="dotted", label="adam lr=0.001", color='red')
# plt.plot(epoch, y_inverse_adam_lr001_wd000001_accuracy, linewidth=2, linestyle="dotted", label="inverse_adam lr=0.01 switch rate=0.0001 wd=0.00001", color='green')
# plt.plot(epoch, y_inverse_adam_sr0_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0", color='purple')
plt.plot(epoch, y_inverse_adam_sr0000005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.000005", color='black')
plt.plot(epoch, y_inverse_adam_sr000001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.00001", color='blue')
plt.plot(epoch, y_inverse_adam_sr000005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.00005", color='orange')
plt.plot(epoch, y_inverse_adam_sr00001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.0001", color='pink')
plt.plot(epoch, y_inverse_adam_sr00005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.0005", color='brown')
plt.plot(epoch, y_inverse_adam_sr0001_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.001", color='green')
plt.plot(epoch, y_inverse_adam_sr001_wd000000005_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=0.001 switch rate=0.01", color='grey')
plt.plot(epoch, y_inverse_adam_lr1en2_accuracy, linewidth=2, linestyle="dotted", label="inverse_adam lr=0.01 switch rate=8e-5", color='grey')
plt.plot(epoch, y_sgdm_accuracy, linewidth=1, linestyle="dotted", label="sgdm weight decay=5e-4", color='pink')
plt.plot(epoch, y_optimal_inverse_adam_accuracy, linewidth=2, linestyle="dotted", label="optimal inverse_adam", color='green')




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