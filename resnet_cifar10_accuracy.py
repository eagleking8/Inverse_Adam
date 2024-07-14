import pickle
import matplotlib.pyplot as plt

sgdm_file = open('./SGDM/SGDM_accuracy_200_epochs_lr=1e-1_momentum=0.9_wd=5e-4_cosine1000_resnet18_cifar10.pkl', 'rb')
adam_file = open('./adam/Adam_accuracy_200_epochs_lr=1e-2_wd=0_cosine1000_resnet18_cifar10.pkl', 'rb')
inverse_adam_if_file = open('./inverse_adam_lr1e-2_sr8e-5_wd1e-2/InverseAdam_accuracy_200_epochs_lr=1e-2_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
inverse_adam_af_file = open('./inverse_adam_af_lr1e-2_wd1e-2/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
adamw_file = open('./adamw/AdamW_accuracy_200_epochs_lr=1e-2_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')

y_sgdm = pickle.load(sgdm_file)[-100:]
y_adam = pickle.load(adam_file)[-100:]
y_inverse_adam_if = pickle.load(inverse_adam_if_file)[-100:]
y_inverse_adam_af = pickle.load(inverse_adam_af_file)[-100:]
y_adamw = pickle.load(adamw_file)[-100:]

epoch = range(100, 200)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

plt.plot(epoch, y_sgdm, linewidth=1, linestyle="solid", label="SGDM", color='green')
plt.plot(epoch, y_adam, linewidth=1, linestyle="solid", label="Adam", color='blue')
plt.plot(epoch, y_inverse_adam_if, linewidth=1, linestyle="solid", label="inverse adam if", color='black')
plt.plot(epoch, y_inverse_adam_af, linewidth=1, linestyle="solid", label="inverse adam af", color='purple')
plt.plot(epoch, y_adamw, linewidth=1, linestyle="solid", label="adamw", color='orange')

plt.legend()
plt.title('accuracy curve')
plt.show()