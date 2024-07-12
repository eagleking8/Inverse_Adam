import pickle
from utils import average_accuracy
import matplotlib.pyplot as plt

inverse_adam_wd0_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=0_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en10_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-10_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd5en10_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=5e-10_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd2en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=2e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd3en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=3e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd5en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=5e-9_resnet18_cifar10.pkl', 'rb')

y_inverse_adam_wd0_accuracy = pickle.load(inverse_adam_wd0_file)
y_inverse_adam_wd1en10_accuracy = pickle.load(inverse_adam_wd1en10_file)
y_inverse_adam_wd5en10_accuracy = pickle.load(inverse_adam_wd5en10_file)
y_inverse_adam_wd1en9_accuracy = pickle.load(inverse_adam_wd1en9_file)
y_inverse_adam_wd2en9_accuracy = pickle.load(inverse_adam_wd2en9_file)
y_inverse_adam_wd3en9_accuracy = pickle.load(inverse_adam_wd3en9_file)
y_inverse_adam_wd5en9_accuracy = pickle.load(inverse_adam_wd5en9_file)

weight_decay_list = [0, 1e-10, 5e-10, 1e-9, 2e-9, 3e-9, 5e-9]

accuracy_list = [average_accuracy(y_inverse_adam_wd0_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1en10_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd5en10_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1en9_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd2en9_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd3en9_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd5en9_accuracy, 10),]


plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('switch rate')  # x轴标签
plt.ylabel('average accuracy(%)')  # y轴标签

plt.plot(weight_decay_list, accuracy_list, linewidth=1, linestyle="solid", color='red', marker='o',
         markeredgecolor='b', markersize='2')

for x, y in zip(weight_decay_list, accuracy_list):
    plt.text(x, y + 0.002, str(x).rstrip('0').rstrip('.'), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()
