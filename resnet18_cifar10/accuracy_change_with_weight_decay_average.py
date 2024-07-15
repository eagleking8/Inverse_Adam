import pickle
from utils import average_accuracy
import matplotlib.pyplot as plt

inverse_adam_wd0_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=0_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd5en3_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=5e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd7en3_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=7e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd9en3_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=9e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd9n9en3_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=9.9e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd9p5en3_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=9.5e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1p1en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1.1e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1p1p01en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1.01e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1p1p05en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1.05e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1p1p07en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1.07e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1p2en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1.2e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd2en2_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=2e-2_resnet18_cifar10.pkl', 'rb')

sgdm_file = open('./SGDM/SGDM_accuracy_200_epochs_lr=0.1_momentum=0.9_batchsize=128_weightdecay=5e-4_resnet18_cifar10.pkl', 'rb')

y_inverse_adam_wd0_accuracy = pickle.load(inverse_adam_wd0_file)
y_inverse_adam_wd5en3_accuracy = pickle.load(inverse_adam_wd5en3_file)
y_inverse_adam_wd7en3_accuracy = pickle.load(inverse_adam_wd7en3_file)
y_inverse_adam_wd9en3_accuracy = pickle.load(inverse_adam_wd9en3_file)
y_inverse_adam_wd9n9en3_accuracy = pickle.load(inverse_adam_wd9n9en3_file)
y_inverse_adam_wd9p5en3_accuracy = pickle.load(inverse_adam_wd9p5en3_file)
y_inverse_adam_wd1en2_accuracy = pickle.load(inverse_adam_wd1en2_file)
y_inverse_adam_wd1p01en2_accuracy = pickle.load(inverse_adam_wd1p1p01en2_file)
y_inverse_adam_wd1p05en2_accuracy = pickle.load(inverse_adam_wd1p1p05en2_file)
y_inverse_adam_wd1p07en2_accuracy = pickle.load(inverse_adam_wd1p1p07en2_file)
y_inverse_adam_wd1p1en2_accuracy = pickle.load(inverse_adam_wd1p1en2_file)
y_inverse_adam_wd1p2en2_accuracy = pickle.load(inverse_adam_wd1p2en2_file)
y_inverse_adam_wd2en2_accuracy = pickle.load(inverse_adam_wd2en2_file)

y_sgdm_accuracy = pickle.load(sgdm_file)


weight_decay_list = [0, 5e-3, 7e-3, 9e-3, 9.5e-3, 9.9e-3, 1e-2, 1.01e-2, 1.05e-2, 1.07e-2, 1.1e-2, 1.2e-2, 2e-2]

scale = 1e2

weight_decay_list = [i * scale for i in weight_decay_list]

accuracy_list = [average_accuracy(y_inverse_adam_wd0_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd5en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd7en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd9en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd9p5en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd9n9en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1p01en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1p05en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1p07en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1p1en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd1p2en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_wd2en2_accuracy, 10),
                ]

sgdm_x_list = [0, 2e-2]

sgdm_x_list = [i * scale for i in sgdm_x_list]

sgdm_accuracy_list = [average_accuracy(y_sgdm_accuracy, 10),
                      average_accuracy(y_sgdm_accuracy, 10),]

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('switch rate (1/' + str(scale).rstrip('0').rstrip('.') + ')')  # x轴标签
plt.ylabel('average accuracy(%)')  # y轴标签

plt.plot(weight_decay_list, accuracy_list, linewidth=1, linestyle="solid", color='red', marker='o',
         markeredgecolor='b', markersize='2')
plt.plot(sgdm_x_list, sgdm_accuracy_list, linewidth=1, linestyle="solid", color='blue', label='sgdm')

for x, y in zip(weight_decay_list, accuracy_list):
    plt.text(x, y + 0.002, str(x).rstrip('0').rstrip('.'), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()
