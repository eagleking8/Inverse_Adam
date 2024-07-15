import pickle
import matplotlib.pyplot as plt

inverse_adamlr001_file = open('InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')

inverse_adam_lr001_l25en9_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l29en9_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=9e-9_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1e-8_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21p1en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1.1e-8_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21p2en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1.2e-8_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21p5en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1.5e-8_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l22en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=2e-8_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l25en8_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-8_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1e-7_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l21p2en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=1.2e-7_warmup_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr001_l22en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=2e-7_warmup_resnet18_cifar10.pkl', 'rb')
# inverse_adam_lr001_l25en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-7_resnet18_cifar10.pkl', 'rb')
# inverse_adam_lr001_l25en6_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-6_warmup_resnet18_cifar10.pkl', 'rb')

y_inverse_adamlr001_accuracy = pickle.load(inverse_adamlr001_file)

y_inverse_adam_lr001_l25en9_accuracy = pickle.load(inverse_adam_lr001_l25en9_file)
y_inverse_adam_lr001_l29en9_accuracy = pickle.load(inverse_adam_lr001_l29en9_file)
y_inverse_adam_lr001_l21en8_accuracy = pickle.load(inverse_adam_lr001_l21en8_file)
y_inverse_adam_lr001_l21p1en8_accuracy = pickle.load(inverse_adam_lr001_l21p1en8_file)
y_inverse_adam_lr001_l21p2en8_accuracy = pickle.load(inverse_adam_lr001_l21p2en8_file)
y_inverse_adam_lr001_l21p5en8_accuracy = pickle.load(inverse_adam_lr001_l21p5en8_file)
y_inverse_adam_lr001_l22en8_accuracy = pickle.load(inverse_adam_lr001_l22en8_file)
y_inverse_adam_lr001_l25en8_accuracy = pickle.load(inverse_adam_lr001_l25en8_file)
y_inverse_adam_lr001_l21en7_accuracy = pickle.load(inverse_adam_lr001_l21en7_file)
y_inverse_adam_lr001_l21p2en7_accuracy = pickle.load(inverse_adam_lr001_l21p2en7_file)
y_inverse_adam_lr001_l22en7_accuracy = pickle.load(inverse_adam_lr001_l22en7_file)
# y_inverse_adam_lr001_l25en7_accuracy = pickle.load(inverse_adam_lr001_l25en7_file)
# y_inverse_adam_lr001_l25en6_accuracy = pickle.load(inverse_adam_lr001_l25en6_file)


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return  total / (epoch_num + 0.0)

l2_coefficient_list = [5e-9, 9e-9, 1e-8, 1.1e-8, 1.2e-8, 1.5e-8, 2e-8, 5e-8, 1e-7, 1.2e-7, 2e-7]

accuracy_list = [average_accuracy(y_inverse_adam_lr001_l25en9_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l29en9_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21p1en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21p2en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21p5en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l22en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l25en8_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21en7_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l21p2en7_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr001_l22en7_accuracy, 10),
                ]

l2_coefficient_list = [i * 1e7 for i in l2_coefficient_list]

vanilla_inverse_adam_accuracy = [average_accuracy(y_inverse_adamlr001_accuracy, 10),
                                 average_accuracy(y_inverse_adamlr001_accuracy, 10),
                                 average_accuracy(y_inverse_adamlr001_accuracy, 10)]
vanilla_inverse_adam_x = [5e-9, 9e-9, 2e-7]
vanilla_inverse_adam_x = [i * 1e7 for i in vanilla_inverse_adam_x]

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('l2 coefficient(1e-7)')    # x轴标签
plt.ylabel('average accuracy(%)')     # y轴标签

# plt.vlines(l2_coefficient_list, 1e-6, accuracy_list, linestyle="dashed")
# plt.hlines(accuracy_list, 0, l2_coefficient_list, linestyle="dashed")

plt.plot(l2_coefficient_list, accuracy_list, linewidth=1, linestyle="solid", label="lr=0.01 switch rate=0.0001 warm up", color='red')
plt.plot(vanilla_inverse_adam_x, vanilla_inverse_adam_accuracy, linewidth=1, linestyle="solid", label="vanilla inverse adam", color='blue')


for x, y in zip(l2_coefficient_list, accuracy_list):
    plt.text(x, y+0.002, '%1f' % x, horizontalalignment='center', verticalalignment='bottom', fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()