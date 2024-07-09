import pickle
import matplotlib.pyplot as plt
optimal_inverse_adam_file = open('./inverse_adam_sr0.0001/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')

inverse_adam_lr1en4_file = open('./inverse_adam_sr8en5/InverseAdam_accuracy_200_epochs_lr=1e-4_switchrate=8e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en3_file = open('./inverse_adam_sr8en5/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=8e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en2_file = open('./inverse_adam_sr8en5/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=8e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en1_file = open('./inverse_adam_sr8en5/InverseAdam_accuracy_200_epochs_lr=0.1_switchrate=8e-5_resnet18_cifar10.pkl', 'rb')

# inverse_adam_lr001_l25en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-7_resnet18_cifar10.pkl', 'rb')
# inverse_adam_lr001_l25en6_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-6_warmup_resnet18_cifar10.pkl', 'rb')

y_optimal_inverse_adam_accuracy = pickle.load(optimal_inverse_adam_file)

y_inverse_adam_lr1en4_accuracy = pickle.load(inverse_adam_lr1en4_file)
y_inverse_adam_lr1en3_accuracy = pickle.load(inverse_adam_lr1en3_file)
y_inverse_adam_lr1en2_accuracy = pickle.load(inverse_adam_lr1en2_file)
y_inverse_adam_lr1en1_accuracy = pickle.load(inverse_adam_lr1en1_file)

# y_inverse_adam_lr001_l25en7_accuracy = pickle.load(inverse_adam_lr001_l25en7_file)
# y_inverse_adam_lr001_l25en6_accuracy = pickle.load(inverse_adam_lr001_l25en6_file)


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return  total / (epoch_num + 0.0)

switch_rate_list = [1e-4, 1e-3, 1e-2, 1e-1]

scale = 10

accuracy_list = [average_accuracy(y_inverse_adam_lr1en4_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr1en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr1en2_accuracy, 10),
                 average_accuracy(y_inverse_adam_lr1en1_accuracy, 10),
                 ]

switch_rate_list = [i * scale for i in switch_rate_list]

vanilla_inverse_adam_accuracy = [average_accuracy(y_optimal_inverse_adam_accuracy, 10),
                                 average_accuracy(y_optimal_inverse_adam_accuracy, 10),]
vanilla_inverse_adam_x = [1e-4, 1e-1]
vanilla_inverse_adam_x = [i * scale for i in vanilla_inverse_adam_x]

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('learning rate(1e-1)')    # x轴标签
plt.ylabel('average accuracy(%)')     # y轴标签

plt.plot(switch_rate_list, accuracy_list, linewidth=1, linestyle="solid", color='red', marker="o", markersize=2, markeredgecolor="b")
plt.plot(vanilla_inverse_adam_x, vanilla_inverse_adam_accuracy, linewidth=1, linestyle="solid", label="optimal inverse adam", color='blue')


for x, y in zip(switch_rate_list, accuracy_list):
    plt.text(x, y+0.002, str(x).rstrip('0').rstrip('.'), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()