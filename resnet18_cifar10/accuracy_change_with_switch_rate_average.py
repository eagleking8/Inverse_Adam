import pickle
import matplotlib.pyplot as plt

optimal_inverse_adam_file = open(
    './inverse_adam_sr0.0001/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')

inverse_adam_sr5en6_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.000005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr1en5_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr5en5_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.00005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr7en5_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=7e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr8en5_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=8e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr9en5_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=9e-5_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr1en4_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr2en4_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=2e-4_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr5en4_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.0005_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr6en4_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=6e-4_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr1en3_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.001_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr2en3_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=2e-3_resnet18_cifar10.pkl', 'rb')
inverse_adam_sr5en3_file = open(
    './inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.005_resnet18_cifar10.pkl', 'rb')

# inverse_adam_sr1en2_file = open('./inverse_adam_lr0.001/InverseAdam_accuracy_200_epochs_lr=0.001_switchrate=0.01_resnet18_cifar10.pkl', 'rb')
# inverse_adam_lr001_l25en7_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-7_resnet18_cifar10.pkl', 'rb')
# inverse_adam_lr001_l25en6_file = open('./inverse_adam_l2_norm/InverseAdam_accuracy_200_epochs_lr=0.01_switchrate=0.0001_l2_norm=5e-6_warmup_resnet18_cifar10.pkl', 'rb')

y_optimal_inverse_adam_accuracy = pickle.load(optimal_inverse_adam_file)

y_inverse_adam_sr5en6_accuracy = pickle.load(inverse_adam_sr5en6_file)
y_inverse_adam_sr1en5_accuracy = pickle.load(inverse_adam_sr1en5_file)
y_inverse_adam_sr5en5_accuracy = pickle.load(inverse_adam_sr5en5_file)
y_inverse_adam_sr7en5_accuracy = pickle.load(inverse_adam_sr7en5_file)
y_inverse_adam_sr8en5_accuracy = pickle.load(inverse_adam_sr8en5_file)
y_inverse_adam_sr9en5_accuracy = pickle.load(inverse_adam_sr9en5_file)
y_inverse_adam_sr1en4_accuracy = pickle.load(inverse_adam_sr1en4_file)
y_inverse_adam_sr2en4_accuracy = pickle.load(inverse_adam_sr2en4_file)
y_inverse_adam_sr5en4_accuracy = pickle.load(inverse_adam_sr5en4_file)
y_inverse_adam_sr6en4_accuracy = pickle.load(inverse_adam_sr6en4_file)
y_inverse_adam_sr1en3_accuracy = pickle.load(inverse_adam_sr1en3_file)
y_inverse_adam_sr2en3_accuracy = pickle.load(inverse_adam_sr2en3_file)
y_inverse_adam_sr5en3_accuracy = pickle.load(inverse_adam_sr5en3_file)


# y_inverse_adam_sr1en2_accuracy = pickle.load(inverse_adam_sr1en2_file)
# y_inverse_adam_lr001_l25en7_accuracy = pickle.load(inverse_adam_lr001_l25en7_file)
# y_inverse_adam_lr001_l25en6_accuracy = pickle.load(inverse_adam_lr001_l25en6_file)


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return total / (epoch_num + 0.0)


switch_rate_list = [5e-6, 1e-5, 5e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 5e-4, 6e-4, 1e-3, 2e-3, 5e-3]

accuracy_list = [average_accuracy(y_inverse_adam_sr5en6_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr1en5_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr5en5_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr7en5_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr8en5_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr9en5_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr1en4_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr2en4_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr5en4_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr6en4_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr1en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr2en3_accuracy, 10),
                 average_accuracy(y_inverse_adam_sr5en3_accuracy, 10),
                 ]

switch_rate_list = [i * 1e3 for i in switch_rate_list]

vanilla_inverse_adam_accuracy = [average_accuracy(y_optimal_inverse_adam_accuracy, 10),
                                 average_accuracy(y_optimal_inverse_adam_accuracy, 10),
                                 average_accuracy(y_optimal_inverse_adam_accuracy, 10)]
vanilla_inverse_adam_x = [0.000005, 0.0001, 0.005]
vanilla_inverse_adam_x = [i * 1e3 for i in vanilla_inverse_adam_x]

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('switch rate(1e-3)')  # x轴标签
plt.ylabel('average accuracy(%)')  # y轴标签

# plt.vlines(l2_coefficient_list, 1e-6, accuracy_list, linestyle="dashed")
# plt.hlines(accuracy_list, 0, l2_coefficient_list, linestyle="dashed")

plt.plot(switch_rate_list, accuracy_list, linewidth=1, linestyle="solid", color='red', marker='o',
         markeredgecolor='b', markersize='2')
plt.plot(vanilla_inverse_adam_x, vanilla_inverse_adam_accuracy, linewidth=1, linestyle="solid",
         label="optimal inverse adam",
         color='blue')

for x, y in zip(switch_rate_list, accuracy_list):
    plt.text(x, y + 0.002, str(x).rstrip('0').rstrip('.'), horizontalalignment='center', verticalalignment='bottom',
             fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()
