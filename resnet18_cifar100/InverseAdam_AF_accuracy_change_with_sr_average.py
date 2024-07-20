import pickle
import matplotlib.pyplot as plt
sr1en6_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr=1e-6_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr5en6_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr=5e-6_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr1en5_file = open('./InverseAdam_AF/InverseAdam_AF_loss_200_epochs_lr=1e-2_sr1e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr2en5_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr2e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr5en5_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr5e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr8en5_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr8e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')

y_sr1en6 = pickle.load(sr1en6_file)
y_sr5en6 = pickle.load(sr5en6_file)
y_sr1en5 = pickle.load(sr1en5_file)
y_sr2en5 = pickle.load(sr2en5_file)
y_sr5en5 = pickle.load(sr5en5_file)
y_sr8en5 = pickle.load(sr8en5_file)


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return  total / (epoch_num + 0.0)

switch_rate_list = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 8e-5]

scale = 10

accuracy_list = [average_accuracy(y_sr1en6, 10),
                 average_accuracy(y_sr5en6, 10),
                 average_accuracy(y_sr1en5, 10),
                 average_accuracy(y_sr2en5, 10),
                 average_accuracy(y_sr5en5, 10),
                 average_accuracy(y_sr8en5, 10),
                 ]

switch_rate_list = [i * scale for i in switch_rate_list]


plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('learning rate(1e-1)')    # x轴标签
plt.ylabel('average accuracy(%)')     # y轴标签

plt.plot(switch_rate_list, accuracy_list, linewidth=1, linestyle="solid", color='red', marker="o", markersize=2, markeredgecolor="b")


for x, y in zip(switch_rate_list, accuracy_list):
    plt.text(x, y+0.002, str(x).rstrip('0').rstrip('.'), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

plt.legend()
plt.title('accuracy curve')
plt.show()