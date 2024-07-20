import pickle
import matplotlib.pyplot as plt
sr1en4_file = open('AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1e-4_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr1en3_file = open('AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1e-3_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr1en2_file = open('AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1e-2_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr1en1_file = open('AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1e-1_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
sr1en0_file = open('AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')

y_sr1en4 = pickle.load(sr1en4_file)
y_sr1en3 = pickle.load(sr1en3_file)
y_sr1en2 = pickle.load(sr1en2_file)
y_sr1en1 = pickle.load(sr1en1_file)
y_sr1en0 = pickle.load(sr1en0_file)



def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return  total / (epoch_num + 0.0)

switch_rate_list = [1e-4, 1e-3, 1e-2, 1e-1, 1]

scale = 10

accuracy_list = [average_accuracy(y_sr1en4, 10),
                 average_accuracy(y_sr1en3, 10),
                 average_accuracy(y_sr1en2, 10),
                 average_accuracy(y_sr1en1, 10),
                 average_accuracy(y_sr1en0, 10),
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