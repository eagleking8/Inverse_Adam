import pickle
import matplotlib.pyplot as plt

sgdm_file = open('./SGDM/SGDM_accuracy_200_epochs_lr=1e-1_wd=5e-4_cosine1000_resnet18_cifar100.pkl', 'rb')
adam_file = open('./Adam/Adam_accuracy_200_epochs_lr=1e-3_wd=0_cosine1000_resnet18_cifar100.pkl', 'rb')
inverse_adam_if_file = open('./InverseAdam_IF/InverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=8e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
inverse_adam_af_file = open('./InverseAdam_AF/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_sr1e-5_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
RAdam_file = open('./RAdam/RAdam_accuracy_200_epochs_lr=1e-3_wd=0_cosine1000_resnet18_cifar100.pkl', 'rb')
RAdamW_file = open('./RAdam/RAdam_accuracy_200_epochs_lr=1e-3_decoupled_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
adamw_lr1en3_file = open('./AdamW/AdamW_accuracy_200_epochs_lr=1e-3_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
ar_inverse_adam_file = open('./AR_InverseAdam/ARInverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=1e-2_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')
inverse_adam_if_wd0_file = open('./InverseAdam_IF/InverseAdam_IF_accuracy_200_epochs_lr=1e-2_sr=8e-5_wd=0_cosine1000_resnet18_cifar100.pkl', 'rb')
sgd_file = open('./SGD/SGD_accuracy_200_epochs_lr=1e-1_momentum=0_wd=5e-4_cosine1000_resnet18_cifar100.pkl', 'rb')
NAdam_file = open('./NAdam/NAdam_accuracy_200_epochs_lr=2e-3_decoupled_wd=0_cosine1000_resnet18_cifar100.pkl', 'rb')
NAdamW_file = open('./NAdam/NAdam_accuracy_200_epochs_lr=2e-3_decoupled_wd=1e-2_cosine1000_resnet18_cifar100.pkl', 'rb')

# inverse_adam_af_file = open('./inverse_adam_af_lr1e-2_wd1e-2/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
# adamw_file = open('./adamw/AdamW_accuracy_200_epochs_lr=1e-2_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
# inverse_adam_if_lr1en3_file= open('./inverse_adam_lr1e-3/InverseAdam_accuracy_200_epochs_lr=1e-3_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
# inverse_adam_af_sr1en4_file = open('./inverse_adam_af_lr1e-2_wd1e-2/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_switchrate=1e-4_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
# inverse_adam_af_sr1en3_file = open('./inverse_adam_af_lr1e-2_wd1e-2/InverseAdam_AF_accuracy_200_epochs_lr=1e-2_switchrate=1e-3_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')

y_sgdm = pickle.load(sgdm_file)
y_adam = pickle.load(adam_file)
y_inverse_adam_if = pickle.load(inverse_adam_if_file)
y_inverse_adam_af = pickle.load(inverse_adam_af_file)
y_adamw_lr1en3 = pickle.load(adamw_lr1en3_file)
y_ar_inverse_adam = pickle.load(ar_inverse_adam_file)
y_inverse_adam_if_wd0 = pickle.load(inverse_adam_if_wd0_file)
y_RAdam = pickle.load(RAdam_file)
y_RAdamW = pickle.load(RAdamW_file)
y_sgd = pickle.load(sgd_file)
y_NAdam = pickle.load(NAdam_file)
y_NAdamW = pickle.load(NAdamW_file)

# y_inverse_adam_af = pickle.load(inverse_adam_af_file)[-100:]
# y_adamw = pickle.load(adamw_file)[-100:]
# y_inverse_adam_if_lr1en3 = pickle.load(inverse_adam_if_lr1en3_file)[-100:]
# y_inverse_adam_af_sr1en4 = pickle.load(inverse_adam_af_sr1en4_file)[-100:]
# y_inverse_adam_af_sr1en3 = pickle.load(inverse_adam_af_sr1en3_file)[-100:]


epoch = range(0, 200)

plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')    # x轴标签
plt.ylabel('accuracy(%)')     # y轴标签

plt.plot(epoch, y_RAdam, linewidth=1, linestyle="solid", label="RAdam", color='green')
plt.plot(epoch, y_RAdamW, linewidth=1, linestyle="dotted", label="RAdamW", color='green')
plt.plot(epoch, y_adamw_lr1en3, linewidth=1, linestyle="solid", label="AdamW", color='blue')
# plt.plot(epoch, y_inverse_adam_af, linewidth=1, linestyle="solid", label="InverseAdam AF", color='black')
plt.plot(epoch, y_sgd, linewidth=1, linestyle="solid", label="SGD", color='purple')
plt.plot(epoch, y_inverse_adam_if_wd0, linewidth=1, linestyle="solid", label="InverseAdam wd=0", color='orange')
plt.plot(epoch, y_adam, linewidth=1, linestyle="solid", label="Adam", color='pink')
plt.plot(epoch, y_inverse_adam_if, linewidth=1, linestyle="dotted", label="InverseAdam", color='pink')
plt.plot(epoch, y_NAdam, linewidth=1, linestyle="dotted", label="NAdam", color='black')
plt.plot(epoch, y_NAdamW, linewidth=1, linestyle="solid", label="NAdamW", color='black')



plt.legend()
plt.title('accuracy curve')
plt.show()