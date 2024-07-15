import pickle
import matplotlib.pyplot as plt

inverse_adam_wd0_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=0_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en10_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-10_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd5en10_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=5e-10_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd2en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=2e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd3en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=3e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd5en9_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=5e-9_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en2_file =  open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en2_warmup_file =  open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_warm_up_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en2_cosine_file =  open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_cosine_resnet18_cifar10.pkl', 'rb')
inverse_adam_wd1en2_cosine1000_file =  open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
inverse_adam_warmup_cosine100_file =  open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_warm_up_cosine100_resnet18_cifar10.pkl', 'rb')
inverse_adam_warmup_cosine1000_file = open('./inverse_adam_lr1p01e-2_sr8e-5/InverseAdam_accuracy_200_epochs_lr=1.01e-2_switchrate=8e-5_wd=1e-2_warm_up_cosine1000_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en2_cosine1000_file = open('./inverse_adam_lr1e-2_sr8e-5_wd1e-2/InverseAdam_accuracy_200_epochs_lr=1e-2_switchrate=8e-5_wd=1e-2_cosine1000_resnet18_cifar10.pkl', 'rb')
inverse_adam_lr1en2_warm_up_cosine100_file = open('./inverse_adam_lr1e-2_sr8e-5_wd1e-2/InverseAdam_accuracy_200_epochs_lr=1e-2_switchrate=8e-5_wd=1e-2_warm_up_cosine100_resnet18_cifar10.pkl', 'rb')

sgdm_file = open('./SGDM/SGDM_accuracy_200_epochs_lr=0.1_momentum=0.9_batchsize=128_weightdecay=5e-4_resnet18_cifar10.pkl', 'rb')


y_inverse_adam_wd0_accuracy = pickle.load(inverse_adam_wd0_file)[-100:]
y_inverse_adam_wd1en10_accuracy = pickle.load(inverse_adam_wd1en10_file)[-100:]
y_inverse_adam_wd5en10_accuracy = pickle.load(inverse_adam_wd5en10_file)[-100:]
y_inverse_adam_wd1en9_accuracy = pickle.load(inverse_adam_wd1en9_file)[-100:]
y_inverse_adam_wd2en9_accuracy = pickle.load(inverse_adam_wd2en9_file)[-100:]
y_inverse_adam_wd3en9_accuracy = pickle.load(inverse_adam_wd3en9_file)[-100:]
y_inverse_adam_wd5en9_accuracy = pickle.load(inverse_adam_wd5en9_file)[-100:]
y_inverse_adam_wd1en2_accuracy = pickle.load(inverse_adam_wd1en2_file)[-100:]
y_inverse_adam_wd1en2_warm_up_accuracy = pickle.load(inverse_adam_wd1en2_warmup_file)[-100:]
y_inverse_adam_wd1en2_cosine_accuracy = pickle.load(inverse_adam_wd1en2_cosine_file)[-100:]
y_inverse_adam_wd1en2_cosine1000_accuracy = pickle.load(inverse_adam_wd1en2_cosine1000_file)[-100:]
inverse_adam_warmup_cosine100_accuracy = pickle.load(inverse_adam_warmup_cosine100_file)[-100:]
inverse_adam_warmup_cosine1000_accuracy = pickle.load(inverse_adam_warmup_cosine1000_file)[-100:]
inverse_adam_lr1en2_cosine1000_accuracy = pickle.load(inverse_adam_lr1en2_cosine1000_file)[-100:]
inverse_adam_lr1en2_warm_up_cosine100_accuracy = pickle.load(inverse_adam_lr1en2_warm_up_cosine100_file)[-100:]


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

plt.plot(epoch, y_inverse_adam_wd0_accuracy, linewidth=1, linestyle="solid", label="inverse_adam wd=0", color='green')
plt.plot(epoch, y_inverse_adam_wd1en2_warm_up_accuracy, linewidth=1, linestyle="solid", label="inverse_adam wd=1e-2 warm up", color='blue')
plt.plot(epoch, y_inverse_adam_wd1en2_cosine_accuracy, linewidth=1, linestyle="solid", label="inverse_adam wd=1e-2 cosine minlr = lr/100", color='black')
plt.plot(epoch, y_inverse_adam_wd1en2_cosine1000_accuracy, linewidth=1, linestyle="solid", label="inverse_adam wd=1e-2 cosine minlr = lr/1000", color='purple')
plt.plot(epoch, inverse_adam_warmup_cosine100_accuracy, linewidth=2, linestyle="dotted", label="inverse_adam wd=1e-2 warm up cosine minlr = lr/100", color='grey')
plt.plot(epoch, inverse_adam_warmup_cosine1000_accuracy, linewidth=1, linestyle="solid", label="inverse_adam wd=1e-2 warm up cosine minlr = lr/1000", color='pink')
plt.plot(epoch, inverse_adam_lr1en2_cosine1000_accuracy, linewidth=1, linestyle="solid", label="inverse_adam lr=1e-2 wd=1e-2 cosine minlr = lr/1000", color='orange')
plt.plot(epoch, inverse_adam_lr1en2_warm_up_cosine100_accuracy, linewidth=1, linestyle="dotted", label="inverse_adam lr=1e-2 wd=1e-2 warm up cosine minlr = lr/100", color='red')

plt.plot(epoch, y_sgdm_accuracy, linewidth=1, linestyle="dashed", label="sgdm weight decay=5e-4", color='pink')

plt.legend()
plt.title('accuracy curve')
plt.show()