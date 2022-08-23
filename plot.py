import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

log_file1 = "run_logs/new_no_advice.log"
log_file2 = "run_logs/new_RCMP.log"

log_file3 = "run_logs/new_random.log"
log_file4 = "run_logs/new_importance.log"

invaders = "run_logs/SpaceInvaders_noadvice.log"
invaders_rcmp = "run_logs/SpaceInvaders_rcmp.log"
invaders_rcmp_aceff = "run_logs/SpaceInvaders_rcmp_aceff.log"
invaders_importance = "run_logs/SpaceInvaders_importance.log"
invaders_random = "run_logs/SpaceInvaders_random.log"
invaders_05importance =  "run_logs/SpaceInvaders_importance0.5.log"
invaders_rcmp005 = "run_logs/SpaceInvaders_rcmp0.05.log"
invaders_rcmp_imsample = "run_logs/SpaceInvaders_rcmp_imsample.log"
invaders_rcmp02 = "run_logs/SpaceInvaders_rcmp0.2.log"

Qbert = "run_logs/Qbert_noadvice.log"
Qbert_rcmp = "run_logs/Qbert_rcmp.log"
Qbert_importance = "run_logs/Qbert_importance.log"
Qbert_random = "run_logs/Qbert_random_raw.log"
Qbert_rcmp02 = "run_logs/Qbert_rcmp0.2.log"
Qbert_importance05 = "run_logs/Qbert_importance0.5.log"

def plt_log(logfile):
    data = open(logfile, 'r')
    reward = []
    epoch = []
    for line in data:
        items = line.split(" ")
        if len(items) < 8 or items[7] != 'epoch':
            continue
        epoch.append(int(float(items[8].split(',')[0])))
        reward.append(float(items[11].split(',')[0]))
        # if float(items[8].split(',')[0]) > 3000000:
        #     break
    plt.plot(epoch, gaussian_filter1d(reward, sigma=1))

    print(f"epoch : {epoch[-1]}, reward : {reward[-1]}")

def plt_used_advice(logfile):
    data = open(logfile, 'r')
    used_advice = []
    epoch = []
    for line in data:
        items = line.split(" ")
        if len(items) < 8 or items[7] != 'epoch':
            continue
        epoch.append(int(float(items[8].split(',')[0])))
        if int(float(items[-1].split(',')[0])) < 0:
            used_advice.append(10000)
        else:
            used_advice.append(10000 - int(float(items[-1].split(',')[0])))
        # if float(items[8].split(',')[0]) > 3000000:
        #     break
    plt.plot(epoch, used_advice)

# def plt_log_acc(logfile):
#     data = open(logfile, 'r')
#     acc_reward = 0
#     reward = []
#     epoch = []
#     for line in data:
#         items = line.split(" ")
#         if len(items) < 8 or items[7] != 'epoch':
#             continue
#         epoch.append(int(float(items[8].split(',')[0])))

#         reward.append(float(items[11].split(',')[0]))
#         if float(items[8].split(',')[0]) > 3000000:
#             break
#     plt.plot(epoch, gaussian_filter1d(reward, sigma=2))
#     print(f"epoch : {epoch[-1]}, reward : {reward[-1]}")

# plt_log(log_file1)
# plt_log(log_file2)
# plt_log(log_file3)

# plt_log(invaders)
# plt_log(invaders_rcmp)
# plt_used_advice(Qbert)
# # plt_used_advice(Qbert_importance)
# plt_used_advice(Qbert_random)
# plt_used_advice(Qbert_rcmp)
# plt_used_advice(Qbert_importance05)

# plt_log(Qbert)
# plt_log(Qbert_rcmp)
# # plt_log(Qbert_rcmp02)
# plt_log(Qbert_random)
# # plt_log(Qbert_importance)
# plt_log(Qbert_importance05)
# plt.xlim(0, 1e7)
# plt.margins(x=0, y=0)
# plt.legend(["noadvice", "rcmp", "random", "importance"])
# plt.title("Qbert-v0")
# plt.savefig("result")

# plt_used_advice(invaders)
# plt_used_advice(invaders_rcmp)
# plt_log(invaders_rcmp_aceff)
# plt_log(invaders_importance)
# plt_used_advice(invaders_random)
# plt_used_advice(invaders_05importance)
# plt_log(invaders_rcmp005)
# plt_log(invaders_rcmp_imsample)
plt_log(invaders)
plt_log(invaders_rcmp)
plt_log(invaders_rcmp005)
# plt_log(invaders_importance)
# plt_log(invaders_random)
# plt_log(invaders_05importance)
plt_log(invaders_rcmp02)
plt.margins(x=0, y=0)
plt.xlim(0, 1e7)
plt.legend(["noadvice", "rcmp", "rcmp0.05", "rcmp0.2"])
plt.title("SpaceInvaders")
plt.savefig("rcmp_compare")


