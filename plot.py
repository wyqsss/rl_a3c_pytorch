import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

log_file1 = "run_logs/new_no_advice.log"
log_file2 = "run_logs/new_RCMP.log"

log_file3 = "run_logs/new_random.log"
log_file4 = "run_logs/new_importance.log"

invaders = "run_logs/SpaceInvaders_noadvice.log"
invaders_rcmp = "run_logs/SpaceInvaders_rcmp.log"

Qbert = "run_logs/Qbert_noadvice.log"

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
plt_log(Qbert)
plt.margins(x=0)
plt.legend(["noadvice"])
plt.title("Qbert")
plt.savefig("result")

