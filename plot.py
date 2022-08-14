import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

log_file1 = "run_logs/noadvice.log"
log_file2 = "run_logs/rcmp_avg_ploss_noeffect_action.log"

log_file3 = "run_logs/rand_advice.log"

def plt_log(logfile):
    data = open(logfile, 'r')
    reward = []
    epoch = []
    for line in data:
        items = line.split(" ")
        if len(items) < 8 or items[7] != 'epoch':
            continue
        epoch.append(int(float(items[8].split(',')[0])))
        reward.append(int(float(items[11].split(',')[0])))
        if float(items[8].split(',')[0]) > 3000000:
            break
    plt.plot(epoch, gaussian_filter1d(reward, sigma=2))
    print(f"epoch : {epoch[-1]}, reward : {reward[-1]}")

plt_log(log_file1)
plt_log(log_file2)
plt_log(log_file3)
plt.margins(x=0)
plt.legend(["noadvice", "rcmp", "random"])
plt.savefig("result")
