import matplotlib.pyplot as plt
import numpy as np
import os


def plotLearning(score_list, filename, window=5):
    # score_list=[scores[k] for k in scores.keys()]
    N = len(score_list[0])

    running_avg = np.zeros((len(score_list), N))
    for i in range(len(score_list)):
        for t in range(N):
            running_avg[i][t] = np.mean(score_list[i][max(0, t - window) : (t + 1)])
    total_mean = np.mean(running_avg, axis=0)
    x = [i for i in range(N)]
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    for rewards in range(len(score_list)):
        plt.plot(x, running_avg[rewards])
    plt.plot(x, total_mean, label="Mean")
    plt.legend()
    path = r"C:\Users\victo\Desktop\VU master\drones\Drones_RL\gym-pybullet-drones\gym_pybullet_drones\VU_Swarm\models\saved_models\test_with_heading"
    plt.savefig(os.path.join(path, filename))
