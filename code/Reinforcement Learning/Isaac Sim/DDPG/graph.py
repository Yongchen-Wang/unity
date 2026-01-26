
import gym
import numpy as np

import matplotlib.pyplot as plt
import json

# from DDPG_env import MazeEnv



def plot_returns(file_path):
    with open(file_path, 'r') as f:
        return_list = json.load(f)

    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on MazeEnv')
    plt.savefig('returns_plot_4.png')  # 保存图形为文件
    plt.close()  # 关闭图形，释放内存
return_file = "return_list.json"
# 在需要绘制学习曲线时，调用plot_returns函数
plot_returns(return_file)
