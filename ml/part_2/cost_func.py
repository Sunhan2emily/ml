from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

cn_fonts = [f.name for f in font_manager.fontManager.ttflist if 'Sim' in f.name]

# 使用其中一个（比如 SimHei）
plt.rcParams['font.family'] = cn_fonts
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    return 1/(1+np.exp(-x))

#y=1时的代价
def cost_1(z):
    return -np.log(sigmoid(z))

#y=0是的代价
def cost_0(z):
    return -np.log(1-sigmoid(z))

#样本数据
X=np.arange(-10,10,0.1)
#计算所有可能的概率
pi_z=sigmoid(X)

plt.plot(pi_z,cost_1(X),label='y=1时的代价')
plt.plot(pi_z,cost_0(X),linestyle='--',label='y=0时的代价')
plt.xlabel('sigmoid(z)')
plt.ylabel('cost')

plt.legend()
plt.show()