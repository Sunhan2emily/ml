import numpy as np
import matplotlib.pyplot as plt
#定义sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#定义输入
z=np.arange(-7,7,0.1)
pi=sigmoid(z)

#绘制曲线
plt.plot(z,pi)
#在x=0处绘制一条垂直的参考线
plt.axvline(0,color='r')
plt.ylim(-0.1,1.1)
#获取当前坐标
ac=plt.gca()
#在y轴方向绘制网格线
ac.yaxis.grid(True)
#设置y轴刻度
plt.yticks([0,0.5,1])
plt.xlabel('z')
plt.ylabel('pi')
plt.show()

