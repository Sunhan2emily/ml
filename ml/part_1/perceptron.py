import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:

    #初始化函数
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        """学习效率"""
        self.eta = eta
        """训练次数"""
        self.n_iter = n_iter
        """随机数种子"""
        self.random_state=random_state
        """权重数组"""
        self.w_ = None
        """记录分类错误的次数"""
        self.errors_=[]

    #计算净输入
    def net_input(self,X):
        """计算权重与特征值的点积"""
        return np.dot(X,self.w_[1:])+self.w_[0]

    #判断函数
    def predict(self,X):
        """如果经输入大于等于0，返回1，否则返回-1"""
        return np.where(self.net_input(X)>=0,1,-1)

    #训练函数 X:特征值 y:分类数组
    def fit(self,X,y):
        """随机数生成器"""
        rgen=np.random.RandomState(self.random_state)
        """在均值为0，标准差为0.01的正态分布上生成随机数，作为初始权重"""
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]+1)
        """按照指定训练次数循环"""
        for _ in np.arange(0,self.n_iter):
            """分配错误的次数"""
            errors=0
            """循环训练样本数据集"""
            for idx,xi in X.iterrows():
                target=y[idx]
                """传入每个样本数据，预测分类标签"""
                predict_value=self.predict(xi)
                """实际值减去预测值，乘以学习效率，获取偏移值"""
                update=self.eta*(target-predict_value)
                """更新权重"""
                self.w_[0]+=update
                self.w_[1:]+=update*xi
                """如果预测失败，记录失败次数"""
                errors+=int(update!=0)
            """存储失败次数"""
            self.errors_.append(errors)

        return self

    #绘制决策边界
    def plot_decision_regions(self,X,y,resolution=0.01):
        """定义标记"""
        markets=('x','o','s','v')
        """定义不同颜色"""
        colors=('red','blue','lightgreen','gray')
        """根据分类标签的数量，创建颜色map"""
        cmap=ListedColormap(colors[:len(np.unique(y))])
        """获取所有特征的最大最小值，扩大一个单位，防止绘制边界紧邻数据点"""
        x1_min,x1_max=X.iloc[:,0].min()-1,X.iloc[:,0].max()+1
        x2_min,x2_max=X.iloc[:,1].min()-1,X.iloc[:,1].max()+1

        """创建网格坐标向量"""
        xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
        """展开所有坐标点，然后进行训练"""
        Z = self.predict(np.c_[xx1.ravel(),xx2.ravel()])
        Z=Z.reshape(xx1.shape)
        """绘制等高线填充图"""
        plt.xlim(x1_min,x1_max)
        plt.ylim(x2_min,x2_max)
        plt.contourf(xx1,xx2,Z, cmap=cmap,alpha=0.3)

        """在填充图上绘制样本数据"""
        for idx,cl in enumerate(np.unique(y)):
            plt.scatter(
                x=X[y==cl].iloc[:,0],
                y=X[y==cl].iloc[:,1],
                c=colors[idx],
                alpha=0.8,
                marker=markets[idx],
                label=np.where(cl==-1,'山鸢尾花','变色鸢尾花')
            )

        plt.legend()
        plt.show()