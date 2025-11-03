from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

#加载数据
iris=datasets.load_iris()
#提取特征和标签
X=iris.data[:,[2,3]]
y=iris.target

#划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

#对特征进行标准化处理
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(eta0=0.01,max_iter=10,random_state=3)
#使用训练数据集训练感知机
ppn.fit(X_train_std,y_train)

#使用测试数据集进行预测
# y_pred=ppn.predict(X_test_std)
# print(f'分类错误的数量：{sum(y_test!=y_pred)}')

def plot_preceptron(classer,X,y,test_idx,resolution=0.02):
    markers=('x','o','*','v','^')
    colors=('red','green','blue','yellow','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1

    #获取网格坐标
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    #对网格上的每个点进行预测
    Z=classer.predict(np.c_[xx1.ravel(),xx2.ravel()])
    #把预测结果改成网格上点对应的形状
    Z=Z.reshape(xx1.shape)

    #绘制区域填充图
    plt.contourf(xx1,xx2,Z,cmap=cmap,alpha=0.3)

    #绘制数据点
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y==cl,0],
            y=X[y==cl,1],
            c=colors[idx],
            marker=markers[idx],
            alpha=1,
            label=f'{cl}',
        )

    if test_idx:
        plt.scatter(
            x=X[test_idx,0],
            y=X[test_idx,1],
            marker='o',
            linewidth=1,
            edgecolor='black',
            s=100,
            label='test',
            alpha=0.3,
        )

    plt.xlabel('sepal length [cm]')
    plt.ylabel('sepal width [cm]')
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.legend()
    plt.show()


#合并训练数据集和测试数据集
X_std=np.vstack((X_train_std,X_test_std))
y_std=np.hstack((y_train,y_test))

plot_preceptron(ppn,X_std,y_std,test_idx=range(len(X_train),len(X)),resolution=0.02)
