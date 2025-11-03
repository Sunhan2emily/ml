#代码实现逻辑回归梯度下降函数
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import util.plot_util as pu


class LogisticRegressionGD:
    def __init__(self,eta=0.01,max_iter=1000,random_state=1):
        self.eta=eta
        self.max_iter=max_iter
        self.random_state=random_state
        """代价"""
        self.cost_=[]
        """权重"""
        self.w_=None

    #sigmoid函数
    def sigmoid(self,z):
        return 1/(1+np.exp(-np.clip(z,-250,250)))

    #净输入
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    #激活函数
    def activation(self,X):
        return self.sigmoid(self.net_input(X))

    #阈值函数
    def predict(self,X):
        return np.where(self.activation(X) >= 0.5, 1, 0)

    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]+1)

        for _ in range(self.max_iter):
            out_put=self.activation(X)
            errors=y-out_put
            #更新权重
            self.w_[1:]+=self.eta*np.dot(X.T,errors)
            self.w_[0]+=self.eta*(errors.sum())

            cost=-y.dot(np.log(out_put))-(1-y).dot(np.log(1-out_put))
            self.cost_.append(cost)

        return self

#---------------------------------------------
iris=datasets.load_iris()
X=iris.data[:100,[2,3]]
y=iris.target[:100]

sc=StandardScaler()
X_std=sc.fit_transform(X)

lrgd=LogisticRegressionGD()
#训练模型
lrgd.fit(X_std,y)
#绘图
pu.plot_decision_regions(classer=lrgd,X=X_std,y=y)
