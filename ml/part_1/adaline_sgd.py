import numpy as np
#随机梯度下降法 每次训练一个样本，然后更新权重

class AdalineSGD:
    def __init__(self,eta=0.01,n_iter=50,shuffle=True,random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.w_=None
        self.cost_=[]
        self.shuffle=shuffle
        self.random_state=random_state


    #净输入
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    #激活函数
    def activation(self,X):
        return X

    #阈值函数
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0,1,-1)

    #初始化权重
    def initialize_weight(self,X):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]+1)

    #重排数据
    def shuffle_data(self,X,y):
        """随机生成0到len(y)的随机排列"""
        r=np.random.permutation(len(y))
        return X[r],y[r]

    #训练
    def fit(self,X,y):
        self.initialize_weight(X)
        for _ in range(self.n_iter):
            """如果重新排列数据"""
            if self.shuffle:
                X,y=self.shuffle_data(X,y)
            cost=[]
            for idx,xi in X.iterrows():
                target=y[idx]
                net_input=self.net_input(xi)
                output=self.activation(net_input)
                error=target-output

                self.w_[1:]+=self.eta*error.dot(xi)
                self.w_[0]+=self.eta*error

                cost.append(0.5*error**2)

            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)




