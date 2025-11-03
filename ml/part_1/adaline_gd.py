import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


#自适应线性神经元
class AdalineGD:
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        """权重"""
        self.w_=None
        """误差"""
        self.cost_=[]

    #净输入
    def net_input(self,X):
       return np.dot(X,self.w_[1:])+self.w_[0]

    #激活函数
    def activation(self,X):
        return X

    #阈值函数
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0,1,-1)

    #训练
    def fit(self,X,y):
        """生成初始权重"""
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]+1)
        print(f'初始权重：{self.w_}")')
        print(f"eta:{self.eta}")
        """循环训练"""
        for _ in range(self.n_iter):
            net_input=self.net_input(X)
            v=self.activation(net_input)
            """误差"""
            errors=y-v
            print(f'误差范围 {errors.min()}-{errors.max()}')

            """计算梯度"""
            gradient_w=X.T.dot(errors)
            gradient_b=errors.sum()

            """更新权重"""
            self.w_[1:]+=self.eta*gradient_w
            self.w_[0]+=self.eta*gradient_b

            print(f'更新后权重：{self.w_}')

            """本次训练的误差"""
            cost=(errors**2).sum()/2.0
            print(f'本次训练的误差：{cost}')
            self.cost_.append(cost)
            print("-"*30)
        return self


#------
# 读取数据
data = pd.read_csv('iris.csv')
d1 = data.iloc[:100, [0, 2, 4]]

X = d1.iloc[:, [0, 1]]
y = d1.iloc[:, 2]
y = np.where(y == 'Iris-setosa', 0, 1)

print("原始数据统计:")
print(f"特征1范围: {X.iloc[:, 0].min():.2f} - {X.iloc[:, 0].max():.2f}")
print(f"特征2范围: {X.iloc[:, 1].min():.2f} - {X.iloc[:, 1].max():.2f}")

# 关键步骤：特征标准化
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
"""特征标准化：特征值-均值/标准差"""
X_std=np.copy(X)
X_std[:,0]=(X.iloc[:,0]-X.iloc[:,0].mean())/X.iloc[:,0].std()
X_std[:,1]=(X.iloc[:,1]-X.iloc[:,1].mean())/X.iloc[:,1].std()

print("\n标准化后数据统计:")
print(f"特征1范围: {X_std[:, 0].min():.2f} - {X_std[:, 0].max():.2f}")
print(f"特征2范围: {X_std[:, 1].min():.2f} - {X_std[:, 1].max():.2f}")

# 训练模型（使用标准化数据）
cost1 = AdalineGD(eta=0.02, n_iter=5).fit(X_std, y)
cost2 = AdalineGD(eta=0.001, n_iter=5).fit(X_std, y)

print(f"\nη=0.02 的最终代价: {cost1.cost_[-1]:.6f}")
print(f"η=0.001 的最终代价: {cost2.cost_[-1]:.6f}")

# 绘制结果
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# 左图：η=0.01
axes[0].plot(range(1, len(cost1.cost_) + 1), cost1.cost_, marker='o', color='blue')
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('代价')
axes[0].set_title('Adaline - 学习速率 0.01 (标准化后)')
axes[0].grid(True, alpha=0.3)

# 右图：η=0.001
axes[1].plot(range(1, len(cost2.cost_) + 1), cost2.cost_, marker='o', color='red')
axes[1].set_xlabel('迭代次数')
axes[1].set_ylabel('代价')
axes[1].set_title('Adaline - 学习速率 0.001 (标准化后)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

