#使用sklearn实现逻辑回归进行多元分类
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import util.plot_util as pu
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

iris=datasets.load_iris()

X,y=iris.data[:,[2,3]],iris.target

lr=LogisticRegression(
    C=100,
    solver='lbfgs',
    random_state=1
)

ovr=OneVsRestClassifier(lr)

#对X进行标准化处理
sc=StandardScaler()
X=sc.fit_transform(X)

#拆分训练数据和测试数据
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=1)

#训练逻辑回归模型
ovr.fit(X_train,y_train)
print(ovr.predict_proba(X[:3]))

#绘制测试数据集的预测结果
test_idx=range(len(X_test),len(X))
pu.plot_decision_regions(ovr,X,y,resolution=0.02)


