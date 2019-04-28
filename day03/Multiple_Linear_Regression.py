'''
    多元线性回归模型
        1.数据预处理 : 导入相关库 --> 处理缺省值 --> 非数值数据编码 --> 标签制作 --> 标准化 -->数据集切分
        2.数据训练 : 导入相关模型 --> fit_transform
        3.预测 : predict
        4.可视化 : plt
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import \
    LabelBinarizer, OneHotEncoder, Imputer, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

x_imputer = LabelEncoder()
X[:, 3] = x_imputer.fit_transform(X[:, 3])

x_onehot = OneHotEncoder(categorical_features=[3])
X = x_onehot.fit_transform(X).toarray()

# 由于X的第维解析后的特征表达为[0,1,2],形成的onehot编码的格式位三维[0/1,0/1,0/1]
# 但是这样会形成虚拟陷阱影响精度.虚拟变量是可以通过其他变量替换.
# 这里形成的三维数据任何一维都可以通过其他两维得到
X = X[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

LR = LinearRegression()
model = LR.fit(x_train, y_train)
print(model.coef_)  # 系数列表
print(model.intercept_)  # 截距

y_pred = model.predict(x_test)

plt.figure()
N = np.arange(len(Y))
p_train = plt.plot(N[0:len(y_train)], y_train, color='b')
print(type(p_train))  # 由此可知plt.plot返回的是一个列表其中只有一个对象
p_test, = plt.plot(N[len(y_train):], y_test, color='g')
p_pred, = plt.plot(N[len(y_train):], y_pred, color='r', linestyle='--')
plt.legend(handles=[p_train[0], p_test, p_pred], labels=['train', 'test', 'pred'])
plt.savefig('MLR')
plt.show()
