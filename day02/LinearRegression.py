'''
    线性归回
'''

'''
    预处理可以调用上一层内容
'''
from sklearn.preprocessing import Imputer, LabelBinarizer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 数据预处理 导入库 --> 读取数据 --> 处理缺失值// --> 非数值数据编码// --> 制作标签// --> 归一化处理// --> 分割数据集
df = pd.read_csv('studentscores.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
print(X)

# x_imouter = Imputer()
# X[:, 1:] = x_imouter.fit_transform(X[:, 1:])
#
# x_str_to_num = LabelEncoder()
# X[:, 0] = x_str_to_num.fit_transform(X[:, 0])
#
# x_onehot = OneHotEncoder(categorical_features=[0], sparse=False)  # 不可以使用以上方法 --> 替换为:categorical_features=[0]
# X = x_onehot.fit_transform(X)
#
# y_onehot = LabelBinarizer()
# Y = y_onehot.fit_transform(Y)
#
# x_standard = StandardScaler()
# X = x_standard.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# 2. 训练模型
LR = LinearRegression(fit_intercept=True)
# fit_intercept --> 是否计算截距
# normalize --> 标准化(一般我们使用预处理而不用这个)
# n_jobs --> 1 or -1 1代表正常 2代表使用所有的cpu运算,只提供加速计算的功能
model = LR.fit(x_train, y_train)

# 3. 进行预测
y_prep = model.predict(x_test)

# 4. 可视化
plt.scatter(x_train,y_train,color='green',linewidths=.6,label='train')
plt.scatter(x_test,y_test,color='red',label='test')
plt.plot(x_test,y_prep,color='blue')
plt.show()