'''
    线性归回
'''

'''
    预处理可以调用上一层内容
'''
from sklearn.preprocessing import Imputer, LabelBinarizer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 数据预处理 导入库 --> 读取数据 --> 处理缺失值 --> 非数值数据编码 --> 制作标签 --> 分割数据集 --> 归一化处理
df = pd.read_csv('/home/tarena/1902_edward/MachineLearning/day01/data.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

x_imouter = Imputer()
X[:, 1:] = x_imouter.fit_transform(X[:, 1:])

x_str_to_num = LabelEncoder()
X[:,0] = x_str_to_num.fit_transform(X[:,0])

x_onehot = OneHotEncoder(categorical_features=[0],sparse=False) # 不可以使用以上方法 --> 替换为:categorical_features=[0]
X = x_onehot.fit_transform(X)

y_onehot = LabelBinarizer()
Y = y_onehot.fit_transform(Y)
print(X)
print(Y)

# 2. 训练模型

LR = LinearRegression(fit_intercept=True)
# fit_intercept --> 是否计算截距



# 3. 进行预测
# 4. 可视化
