'''
    数据预处理:
        1.导入库  --> np and pd
        2.导入数据 --> pd.read_csv()
        3.处理丢失 --> Imputer
        4.解析分类 --> Onehotencoder
        5.拆分数据 -->
        6.特征缩放 -->
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder,LabelBinarizer

df = pd.read_csv('data.csv')
X = df.iloc[:, :-1].values  # iloc与values是属性
Y = df.iloc[:, -1].values
print(Y)

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
x_labelencoder = LabelEncoder()
X[:,0] = x_labelencoder.fit_transform(X[:,0])
print(X)
x_onehot = OneHotEncoder(categorical_features=[0])

lb = LabelBinarizer()
Y = lb.fit_transform(Y)

