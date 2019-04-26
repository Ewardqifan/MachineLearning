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

# oneHotEncode()类的使用 --> (n_values,  categorical_features,  dtype,  sparse,  handle_unknown)
# 首先搞懂fit transform 以及 fit_transform 的用法比较:
# fit是生成transform的信息,用于分析数据
# transform是将fit后产生的数据形式适用于数据
# 而fit_transform是两者的结合

# n_values: 代表每个维度数据,需要用几维的one_hot编码实现,默认为'auto'自动分析
# e.g.:enc = OneHotEncoder(n_values = [2, 3, 4])
#      enc.fit([[0, 0, 3],
#               [1, 1, 0]])
#      ans = enc.transform([[0, 2, 3]]).toarray()
#      输出 [[ 1.  0.  0.  0.  1.  0.  0.  0.  1.]]
#     ps:  如果训练样本中有丢失的分类特征值，我们就必须显示地设置参数 n_values 了，这样防止编码出错

# categorical_features:对哪一维的数据进行编码 把需要编码的维度索引放在列表中

# dtype:输出类型格式,默认为'numpy.float64'

# sparse: 如果设置 False 与 transform后的.toarray()的参数效果相同

# handle_unknown:

lb = LabelBinarizer()
Y = lb.fit_transform(Y)

