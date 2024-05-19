# import xlsxwriter
#
# workbook = xlsxwriter.Workbook('result.xlsx') # 建立文件
#
# worksheet = workbook.add_worksheet() # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
#
# num_done = 500000
# worksheet.write('A1', '步数')
# worksheet.write('A2', 'model')
# worksheet.write('A3', 'reward')
# worksheet.write('A4', 'win_agent')
#
# worksheet.write('A1', num_done) # 向A1写入
#
# worksheet.write(1,1,'guoshun')#向第二行第二例写入guoshun
# workbook.close()

# for i in range(10):
#     if i%5 == 0:
#         print(i)

import random

# a = [ 1.2490969  5.        -1.0564386]
# b = (3,4,5)
# c = random.randint(1,2)
# print(c)
# s1 = 0.1225
# if s1 < 0.125 and s1 >= 0:
#     s1 = 8
#     print(s1)

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
# from d2l import torch as d2l
from torch.utils import data
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.111111
b = np.loadtxt('data.txt', delimiter=',')
X_all = b[:,0:4]
y_all = b[:,4]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=42)

# 转化成torch.tensor类型
n_train = X_train.shape[0]
train_features = torch.tensor(X_train,
                              dtype = torch.float32)
test_features = torch.tensor(X_test,
                             dtype = torch.float32)
train_labels = torch.tensor(y_train.reshape(-1,1),
                            dtype = torch.float32)
test_labels = torch.tensor(y_test.reshape(-1,1),
                            dtype = torch.float32)


# 定义训练用的损失函数
loss = nn.MSELoss()
# 输入特征数
in_features = train_features.shape[1]

# 线性回归模型
# def get_net():
#    net = nn.Sequential(nn.Linear(in_features, 1))
#    return net

# mlp
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
#     net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# y的值比较大，所以都先取一个log，缩小范围，再用均方根误差
def log_rmse(net, features, labels):
    # torch.clamp(input, min, max, out=None) → Tensor
    # 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# 训练函数
def train(net, train_features, train_labels, test_features, test_labels,
         num_epochs, learning_rate, weight_decay, batch_size):
    # 数据迭代器，用于每次得到随机的一组batch
    train_iter = data.DataLoader(dataset = data.TensorDataset(train_features, train_labels),
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers = 0,
                                drop_last = True)
    # 设置优化器， 这里用了Adam
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate,
                                weight_decay = weight_decay)
    # 保存每一轮迭代之后的损失
    train_ls, test_ls = [], []
    # num_epochs轮训练
    for epoch in range(num_epochs):
        # 变成train模式
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # 变成eval模式
        net.eval()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# k折交叉验证，训练数据在第i折，X: 特征， y: 标签
def get_k_fold_data(k, i, X, y):
    # 要保证k>1
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # slice用于获取一个切片对象 https://m.runoob.com/python/python-func-slice.html
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx,:], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# k折交叉验证
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    # k折交叉验证的平均训练集损失和验证集损失
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # *data用于把data解包成X_train, y_train, X_test, y_test
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.xlim([1, num_epochs])
            plt.plot(list(range(1,num_epochs + 1)), train_ls, label = 'train')
            plt.yscale('log')
            plt.plot(list(range(1,num_epochs + 1)), valid_ls, label = 'valid')
            plt.legend()
            plt.show()
        print(f'fold {i+1}, train log rmse {float(train_ls[-1]):f}, valid log rmse {float(valid_ls[-1]):f}, ')
    # 取平均损失
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 30, 0.05, 0.3, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证：平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'train log rmse {float(train_ls[-1]):f}')
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.xlim([1, num_epochs])
    plt.plot(list(range(1,num_epochs + 1)), train_ls)
    plt.yscale('log')
    plt.show()
    # 转换成eval模式
    net.eval()
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis = 1)
    submission.to_csv('submission.csv', index = False)



train_and_pred(train_features, test_features, train_labels, test_labels, num_epochs, lr, weight_decay, batch_size)
