import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import time


TEST_SIZE = 0.111111
b = np.loadtxt('data.txt', delimiter=',')
X = b[:, 0:4]
y = b[:, 4]
y = y.astype(int)
# print(len(set(y.tolist())))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# print(type(X_train))
# exit()
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

data_train = TensorDataset(X_train, y_train)
data_loader_train = DataLoader(dataset=data_train,
                            batch_size=64,
                            shuffle=True)
data_test = TensorDataset(X_test, y_test)
data_loader_test = DataLoader(dataset=data_test,
                            batch_size=64,
                            shuffle=True)


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(4, 12)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(12, 11)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(11, 10)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(10, 9)
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(9, 25)
        self.relu5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(25, 8)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        return x


def train(model):
    # 损失函数，它将网络的输出和目标标签进行比较，并计算它们之间的差异。在训练期间，我们尝试最小化损失函数，以使输出与标签更接近
    cost = torch.nn.CrossEntropyLoss()
    # 优化器的一个实例，用于调整模型参数以最小化损失函数。
    # 使用反向传播算法计算梯度并更新模型的权重。在这里，我们使用Adam优化器来优化模型。model.parameters()提供了要优化的参数。
    optimizer = torch.optim.Adam(model.parameters())
    # 设置迭代次数
    epochs = 100
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for data in data_loader_train:
            inputs, labels = data
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = cost(outputs, labels.long())
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        print('[%d/%d] loss:%.3f, correct:%.3f%%, time:%s' %
              (epoch + 1, epochs, sum_loss / len(data_loader_train),
               100 * train_correct / len(data_train),
               time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    model.train()


# 测试模型
def test(model, test_loader):
    model.eval()
    test_correct = 0
    for data in test_loader:
        inputs, lables = data
        inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
        # inputs = torch.flatten(inputs, start_dim=1)  # 展并数据
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(f'Accuracy on test set: {100 * test_correct / len(data_test):.3f}%')


model = MLP()
train(model)
test(model, data_loader_test)
