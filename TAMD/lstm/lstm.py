
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 文件读取
def get_Data(data, col):
    data = data.iloc[:, col:col + 1]  # 以三个特征作为数据
    label = data  # 取最后一个特征作为标签
    return data, label
#%%


# 时间向量转换
def split_windows(data, seq_length, col):
    x = []
    y = []
    for i in range(len(data) - seq_length):  # range的范围需要减去时间步长和1
        _x = data[i:(i + seq_length), :]
        # _y=_x
        _y = data[i + seq_length, :]
        x.append(_x)
        y.append(_y)
    x, y = np.array(x), np.array(y)
    print('x.shape,y.shape=\n', x.shape, y.shape)
    return x, y
#%%


# 数据分离
def split_time(x, y, length):
    train_size = int(len(y) - length)
    test_size = len(y) - train_size

    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))

    x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    return x_data, y_data, x_train, y_train, x_test, y_test
#%%


# 数据装入
def data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size):
    num_epochs = n_iters / (len(x_train) / batch_size)  # n_iters代表一次迭代
    num_epochs = int(num_epochs)
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              drop_last=True)

    return train_loader, test_loader, num_epochs
#%%


import torch.nn as nn


# 定义一个类
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # LSTM层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        batch_size, seq_len = x.size()[0], x.size()[1]  # x.shape=(604,3,3)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))  # output(5, 30, 64)
        pred = self.fc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred