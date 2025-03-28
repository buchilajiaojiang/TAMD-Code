import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from ADA.TDC import get_index, get_split_time, split_data


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def split_windows(data, seq_length, col=0):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length), col])
        y.append(data[i + seq_length, col])
    return np.array(x), np.array(y)


def train_model(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # 数据加载
    df = pd.read_csv(args.data_path, index_col=0)
    time_index = df.index

    # 数据预处理
    scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(len(df.columns))]
    scaled_data = [scaler.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
                   for i, scaler in enumerate(scalers)]

    # 模型初始化
    model = LSTMModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练过程
    predictions = {}
    true_values = {}

    for col in tqdm(range(len(df.columns)), desc=f"Training {args.model_name}"):
        # 数据准备
        data = scaled_data[col]
        train_data = data[:-300]
        valid_data = data[-300:-150]
        test_data = data[-150:]

        # 分割时间序列
        split_time_list = get_split_time(args.domain, mode='tdc', data=df.iloc[:, col],
                                         dis_type=args.distance_type, dis=args.distance)
        data_list = split_data(df.iloc[:, col], split_time_list)

        # 创建数据加载器
        train_x, train_y = [], []
        for i in range(len(data_list[:-300])):
            x, y = split_windows(data=data_list[i], seq_length=args.seq_length)
            train_x.append(torch.Tensor(x))
            train_y.append(torch.Tensor(y))

        train_loaders = [
            DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=False)
            for x, y in zip(train_x, train_y)
        ]

        # 训练循环
        best_score = float('inf')
        early_stop = 0

        for epoch in range(args.epochs):
            model.train()
            # 训练步骤...

            # 验证步骤...

            if valid_loss.item() < best_score:
                best_score = valid_loss.item()
                torch.save(model.state_dict(), f"{args.output_dir}/{args.model_name}_best.pth")
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > args.early_stop:
                    break

        # 测试评估
        model.load_state_dict(torch.load(f"{args.output_dir}/{args.model_name}_best.pth"))
        test_x, test_y = split_windows(test_data, args.seq_length)
        test_x = torch.Tensor(test_x).to(device)
        test_y = torch.Tensor(test_y).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(test_x.unsqueeze(-1))
            pred = scalers[col].inverse_transform(pred.cpu().numpy())
            true_val = scalers[col].inverse_transform(test_y.cpu().numpy())

            predictions[col] = pred
            true_values[col] = true_val

            # 保存结果
            plt.figure()
            plt.plot(true_val, label='True')
            plt.plot(pred, label='Predicted')
            plt.title(f"{args.model_name} Prediction - Column {col}")
            plt.legend()
            plt.savefig(f"{args.output_dir}/{args.model_name}_col{col}.png")
            plt.close()

    # 保存最终预测结果
    pd.DataFrame(predictions).to_csv(f"{args.output_dir}/{args.model_name}_predictions.csv")
    pd.DataFrame(true_values).to_csv(f"{args.output_dir}/{args.model_name}_true_values.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting with LSTM")

    # 必需参数
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file")
    parser.add_argument("--model_name", type=str, default="LSTM",
                        help="Name of the model (used for saving results)")

    # 模型参数
    parser.add_argument("--input_size", type=int, default=1, help="Input feature dimension")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden layer size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers")
    parser.add_argument("--output_size", type=int, default=1, help="Output dimension")

    # 训练参数
    parser.add_argument("--seq_length", type=int, default=24, help="Sequence length for time windows")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--early_stop", type=int, default=50, help="Early stopping patience")

    # TDC参数
    parser.add_argument("--domain", type=int, required=True, help="Domain parameter for TDC")
    parser.add_argument("--distance_type", type=str, default="euclidean",
                        help="Distance metric type for TDC")
    parser.add_argument("--distance", type=float, required=True, help="Distance threshold for TDC")

    # 系统参数
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")

    args = parser.parse_args()
    train_model(args)