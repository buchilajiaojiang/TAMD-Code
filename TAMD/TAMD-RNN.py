import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ADA.TDC import get_index, get_split_time, split_data
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def input_data(seq, ws):
    out_seq, out_label = [], []
    for i in range(len(seq) - ws):
        out_seq.append(seq[i:i + ws])
        out_label.append(seq[i + ws])
    return torch.stack(out_seq), torch.stack(out_label)


def main(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # 数据加载
    df = pd.read_csv(args.data_path, parse_dates=True, index_col='date')
    time_index = df.index

    # 数据预处理
    scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(len(df.columns))]
    scaled_data = [scaler.fit_transform(df.iloc[:, i].values.reshape(-1, 1))
                   for i, scaler in enumerate(scalers)]

    # 模型初始化
    model = SimpleRNN(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练和预测结果存储
    predictions = {}
    true_values = {}

    for col in tqdm(range(len(df.columns)), desc=f"Training {args.model_name}"):
        # 数据准备
        data = scaled_data[col]
        train_data = data[:-300]
        valid_data = data[-300:-150]
        test_data = data[-150:]

        # TDC分割
        split_time_list = get_split_time(args.domain, mode='tdc',
                                         data=df.iloc[:, col],
                                         dis_type=args.distance_type,
                                         dis=args.distance)
        data_list = split_data(df.iloc[:, col], split_time_list)

        # 数据加载器
        train_loaders = []
        for ts in data_list[:-300]:
            x, y = input_data(torch.FloatTensor(ts.values), args.window_size)
            train_loaders.append(DataLoader(
                TensorDataset(x, y),
                batch_size=args.batch_size,
                shuffle=True
            ))

        # 训练循环
        best_score = float('inf')
        early_stop = 0

        for epoch in range(args.epochs):
            model.train()
            # 训练步骤...

            # 验证步骤...

            if valid_loss < best_score:
                best_score = valid_loss
                torch.save(model.state_dict(), f"{args.output_dir}/{args.model_name}_best.pth")
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > args.early_stop:
                    break

        # 测试评估
        model.load_state_dict(torch.load(f"{args.output_dir}/{args.model_name}_best.pth"))
        x_test, y_test = input_data(torch.FloatTensor(test_data), args.window_size)
        with torch.no_grad():
            pred = model(x_test.to(device))
            pred = scalers[col].inverse_transform(pred.cpu().numpy())
            true_val = scalers[col].inverse_transform(y_test.numpy())

            predictions[col] = pred
            true_values[col] = true_val

            # 可视化保存
            plt.figure()
            plt.plot(true_val, label='True')
            plt.plot(pred, label='Predicted')
            plt.title(f"{args.model_name} Prediction - Column {col}")
            plt.legend()
            plt.savefig(f"{args.output_dir}/{args.model_name}_col{col}.png")
            plt.close()

    # 保存最终结果
    pd.DataFrame(predictions).to_csv(f"{args.output_dir}/{args.model_name}_predictions.csv")
    pd.DataFrame(true_values).to_csv(f"{args.output_dir}/{args.model_name}_true_values.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting with RNN")

    # 必需参数
    parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
    parser.add_argument("--model_name", type=str, default="RNN", help="模型名称")

    # 模型参数
    parser.add_argument("--input_size", type=int, default=1, help="输入特征维度")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--output_size", type=int, default=1, help="输出维度")

    # 训练参数
    parser.add_argument("--window_size", type=int, default=24, help="时间窗口大小")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=200, help="训练周期数")
    parser.add_argument("--early_stop", type=int, default=50, help="早停耐心值")

    # TDC参数
    parser.add_argument("--domain", type=int, required=True, help="领域参数")
    parser.add_argument("--distance_type", type=str, default="adv", help="距离类型")
    parser.add_argument("--distance", type=float, required=True, help="距离阈值")

    # 系统参数
    parser.add_argument("--use_gpu", action="store_true", help="启用GPU加速")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")

    args = parser.parse_args()
    main(args)