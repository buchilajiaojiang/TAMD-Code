import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from ADA.TDC import get_split_time, TDC, get_index, split_data


def input_data(seq, ws):
    """生成时间序列窗口数据集"""
    out = []
    L = len(seq)
    for i in range(L - ws):
        window = seq[i:i + ws]
        label = seq[i + ws:i + ws + 1]
        out.append((window, label))
    return out


class CNNnetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=4):
        super().__init__()
        self.conv1d = nn.Conv1d(1, input_size, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size * (args.window_size + 1 - kernel_size), hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def main(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # 数据加载与预处理
    data = pd.read_csv(args.data_path).iloc[:, 1:].values.astype(float)
    scaler = MinMaxScaler(feature_range=(args.min_val, args.max_val))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

    # 分割数据集
    split_time_list = get_split_time(args.domain, mode=args.tdc_mode, data=data,
                                     dis_type=args.distance_type, dis=args.distance)
    data_list = split_data(data, split_time_list)

    # 创建数据加载器
    window_size = args.window_size
    test_size = args.test_size
    dataset = input_data(torch.FloatTensor(data_normalized), window_size)
    test_data = dataset[-test_size:]

    # 模型配置
    model = CNNnetwork(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size
    ).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练循环
    best_score = float('inf')
    early_stop_counter = 0
    for epoch in range(args.epochs):
        model.train()
        # 训练逻辑...

        # 早停机制
        if early_stop_counter > args.early_stop:
            break

    # 模型评估
    model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    # 预测与评估逻辑...

    # 保存结果
    if args.save_results:
        plt.savefig(f"{args.output_dir}/prediction_plot.png")
        pd.DataFrame(predictions).to_csv(f"{args.output_dir}/predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting with CNN")

    # 必需参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input data file")

    # 模型参数
    parser.add_argument("--input_size", type=int, default=32,
                        help="Input feature dimension for CNN")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden layer dimension")
    parser.add_argument("--output_size", type=int, default=1,
                        help="Output dimension")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--early_stop", type=int, default=50,
                        help="Early stopping patience")

    # 数据处理参数
    parser.add_argument("--window_size", type=int, default=24,
                        help="Sliding window size for time series")
    parser.add_argument("--test_size", type=int, default=100,
                        help="Size of test set")
    parser.add_argument("--min_val", type=float, default=0.0,
                        help="Minimum value for normalization")
    parser.add_argument("--max_val", type=float, default=1.0,
                        help="Maximum value for normalization")

    # TDC参数
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain parameter for TDC")
    parser.add_argument("--tdc_mode", type=str, default="tdc",
                        help="Mode for TDC calculation")
    parser.add_argument("--distance_type", type=str, default="euclidean",
                        help="Distance metric type")
    parser.add_argument("--distance", type=float, required=True,
                        help="Distance threshold")

    # 系统参数
    parser.add_argument("--use_gpu", action="store_true",
                        help="Enable GPU acceleration")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--model_save_path", type=str,
                        default="./model/best_model.pth",
                        help="Path to save trained model")
    parser.add_argument("--save_results", action="store_true",
                        help="Save prediction results and plots")

    args = parser.parse_args()
    main(args)