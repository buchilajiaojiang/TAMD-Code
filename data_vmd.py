import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from vmdpy.vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def hurst_exponent(time_series):
    """Calculate the Hurst exponent of a time series."""
    n = len(time_series)
    range_values = np.ptp(time_series)
    mean_values = np.mean(time_series)
    cumsum = np.cumsum(time_series - mean_values)

    std_dev_cumsum = np.zeros(n)
    for i in range(1, n + 1):
        windowed_cumsum = cumsum[:i]
        windowed_std_dev = np.std(windowed_cumsum)
        std_dev_cumsum[i - 1] = windowed_std_dev

    log_window_sizes = np.log(np.arange(1, n + 1))
    log_std_dev_cumsum = np.log(std_dev_cumsum + 1e-10)
    slope, _ = np.polyfit(log_window_sizes, log_std_dev_cumsum, 1)
    return slope


def preprocess_data(data, min_val, max_val):
    """数据预处理：异常值处理和平滑"""
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    original_data = scaler.fit_transform(data.reshape(-1, 1))

    # 异常值处理
    pop = data[(data <= min_val) & (data >= max_val)]
    resid = data[(data > min_val) | (data < max_val)]
    resid.fillna(0, inplace=True)
    smoothed_df = pop.interpolate(method='linear')

    # 时间序列插值
    time = np.arange(0, len(data))
    interp_function = interp1d(time, smoothed_df.values.reshape(-1), kind='linear')
    interp_data = interp_function(time)

    # 逆归一化并计算残差
    interp_data = scaler.inverse_transform(interp_data.reshape(-1, 1)).reshape(-1)
    resid_1 = resid.values.reshape(-1)
    nonzero_idx = np.nonzero(resid_1)
    resid_1[nonzero_idx] -= interp_data[nonzero_idx]

    return scaler, interp_data, resid_1


def optimize_vmd_params(data, K_range, alpha_range, tau=0, DC=1, init=1, tol=1e-6):
    """优化VMD参数"""
    min_entropy = float('inf')
    min_mse = float('inf')
    K_opt = K_range[0]
    alpha_opt = alpha_range[0]

    for K in range(K_range[0], K_range[1] + 1):
        for alpha in range(alpha_range[0], alpha_range[1] + 1, 100):
            try:
                modes, _, _ = VMD(data, alpha, tau, K, DC, init, tol)
                total_mode = np.sum(modes, axis=0)
                mse = mean_squared_error(data[:-1], total_mode)

                if mse < min_mse:
                    min_mse = mse
                    K_opt = K
                    alpha_opt = alpha
            except:
                continue

    return K_opt, alpha_opt


def vmd_reconstruction(data, K, alpha, resid, weight_method='energy'):
    """VMD分解与加权重构"""
    # 执行VMD分解
    modes, _, _ = VMD(data, alpha, tau=0, K=K, DC=1, init=1, tol=1e-6)

    # 权重计算
    if weight_method == 'energy':
        variances = [np.var(mode) for mode in modes]
        total_variance = sum(variances)
        weights = [v / total_variance for v in variances]
    else:  # 等权重
        weights = [1 / K] * K

    # 加权叠加
    weighted_mode = np.sum([w * mode for w, mode in zip(weights, modes)], axis=0)

    # 合并残差
    final_output = weighted_mode + resid[:len(weighted_mode)]
    return final_output


def main(args):
    # 加载数据
    df = pd.read_csv(args.data_path, header=0).iloc[:, 1:]
    data = df.values.reshape(-1)

    # 计算统计量
    print("Hurst Exponent:", hurst_exponent(data))
    print("Data Statistics:")
    print(df.describe())

    # 数据预处理
    scaler, interp_data, resid = preprocess_data(data, args.min_val, args.max_val)

    # 优化VMD参数
    K_opt, alpha_opt = optimize_vmd_params(
        interp_data,
        K_range=(args.K_min, args.K_max),
        alpha_range=(args.alpha_min, args.alpha_max)
    )
    print(f"Optimal Parameters: K={K_opt}, alpha={alpha_opt}")

    # 两种重构方法对比
    recon_equal = vmd_reconstruction(interp_data, K_opt, alpha_opt, resid, 'equal')
    recon_energy = vmd_reconstruction(interp_data, K_opt, alpha_opt, resid, 'energy')

    # 评估指标
    target = scaler.transform(data.reshape(-1, 1))[:-1].reshape(-1)
    print("\nEqual Weight Reconstruction:")
    print("MAE:", mean_absolute_error(target, recon_equal))
    print("MSE:", mean_squared_error(target, recon_equal))

    print("\nEnergy Weighted Reconstruction:")
    print("MAE:", mean_absolute_error(target, recon_energy))
    print("MSE:", mean_squared_error(target, recon_energy))

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(target, label='Original')
    plt.plot(recon_equal, label='Equal Weight')
    plt.plot(recon_energy, label='Energy Weight')
    plt.legend()
    plt.savefig(args.output_path)
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMD-based Time Series Processing')
    parser.add_argument('--data_path', type=str, required=True, help='Input data file path')
    parser.add_argument('--output_path', type=str, default='result.png', help='Output figure path')
    parser.add_argument('--min_val', type=float, required=True, help='Minimum valid value')
    parser.add_argument('--max_val', type=float, required=True, help='Maximum valid value')
    parser.add_argument('--K_min', type=int, default=2, help='Minimum K for VMD')
    parser.add_argument('--K_max', type=int, default=10, help='Maximum K for VMD')
    parser.add_argument('--alpha_min', type=int, default=1000, help='Minimum alpha for VMD')
    parser.add_argument('--alpha_max', type=int, default=3000, help='Maximum alpha for VMD')

    args = parser.parse_args()
    main(args)
