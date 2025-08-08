import numpy as np
import matplotlib.pyplot as plt
# 1.创建一个名为 radar_simulator.py 的文件。

# 2.重构代码: 将雷达参数封装到一个字典 radar_params 中。

# 3.函数化: 编写一个名为 simulate_chirps 的函数，该函数接收 radar_params 字典和 targets 列表作为输入，返回生成的 raw_data 矩阵。这样可以方便地重复使用和测试。

# 4.多目标模拟: 使用您编写的函数，模拟一个包含三个目标的场景：

# 目标A: 距离30m，速度 5m/s, 信号强度 1.0

# 目标B: 距离60m，速度 -10m/s, 信号强度 0.7

# 目标C (静止 clutter): 距离90m, 速度 0m/s, 信号强度 1.5

# 5.可视化验证:

# 绘制出最终 raw_data 的实部图像。

# （挑战任务）取出第一个Chirp的数据（raw_data[0, :]），对其做一维FFT，并绘制出其幅度谱，看看是否能找到分别对应三个目标距离的三个频率峰值。
def simulate_chirps(radar_params, targets, noise=False):
    """
    模拟生成FMCW雷达的原始数据矩阵
    :param radar_params: 包含雷达参数的字典
    :param targets: 目标列表，每个目标格式为 [range, velocity, snr]
    :param noise: 是否添加噪声
    :return: (num_chirps, num_samples) 大小的复数矩阵 raw_data
    """
    # 从字典中解包参数
    c = radar_params['c']
    T_chirp = radar_params['T_chirp']
    num_samples = radar_params['num_samples']
    num_chirps = radar_params['num_chirps']
    sweep_slope = radar_params['sweep_slope']
    lambda_c = radar_params['lambda_c']

    # 创建时间和chirp的二维网格
    t_grid, chirp_idx_grid = np.meshgrid(
        np.linspace(0, T_chirp, num_samples),
        np.arange(num_chirps)
    )
    
    raw_data = np.zeros((num_chirps, num_samples), dtype=np.complex64)
    print("正在生成模拟数据...")

    for r, v, snr in targets:
        f_beat = sweep_slope * (2 * r / c)
        f_doppler = 2 * v / lambda_c
        phase_grid = 2j * np.pi * (f_beat * t_grid + f_doppler * T_chirp * chirp_idx_grid)
        raw_data += snr * np.exp(phase_grid)

    if noise:
        noise_power = 0.05
        raw_data += noise_power * (np.random.randn(*raw_data.shape) + 1j * np.random.randn(*raw_data.shape))
    
    print("数据生成完毕。")
    return raw_data

if __name__ == '__main__':
    # --- 1. 参数设置 ---
    radar_params = {
        'c': 3e8, 'f_start': 77e9, 'bandwidth': 1e9,
        'T_chirp': 40e-6, 'num_samples': 1024, 'num_chirps': 256,
    }
    radar_params['f_c'] = radar_params['f_start'] + radar_params['bandwidth'] / 2
    radar_params['lambda_c'] = radar_params['c'] / radar_params['f_c']
    radar_params['sweep_slope'] = radar_params['bandwidth'] / radar_params['T_chirp']
    radar_params['fs'] = radar_params['num_samples'] / radar_params['T_chirp']

    # --- 2. 目标定义 ---
    targets_truth = [
        [30, 5, 1.0],
        [60, -10, 0.7],
        [90, 0, 1.5]
    ]

    # --- 3. 生成数据 ---
    raw_data = simulate_chirps(radar_params, targets_truth)
    
    # --- 4. 可视化验证 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制raw_data实部
    plt.figure(figsize=(10, 6))
    plt.imshow(np.real(raw_data), aspect='auto')
    plt.title('raw_data 信号实部')
    plt.xlabel('快时间采样点')
    plt.ylabel('Chirp 序号')
    plt.colorbar()
    plt.show()

    # 挑战任务：验证单个Chirp的距离FFT
    single_chirp_data = raw_data[0, :]
    range_fft = np.fft.fft(single_chirp_data * np.hanning(radar_params['num_samples']))
    range_fft_magnitude = np.abs(range_fft)
    
    # 计算距离轴
    range_axis = np.arange(radar_params['num_samples']) * (radar_params['fs'] * radar_params['c']) / (2 * radar_params['sweep_slope'] * radar_params['num_samples'])


    plt.figure(figsize=(10, 6))
    plt.plot(range_axis[:radar_params['num_samples']//2], range_fft_magnitude[:radar_params['num_samples']//2])
    plt.title('单个Chirp的距离频谱')
    plt.xlabel('距离 (m)')
    plt.ylabel('幅度')
    plt.grid(True)
    plt.show()