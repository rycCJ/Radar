# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
#  --- 1. FMCW 雷达参数定义 ---
# 这些参数可以根据您的具体应用进行修改
c = 3e8  # 光速 (m/s)
f_start = 77e9  # 起始频率 (Hz), 77GHz
bandwidth = 1e9  # 调频带宽 (Hz), 1GHz
T_chirp = 40e-6  # 单个调频脉冲持续时间 (s), 40us
num_samples = 1024  # 每个Chirp的采样点数
num_chirps = 256   # 帧中的Chirp数量

# 计算衍生参数
f_c = f_start + bandwidth / 2  # 中心频率
lambda_c = c / f_c             # 中心频率对应波长
sweep_slope = bandwidth / T_chirp  # 调频斜率
fs = num_samples / T_chirp     # 采样率

# %%
# --- 2. 模拟测试数据 ---
# 假设场景中有两个移动目标
# target = [距离(m), 速度(m/s), 信号强度]
targets_truth = [
    [50, 10, 1],    # 目标1: 50米远, 速度10m/s (约36 km/h)
    [80, -15, 0.8]  # 目标2: 80米远, 速度-15m/s (约-54 km/h, 正在靠近)
]

# 创建时间和chirp索引
t = np.linspace(0, T_chirp, num_samples)  # 快速时间轴 (行)
chirp_idx = np.arange(num_chirps)        # 慢速时间轴 (列)
raw_data = np.zeros((num_chirps, num_samples), dtype=np.complex64)

print("正在生成模拟数据...")
# 为每个目标生成信号并叠加
for r, v, snr in targets_truth:
    # 计算差拍频率 (对应距离)
    t_delay = 2 * r / c
    f_beat = sweep_slope * t_delay
    
    # 计算多普勒频移 (对应速度)
    f_doppler = 2 * v / lambda_c
    
    # 生成信号
    phase_beat = 2j * np.pi * f_beat * t
    phase_doppler = 2j * np.pi * f_doppler * T_chirp * chirp_idx
    
    # 叠加每个目标的信号
    raw_data += snr * np.exp(np.outer(phase_doppler, np.ones(num_samples))) * \
                    np.exp(np.outer(np.ones(num_chirps), phase_beat))

# 添加高斯白噪声
noise_power = 0.05
raw_data += noise_power * (np.random.randn(*raw_data.shape) + 1j * np.random.randn(*raw_data.shape))

print("数据生成完毕。")


# %%
# ======================================================================
# --- 新增的可视化环节 ---
# ======================================================================
print("开始生成信号分析图...")

# --- 可视化1：单个目标的差拍信号 (体现距离) ---
# 我们只看第一个目标，在第一个chirp时的信号
r1, v1, snr1 = targets_truth[0]
t_delay1 = 2 * r1 / c
f_beat1 = sweep_slope * t_delay1
# 只为第一个目标生成无噪声的信号 (在第一个chirp, n=0)
signal_target1_chirp1 = snr1 * np.exp(2j * np.pi * f_beat1 * t)

plt.figure(figsize=(12, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(t * 1e6, np.real(signal_target1_chirp1)) # t转换为微秒(us)
plt.title(f'可视化1: 单个目标(50m)在一个Chirp内的差拍信号\n(信号频率决定距离)', fontsize=16)
plt.xlabel('时间 (us)', fontsize=12)
plt.ylabel('信号幅度', fontsize=12)
plt.grid(True)
plt.text(5, 0.8, f'差拍频率 ≈ {f_beat1/1e6:.2f} MHz', fontsize=12, color='red')
plt.show()


# --- 可视化2：多目标叠加与噪声 (单个Chirp的最终形态) ---
# 计算第二个目标的信号
r2, v2, snr2 = targets_truth[1]
t_delay2 = 2 * r2 / c
f_beat2 = sweep_slope * t_delay2
signal_target2_chirp1 = snr2 * np.exp(2j * np.pi * f_beat2 * t)
# 两个目标信号的纯净叠加
clean_sum_chirp1 = signal_target1_chirp1 + signal_target2_chirp1
# 从我们生成的最终数据中，取出第一个chirp的数据（已包含噪声）
final_noisy_chirp1 = raw_data[0, :]

plt.figure(figsize=(12, 8))
plt.plot(t * 1e6, np.real(clean_sum_chirp1), label='目标1+目标2 (纯净叠加)')
plt.plot(t * 1e6, np.real(final_noisy_chirp1), label='最终信号 (含噪声)', alpha=0.7)
plt.title('可视化2: 多目标信号叠加与噪声的影响 (单个Chirp)', fontsize=16)
plt.xlabel('时间 (us)', fontsize=12)
plt.ylabel('信号幅度', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# --- 可视化3：跨Chirp的相位变化 (体现速度) ---
# 我们在时间轴上选择一个固定的采样点，观察它在所有chirp中的信号变化
# 这个变化就体现了多普勒效应
sample_index_for_doppler = 50 # 随便选一个点
# 提取这个采样点在所有chirp上的信号值（这被称为“慢时间”信号）
slow_time_signal = raw_data[:, sample_index_for_doppler]

plt.figure(figsize=(12, 8))
plt.plot(chirp_idx, np.real(slow_time_signal))
plt.title(f'可视化3: 固定采样点(第{sample_index_for_doppler}个)在所有Chirp上的信号变化\n(信号频率决定速度)', fontsize=16)
plt.xlabel('Chirp 序号', fontsize=12)
plt.ylabel('信号幅度', fontsize=12)
plt.grid(True)
plt.show()

print("信号分析图生成完毕。")


# %%
# --- 3. FMCW 信号处理 (FFT部分) ---
# (这部分代码保持不变)
print("第一步: 正在执行距离维FFT...")
window_range = np.hanning(num_samples)
range_fft_data = np.fft.fft(raw_data * window_range, axis=1)

print("第二步: 正在执行速度维FFT...")
window_doppler = np.hanning(num_chirps)
doppler_fft_data = np.fft.fft(range_fft_data * window_doppler[:, np.newaxis], axis=0)

rdm = np.fft.fftshift(doppler_fft_data, axes=0)
print("第三步: 正在生成RDM效果图...")
rdm_db = 20 * np.log10(np.abs(rdm))
rdm_db[np.isneginf(rdm_db)] = 0

range_res = c / (2 * bandwidth)
range_axis = np.arange(num_samples / 2) * range_res
rdm_plot = rdm_db[:, :int(num_samples / 2)]

doppler_res = lambda_c / (2 * num_chirps * T_chirp)
velocity_axis = np.arange(-num_chirps / 2, num_chirps / 2) * doppler_res

# %%
# --- 4. RDM (距离多普勒图) 可视化 ---
# (这部分代码保持不变)
plt.figure(figsize=(12, 8))
plt.imshow(rdm_plot, 
           aspect='auto', 
           extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
           cmap='jet')

plt.title('距离-多普勒图 (Range-Doppler Map)', fontsize=16)
plt.xlabel('距离 (m)', fontsize=16)
plt.ylabel('速度 (m/s)', fontsize=16)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# for r, v, _ in targets_truth:
#     plt.plot(r, v, 'rx', markersize=10, markeredgewidth=2, label=f'真值: ({r}m, {v}m/s)')
plt.legend(loc='upper right') # 移除facecolor以适应浅色背景
plt.colorbar(label='信号强度 (dB)') # 把colorbar加回来
plt.show()

print("处理完成。")