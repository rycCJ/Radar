import numpy as np
import matplotlib.pyplot as plt
# 假设 simulate_chirps 函数已在此处定义，同Day 1

def process_to_rdm(raw_data, radar_params):
    """
    将原始数据处理成距离-多普勒图 (RDM)
    """
    num_chirps = radar_params['num_chirps']
    num_samples = radar_params['num_samples']
    
    # 1. 距离维 FFT
    range_fft_data = np.fft.fft(raw_data * np.hanning(num_samples), axis=1)
    
    # 2. 速度维 FFT
    doppler_fft_data = np.fft.fft(range_fft_data * np.hanning(num_chirps)[:, np.newaxis], axis=0)
    
    # 3. 速度维移位
    rdm = np.fft.fftshift(doppler_fft_data, axes=0)
    
    # 4. 转换为dB
    rdm_db = 20 * np.log10(np.abs(rdm) + 1e-6) # 加小量防止log(0)
    
    # 5. 计算坐标轴
    # 距离轴
    range_res = radar_params['c'] / (2 * radar_params['bandwidth'])
    range_axis = np.arange(num_samples / 2) * range_res
    
    # 速度轴
    doppler_res = radar_params['lambda_c'] / (2 * num_chirps * radar_params['T_chirp'])
    velocity_axis = np.arange(-num_chirps / 2, num_chirps / 2) * doppler_res
    
    # 只返回RDM的前半部分（对应正距离）
    return rdm_db[:, :num_samples // 2], range_axis, velocity_axis

if __name__ == '__main__':
    # ... 此处省略Day 1的参数定义和数据模拟部分 ...
    # 假设 raw_data 已通过 simulate_chirps 生成
    
    radar_params = {
        'c': 3e8, 'f_start': 77e9, 'bandwidth': 1e9,
        'T_chirp': 40e-6, 'num_samples': 1024, 'num_chirps': 256,
    }
    radar_params['f_c'] = radar_params['f_start'] + radar_params['bandwidth'] / 2
    radar_params['lambda_c'] = radar_params['c'] / radar_params['f_c']
    radar_params['sweep_slope'] = radar_params['bandwidth'] / radar_params['T_chirp']
    
    targets_truth = [[50, 10, 1]] # 用一个简单目标测试
    raw_data = simulate_chirps(radar_params, targets_truth, noise=False)


    # --- 调用RDM处理函数 ---
    rdm_plot, range_axis, velocity_axis = process_to_rdm(raw_data, radar_params)

    # --- 绘图 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8))
    plt.imshow(rdm_plot, 
               aspect='auto', 
               extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
               cmap='jet')

    plt.title('距离-多普勒图 (RDM)', fontsize=16)
    plt.xlabel('距离 (m)', fontsize=12)
    plt.ylabel('速度 (m/s)', fontsize=12)
    plt.colorbar(label='信号强度 (dB)')
    plt.show()