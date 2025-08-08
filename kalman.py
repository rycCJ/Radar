import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.stats import norm

# --- 1. 系统设定与数据生成 (与之前相同) ---
dt = 1.0
n_steps = 60
velocity = 5.0
process_variance = 0.5**2
measurement_variance = 1.5**2
initial_mean = 0.0
initial_variance = 5.0**2

# --- 数据生成逻辑 (与之前相同) ---
true_positions = np.zeros(n_steps)
measurements = np.zeros(n_steps)
prior_means = np.zeros(n_steps)
prior_variances = np.zeros(n_steps)
posterior_means = np.zeros(n_steps)
posterior_variances = np.zeros(n_steps)
true_positions[0] = 0.0
prior_means[0] = initial_mean
prior_variances[0] = initial_variance
for t in range(n_steps):
    if t > 0:
        # true_positions[t] = true_positions[t-1] + velocity * dt + np.random.normal(0, np.sqrt(process_variance))   #线性运动模型
        
        # 定义正弦波的参数
        amplitude = 50  # 振幅
        frequency = 1   # 角频率 (控制振荡速度)  #process_variance = 5**2
        # 直接用关于时间t的函数计算当前位置，而不是依赖上一步的位置
        # 这样可以确保轨迹是我们想要的形状
        true_positions[t] = amplitude * np.sin(frequency * t * dt) + np.random.normal(0, np.sqrt(process_variance))


        # # 在 for t in range(n_steps): 循环内部
        # initial_pos = 0.0
        # initial_vel = 0.0
        # acceleration = 0.1 # 恒定的加速度
        # time = t * dt
        # true_positions[t] = initial_pos + initial_vel * time + 0.5 * acceleration * time**2 + np.random.normal(0, np.sqrt(process_variance))
        # 在 for t in range(n_steps): 循环内部


        # L = 100  # 曲线的最大值 (上限)
        # k = 0.2  # 曲线的陡峭程度
        # t0 = n_steps / 2 # 曲线的中点位置
        # time = t * dt
        # true_positions[t] = L / (1 + np.exp(-k * (time - t0))) + np.random.normal(0, np.sqrt(process_variance))


    measurements[t] = true_positions[t] + np.random.normal(0, np.sqrt(measurement_variance))
    precision_prior = 1 / prior_variances[t]
    precision_measurement = 1 / measurement_variance
    posterior_variances[t] = 1 / (precision_prior + precision_measurement)
    posterior_means[t] = posterior_variances[t] * (precision_prior * prior_means[t] + precision_measurement * measurements[t])
    if t < n_steps - 1:
        # prior_means[t+1] = posterior_means[t] + velocity * dt  #线性运动模型
        # 定义正弦波的参数
        amplitude = 50  # 振幅
        omega  = 1   # 角频率 (控制振荡速度)
        prior_means[t+1] = amplitude * np.sin(omega * (t+1) * dt) # 正弦波运动模型
        prior_variances[t+1] = posterior_variances[t] + process_variance

# --- 2. 交互式动画设置 ---
fig, (ax_traj, ax_pdf) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 3]})
fig.subplots_adjust(bottom=0.2, hspace=0.3)

# 初始化轨迹图
ax_traj.set_xlim(0, n_steps)
ax_traj.set_ylim(min(measurements)-5, max(true_positions)+5)
ax_traj.set_title('Object Trajectory Tracking', fontsize=16)
ax_traj.set_xlabel('Time Step', fontsize=12)
ax_traj.set_ylabel('Position (m)', fontsize=12)
ax_traj.grid(True)
true_line, = ax_traj.plot([], [], 'k-', label='True Position', lw=3)
measurement_points, = ax_traj.plot([], [], 'mx', label='Measurements', markersize=10)
posterior_line, = ax_traj.plot([], [], 'r-', label='Kalman Filter Estimate', lw=3)
confidence_interval_traj = ax_traj.fill_between([], [], [], color='r', alpha=0.2, label='95% Confidence Interval')
current_time_vline = ax_traj.axvline(0, color='gold', linestyle='--', lw=2, label='Current Time')
ax_traj.legend(loc='upper left', fontsize=12)

# 初始化PDF图
x_range = np.linspace(np.min(measurements) - 10, np.max(true_positions) + 10, 1000)
ax_pdf.set_title('Belief Distribution (PDF) Evolution', fontsize=16)

# 动画更新函数
def update_plots(frame):
    t = int(frame)
    # 更新轨迹图
    true_line.set_data(range(t + 1), true_positions[:t + 1])
    measurement_points.set_data(range(t + 1), measurements[:t + 1])
    posterior_line.set_data(range(t + 1), posterior_means[:t + 1])
    global confidence_interval_traj
    if confidence_interval_traj:
        confidence_interval_traj.remove()
    lower_bound = posterior_means - 2 * np.sqrt(posterior_variances)
    upper_bound = posterior_means + 2 * np.sqrt(posterior_variances)
    confidence_interval_traj = ax_traj.fill_between(range(t + 1), lower_bound[:t+1], upper_bound[:t+1], color='r', alpha=0.2)
    current_time_vline.set_xdata([t, t])

    # 更新PDF图
    ax_pdf.clear()
    ax_pdf.set_ylim(0, 0.8)
    ax_pdf.set_xlim(x_range[0], x_range[-1])
    ax_pdf.set_title(f'Belief Distribution at Time Step: {t}', fontsize=16)
    ax_pdf.set_xlabel('Position (m)', fontsize=12)
    ax_pdf.set_ylabel('Probability Density', fontsize=12)
    ax_pdf.grid(True)
    prior_pdf = norm.pdf(x_range, prior_means[t], np.sqrt(prior_variances[t]))
    ax_pdf.plot(x_range, prior_pdf, 'b--', label=f'Prior (Prediction)', lw=2)
    if t > 0:
        last_post_pdf = norm.pdf(x_range, posterior_means[t-1], np.sqrt(posterior_variances[t-1]))
        ax_pdf.plot(x_range, last_post_pdf, 'grey', linestyle=':', label=f'Last Posterior', lw=2)
    likelihood_pdf = norm.pdf(x_range, measurements[t], np.sqrt(measurement_variance))
    ax_pdf.plot(x_range, likelihood_pdf, 'g:', label=f'Likelihood (Measurement)', lw=2)
    ax_pdf.axvline(measurements[t], color='g', linestyle=':', lw=1)
    post_pdf = norm.pdf(x_range, posterior_means[t], np.sqrt(posterior_variances[t]))
    ax_pdf.plot(x_range, post_pdf, 'r-', label=f'Posterior (Update)', lw=3)
    ax_pdf.fill_between(x_range, post_pdf, color='red', alpha=0.2)
    ax_pdf.axvline(posterior_means[t], color='r', linestyle='-.', lw=1)
    ax_pdf.legend(loc='upper right')
    
    # 修正：移除这一行来打破无限递归循环
    # time_slider.set_val(t)
    
    fig.canvas.draw_idle()

# --- 3. 创建交互式控件 ---
ani = FuncAnimation(fig, update_plots, frames=range(n_steps), interval=200, repeat=False, blit=False)
ax_slider = fig.add_axes([0.15, 0.1, 0.7, 0.03])
ax_pause_button = fig.add_axes([0.7, 0.04, 0.1, 0.04])
ax_play_button = fig.add_axes([0.81, 0.04, 0.1, 0.04])
time_slider = Slider(ax=ax_slider, label='Time Step', valmin=0, valmax=n_steps - 1, valinit=0, valstep=1)
pause_button = Button(ax_pause_button, 'Pause', hovercolor='0.975')
play_button = Button(ax_play_button, 'Play', hovercolor='0.975')

def pause_animation(event):
    ani.pause()
def play_animation(event):
    ani.resume()
def slider_update(val):
    ani.pause()
    update_plots(int(val))

pause_button.on_clicked(pause_animation)
play_button.on_clicked(play_animation)
time_slider.on_changed(slider_update)

ani.pause()
update_plots(0)
plt.show()