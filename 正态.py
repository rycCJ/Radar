# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# %%
# 参数设定
mu_x = 0        # 先验均值
sigma_x = 1     # 先验标准差
sigma_r = 1     # 噪声标准差
y_obs = 1       # 观测值 Y = x + r

# 先验分布 P(x)
x_vals = np.linspace(-4, 4, 1000)
prior = norm.pdf(x_vals, mu_x, sigma_x)

# 似然函数 P(y | x) = P(r = y - x)
likelihood = norm.pdf(y_obs - x_vals, 0, sigma_r)

# 非归一化后验
unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, x_vals)  # 归一化

# 精确后验分布参数（根据公式）
sigma_post_sq = 1 / (1 / sigma_x**2 + 1 / sigma_r**2)
mu_post = sigma_post_sq * (mu_x / sigma_x**2 + y_obs / sigma_r**2)
posterior_exact = norm.pdf(x_vals, mu_post, np.sqrt(sigma_post_sq))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_vals, prior, label='Prior $P(x)$', linestyle='--')
plt.plot(x_vals, likelihood, label='Likelihood $P(y=1 | x)$', linestyle='--')
plt.plot(x_vals, posterior, label='Posterior $P(x | y=1)$ (computed)', linewidth=2)
plt.plot(x_vals, posterior_exact, label='Posterior $P(x | y=1)$ (exact)', linestyle=':')
plt.title('Bayesian Inference: Prior, Likelihood, and Posterior')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 设定参数

# --- 先验分布 P(x) 的参数 ---
# 我们对 x 的初始信念。假设我们认为 x 可能在 5 附近，但不太确定。
mu_x = 5.0
sigma_x_sq =0.5  # 方差较大，代表不确定性高

# --- 噪声 r ~ N(0, s^2) 的参数 ---
# 这是我们测量过程的不确定性。方差越小，测量越准。
s_sq = 1.0  # 噪声方差

# --- 模拟真实情况 ---
# 在现实世界中我们永远不知道真实值，但为了模拟，我们先设定一个
x_true = 8.0

# 2. 模拟一次观测

# 根据 Y = x + r 生成一个观测值 Y
# np.random.normal 的第二个参数是标准差，所以需要开方
noise = np.random.normal(0, np.sqrt(s_sq))
Y = x_true + noise

print(f"真实状态 (我们想知道的): x_true = {x_true:.2f}")
print(f"测量噪声: r = {noise:.2f}")
print(f"我们得到的观测值: Y = {Y:.2f}\n")


# 3. 贝叶斯更新 (核心计算)

# 使用我们之前推导出的公式计算后验分布的均值和方差
# 后验均值是先验均值和观测值的加权平均
mu_posterior = (sigma_x_sq / (sigma_x_sq + s_sq)) * Y + (s_sq / (sigma_x_sq + s_sq)) * mu_x

# 后验方差比先验和噪声方差都小
sigma_posterior_sq = (sigma_x_sq * s_sq) / (sigma_x_sq + s_sq)

print("--- 贝叶斯更新结果 ---")
print(f"后验均值 (对x的最终估计): μ_x|Y = {mu_posterior:.2f}")
print(f"后验方差 (最终估计的不确定性): σ_x|Y^2 = {sigma_posterior_sq:.2f}")


# 4. 可视化结果

# 创建一个 x 值的范围用于绘图
x_range = np.linspace(0, 15, 500)
sigma_x = np.sqrt(sigma_x_sq)
s = np.sqrt(s_sq)
sigma_posterior = np.sqrt(sigma_posterior_sq)


# 计算各个分布的概率密度函数 (PDF)
prior_pdf = norm.pdf(x_range, mu_x, sigma_x)
# 似然函数 P(Y|x) 是 x 的函数，其形状由 Y 和噪声方差 s^2 决定
# 它的峰值位于我们观测到的 Y
likelihood_pdf = norm.pdf(x_range, Y, s)
posterior_pdf = norm.pdf(x_range, mu_posterior, sigma_posterior)


# 开始绘图
plt.figure(figsize=(12, 7))
plt.plot(x_range, prior_pdf, label=f'Prior P(x): N({mu_x:.1f}, {sigma_x_sq:.1f})', color='royalblue', lw=2)
plt.plot(x_range, likelihood_pdf, label=f'Likelihood P(Y|x): N(Y={Y:.2f}, {s_sq:.1f})', color='darkorange', lw=2, linestyle='--')
plt.plot(x_range, posterior_pdf, label=f'Posterior P(x|Y): N({mu_posterior:.2f}, {sigma_posterior_sq:.2f})', color='forestgreen', lw=3)

# 绘制真实值和观测值的位置
plt.axvline(x=x_true, color='red', linestyle=':', label=f'True Value x_true = {x_true:.2f}')
plt.axvline(x=Y, color='gold', linestyle=':', label=f'Observation Y = {Y:.2f}')


# 美化图表
plt.title('Bayesian Inference: Updating Beliefs with Data', fontsize=16)
plt.xlabel('Value of x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# %%
