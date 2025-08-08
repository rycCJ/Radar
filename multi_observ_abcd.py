import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. 参数设定 ---
# a) 先验分布 Theta ~ N(x0, sigma0_sq) 的参数
# 这是我们开始观测前的初始信念
x0 = 10.0          # 先验均值 (我们最初猜测 Theta 在10附近)
sigma0_sq = 9.0    # 先验方差 (我们对猜测很不确定，方差较大)

# b) 噪声 W_i ~ N(0, sigma_i_sq) 的参数
# 代表了每次观测的不确定性或测量误差
# 我们进行5次观测，每次的测量精度可能不同
sigma_i_sq_list = np.array([2.0, 5.0, 1.5, 3.0, 10.0])
n = len(sigma_i_sq_list)

# c) 设定一个用于模拟的真实 Theta 值
# 在现实中这是我们永远无法知道的未知数，这里仅用于生成模拟数据
theta_true = 15.0

print(f"--- 初始设定 ---")
print(f"先验分布: N(mean={x0}, variance={sigma0_sq})")
print(f"真实Theta (待估计): {theta_true}")
print(f"将进行 {n} 次观测，其噪声方差分别为: {sigma_i_sq_list}\n")


# --- 2. 模拟生成观测数据 ---
# 根据模型 X_i = Theta + W_i 生成数据
# np.random.normal 使用标准差，所以需要对访查进行开方
observations_x = np.zeros(n)
for i in range(n):
    noise_w = np.random.normal(0, np.sqrt(sigma_i_sq_list[i]))
    observations_x[i] = theta_true + noise_w

print(f"--- 模拟观测 ---")
print(f"生成的 {n} 个观测数据 X = {np.round(observations_x, 2)}\n")


# --- 3. 求解 b) 题: 计算后验分布 ---
# 这个过程完全基于我们在b)题推导出的公式

# a) 计算后验方差 (Posterior Variance)
# 后验精度 = 先验精度 + 所有似然的精度之和
# 精度是方差的倒数
inv_sigma_post_sq = (1/sigma0_sq) + np.sum(1/sigma_i_sq_list)
sigma_post_sq = 1 / inv_sigma_post_sq

# b) 计算后验均值 (Posterior Mean)
# mu_post = sigma_post^2 * (x0/sigma0^2 + sum(xi/sigma_i^2))
term_prior = x0 / sigma0_sq
term_likelihood = np.sum(observations_x / sigma_i_sq_list)
mu_post = sigma_post_sq * (term_prior + term_likelihood)

print(f"--- b) 后验分布结果 ---")
print(f"后验分布为: N(mean={mu_post:.4f}, variance={sigma_post_sq:.4f})\n")


# --- 4. 求解 c) 题: MMSE 和 MAP 估计 ---
# a) MMSE 估计是后验分布的均值
theta_hat_mmse = mu_post

# b) MAP 估计是后验分布的峰值点
# 因为后验是高斯分布，所以其峰值点就是均值
theta_hat_map = mu_post

print(f"--- c) 估计结果 ---")
print(f"MMSE 估计值: {theta_hat_mmse:.4f}")
print(f"MAP 估计值: {theta_hat_map:.4f}")
print("结论: 两者相等，因为后验分布是高斯分布，其均值和峰值（众数）相同。\n")


# --- 5. 求解 d) 题: 均方误差 (MSE) ---
# a) MMSE估计的MSE就是后验分布的方差
mse = sigma_post_sq
print(f"--- d) 均方误差(MSE) ---")
print(f"通用情况下的MSE (即后验方差): {mse:.4f}")

# b) 所有方差相等的特殊情况
sigma_sq_special = 4.0 # 假设所有方差都等于4
mse_special_case = sigma_sq_special / (n + 1)
print(f"假设所有方差均为{sigma_sq_special}, MSE应为 {sigma_sq_special}/({n}+1) = {mse_special_case:.4f}\n")


# --- 6. 可视化结果 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

# a) 绘制分布曲线
x_range = np.linspace(5, 20, 1000)
prior_pdf = norm.pdf(x_range, x0, np.sqrt(sigma0_sq))
posterior_pdf = norm.pdf(x_range, mu_post, np.sqrt(sigma_post_sq))

# 计算并绘制联合似然函数(作为theta的函数)
# L(theta) 正比于 exp(-0.5 * sum((xi-theta)^2/sigma_i^2))
log_likelihood = np.zeros_like(x_range)
for x_i, var_i in zip(observations_x, sigma_i_sq_list):
    log_likelihood += norm.logpdf(x_i, x_range, np.sqrt(var_i))
# 将对数似然转换为似然并归一化以便于绘图
likelihood_pdf = np.exp(log_likelihood)
likelihood_pdf /= np.trapz(likelihood_pdf, x_range) # 归一化

ax.plot(x_range, prior_pdf, 'b--', label=f'Prior: N({x0:.1f}, {sigma0_sq:.1f})', lw=2)
ax.plot(x_range, likelihood_pdf, 'g:', label=f'Likelihood (from {n} observations)', lw=2)
ax.plot(x_range, posterior_pdf, 'r-', label=f'Posterior: N({mu_post:.2f}, {sigma_post_sq:.2f})', lw=3)
ax.fill_between(x_range, posterior_pdf, color='red', alpha=0.1)


# b) 绘制关键的数值点
ax.axvline(theta_true, color='k', linestyle='-', label=f'True $\Theta$ = {theta_true:.2f}', lw=2)
ax.axvline(mu_post, color='r', linestyle='-.', label=f'Posterior Mean (Estimate) = {mu_post:.2f}', lw=2)
ax.axvline(x0, color='b', linestyle='-.', label=f'Prior Mean = {x0:.1f}', lw=2)

# c) 美化图表
ax.set_title('Bayesian Inference: From Prior to Posterior', fontsize=18)
ax.set_xlabel('Value of $\Theta$', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.legend(fontsize=12)
ax.set_ylim(bottom=0)
plt.show()