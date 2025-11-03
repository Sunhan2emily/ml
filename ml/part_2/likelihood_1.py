import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import binom

cn_fonts = [f.name for f in font_manager.fontManager.ttflist if 'Sim' in f.name]
print("可用中文字体：", cn_fonts)

# 使用其中一个（比如 SimHei）
plt.rcParams['font.family'] = cn_fonts
plt.rcParams['axes.unicode_minus'] = False

# 假设我们抛了10次硬币，观察到7次正面
n_trials = 10
n_success = 7

def likelihood_function(p):
    """似然函数：给定正面概率p，观察到7次正面的可能性"""
    return binom.pmf(n_success, n_trials, p)

# 计算不同p值的似然
p_values = np.linspace(0, 1, 100)
likelihoods = [likelihood_function(p) for p in p_values]
print(likelihoods)
# 可视化
plt.figure(figsize=(10, 6))
plt.plot(p_values, likelihoods, 'b-', linewidth=2)
plt.axvline(x=0.7, color='red', linestyle='--', label='最大似然估计 (p=0.7)')
plt.xlabel('硬币正面概率 p')
plt.ylabel('似然值 L(p|数据)')
plt.title('似然函数: 10次抛硬币观察到7次正面')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 找到最大似然估计
mle_index = np.argmax(likelihoods)
mle_p = p_values[mle_index]
print(f"最大似然估计: p = {mle_p:.3f}")
print(f"这很直观：观察到7/10次正面，最合理的p值是0.7")