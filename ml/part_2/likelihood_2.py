import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

cn_fonts = [f.name for f in font_manager.fontManager.ttflist if 'Sim' in f.name]

# 使用其中一个（比如 SimHei）
plt.rcParams['font.family'] = cn_fonts
plt.rcParams['axes.unicode_minus'] = False

# 我们看到：10次抛掷，7次正面
# 画图看看哪个硬币最可能产生这个结果

p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
likelihoods = []

for p in p_values:
    # 计算这个硬币产生"7正3反"的可能性
    likelihood = (p**7) * ((1-p)**3)
    likelihoods.append(likelihood)

plt.figure(figsize=(10, 5))
bars = plt.bar([str(p) for p in p_values], likelihoods, color='lightblue')

# 标出最可能的那个
most_likely_index = np.argmax(likelihoods)
bars[most_likely_index].set_color('red')

plt.xlabel('硬币的正面概率')
plt.ylabel('产生"7正3反"的可能性')
plt.title('猜谜游戏：哪个硬币最可能产生我们的观测结果？')
plt.show()

print(f"最大似然估计：正面概率 = {p_values[most_likely_index]}")
print("这很直观：看到7/10次正面，最合理的猜测就是p=0.7！")