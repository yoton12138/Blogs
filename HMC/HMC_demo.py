# -*- coding: utf-8 -*-
"""
Created on Fri Sep  24 18:22:50 2021
HMC参数估计
@author: Yoton12138
"""

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import elementwise_grad as egrad

np.random.seed(1234)                          # 用来复现结果
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负数的负号显示问题

#  首先创建模型和一些数据, 根据观测数据推断模型的参数a,b.
a = 10
b = 0.2
noise = 0.2

f = lambda x: a*x**b + 450
x_obs = np.linspace(0.1, 4, 40)
y_true = f(x_obs)
y_obs = y_true + noise*np.random.randn(1, x_obs.shape[0])
plt.figure(0)
plt.plot(x_obs, y_true, label="真实值")
plt.scatter(x_obs, y_obs, label="观测值", marker="*", color="orange")
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

# 定义一些必须的函数


def lognorm_pdf(x, mu=0., sigma=1., epsilon=1e-8):
    """
    :param x:变量值
    :param mu: 均值
    :param sigma: 标准差
    :param epsilon: 1e-8 数值稳定性
    :return: x的对数概率密度
    """
    pdf = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2))
    logpdf = np.log(pdf + epsilon)
    return logpdf

def uuu(q):
    """
    :param q: 位置变量，即感兴趣的参数（本例中的a,b）
    :return:  负对数似然（势能）
    """
    obj = y_obs - (q[0]*x_obs**q[1] + 450)
    obj_prob = -lognorm_pdf(obj)
    nll = np.sum(obj_prob)
    return nll


u_q = egrad(uuu)       # 势能函数对位置的梯度，function

# 蒙特卡洛模拟的参数
n = 0                  # 总采样次数
accepted = 0
n_var = 2
SampleNum = 500        # 期望接受样本数
burn = 0.5             # 燃烧期
# HMC参数
L = 30                 # 跳步数
step_size = 0.01       # 步长    这两个变量的敏感性很高
mu = np.zeros(n_var)   # 动量均值
M = np.identity(n_var) # 质量矩阵，即协方差
scale = 1.0

q = np.random.randn(2)  # 初始位置
q = np.expand_dims(q, axis=1)

# 执行 HMC
while (accepted < SampleNum):
    p_0 = mvn.rvs(mu, scale*M)  # 辅助动量采样 标准正态
    p_1 = np.copy(p_0)
    q_0 = q[:, -1]
    q_1 = np.copy(q_0)

    # leapfrog integration
    for s in range(L):
        p_1 -= step_size * u_q(q_1) / 2
        q_1 += step_size * p_1
        p_1 -= step_size * u_q(q_1) / 2

    p_1 = -p_1  # 可有可无，二次形式的动能正负号没有区别

    uq_0 = uuu(q_0)
    uq_1 = uuu(q_1)

    kp_0 = np.sum(0.5 * np.dot(p_0, p_0))
    kp_1 = np.sum(0.5 * np.dot(p_1, p_1))
    # exp(-U(q*) + U(q) - K(p*) + K(p)) 即 exp(-H(q*,p*)-(-H(q,p)))
    H_1 = -uq_1 - kp_1
    H_0 = -uq_0 - kp_0
    H = H_1 - H_0

    u = np.log(np.random.uniform())
    alpha = min(0, H)

    if u < alpha:
        q = np.column_stack((q, q_1))
        accepted += 1
        if accepted % 100 == 0:
            print("已采集并接受{}个样本".format(accepted))
    else:
        q = np.column_stack((q, q_0))
    n += 1

q_burned = q[:, int(burn*SampleNum):]
q_mean = np.mean(q_burned, axis=1)
q_std = np.std(q_burned, axis=1)
print("**********************************")
print("接受率:{}".format(accepted/n))
print("参数a的均值为  {}, 标准差为  {}".format(round(q_mean[0], 4), round(q_std[0], 4)))
print("参数b的均值为  {}, 标准差为  {}".format(round(q_mean[1], 4), round(q_std[1], 4)))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(q[0, :], q[1, :])
axes[0].grid()
axes[0].set_title("模拟的马尔可夫链,接受率为:" + str(round(accepted/n, 2)))
axes[0].set_xlabel("a")
axes[0].set_ylabel("b")
axes[1].plot(q[0, :], q[1, :])
axes[1].grid()
axes[1].set_xlim([9.25, 10.75])
axes[1].set_ylim([0.125, 0.275])
axes[1].set_title("模拟的典型集")
axes[1].set_xlabel("a")
axes[1].set_ylabel("b")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("各参数的边缘分布")
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)  # 相当于格子分成3行3列,列跨度为3，行跨度为1
ax1.hist(q_burned[0, :], bins=15, edgecolor='black', rwidth=0.7)
ax1.set_xlabel("a")
ax1.set_ylabel("频数")
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=1)
ax2.plot(q_burned[0, :])
ax2.set_ylabel("a")
ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
ax3.hist(q_burned[1, :], bins=15, edgecolor='black', rwidth=0.7)
ax3.set_xlabel("b")
ax3.set_ylabel("频数")
ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2, rowspan=1)
ax4.plot(q_burned[1, :])
ax4.set_ylabel("b")
plt.tight_layout()
plt.show()
