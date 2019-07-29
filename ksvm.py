import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import numpy as np
# 载入绘图模块
import matplotlib.pyplot as plt  # 二维绘图模块（三维图需要展示在二维画布上）
from mpl_toolkits.mplot3d import Axes3D  # 三维绘图模块

# 数据
X, y = make_blobs(centers=4, random_state=8)
y = y % 2
# 添加第二个特征的平方
X_new = np.hstack([X, X[:, 1:] ** 2])

# 使用 Axes3D() 创建 3D 图形对象。
fig = plt.figure()
ax = Axes3D(fig,elev=-152,azim=-26)

mask = (y == 0)
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# 50个点
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX,YY=np.meshgrid(xx,yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / (-coef[2])
ax.plot_surface(XX,YY,ZZ)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2],
           c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2],
           c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_zlabel("Feature1 **2")

# linear_svm = LinearSVC().fit(X, y)
# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
plt.show()
