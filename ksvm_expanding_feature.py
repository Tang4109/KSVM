import sklearn.datasets
import mglearn
import matplotlib.pyplot
import sklearn.svm
import numpy as np
import mpl_toolkits.mplot3d

X, y = sklearn.datasets.make_blobs(centers=4, random_state=8)
y = y % 2
# 添加第二个特征的平方
X_new = np.hstack([X, X[:, 1:] ** 2])
figure = matplotlib.pyplot.figure()
ax = mpl_toolkits.mplot3d.Axes3D(figure, elev=-152, azim=-26)
mask = (y == 0)
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
matplotlib.pyplot.show()
