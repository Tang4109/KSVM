from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1加载数据
cancer = load_breast_cancer()
# 2切割数据
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 3归一化
min_on_training = X_train.min(axis=0)  # 求列最小
max_on_training = X_train.max(axis=0)  # 求列最大
range_on_training = max_on_training - min_on_training
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
#4训练模型
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
score_train = svc.score(X_train_scaled, y_train)
score_test = svc.score(X_test_scaled, y_test)
print(score_train, score_test)
