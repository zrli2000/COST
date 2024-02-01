import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# for single-label classification
from base import COST_Classifier as COST_Classifier_single
# for multi-label classification
from base_multi import COST_Classifier as COST_Classifier_multi
from sklearn.model_selection import train_test_split

#! Note that the labels should start from 0: y = [0,2,4,1,3,1,2...] but not y = [1,3,5,2,4,2,3...]

# Example 1: single-label classification
# load dataset
columns = [ 'Led1', 'Led2', 'Led3', 'Led4', 'Led5', 'Led6', 'Led7', 'class']
data = pd.read_csv('led7digit.csv', delimiter=',')
X = data[data.columns[:-1]].values
y = data['class'].values

# split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2)

# train model and predict
cost = COST_Classifier_single()
b1 = 100
b2 = 10
unique_y_nums = len(np.unique(y))

# subspaces generation
X_train, X_valid, X_test = cost.feature_subset_generation(X_train, X_valid, X_test, y_train, b1, b2)

# predict: the training process in included in the predict function
# regular: the predicted labels with no reject and refine option
# regular_reject: the predicted labels with reject option
# regular_refine: the predicted labels with refine option
# y_pred_reject_refine: the predicted labels with reject and refine option
regular, regular_reject, regular_refine, y_pred_reject_refine, _ = cost.predict(X_train, y_train, X_valid, y_valid, X_test, unique_y_nums, 'rOP')
print(accuracy_score(y_test, regular))






# Example 2: multi-label classification
# load dataset
df = pd.read_csv('emotions-kmeans.csv',delimiter=',')
X = df.iloc[:,:72].values
labels_array = df.iloc[:, -6:].values
# store every sample's labels as a list in y, for example y = [[1,2],[1],[3,4,5]]
y = []
for row in labels_array:
    # 找到值为1的元素的索引，将索引加1得到类别编号（因为索引是从0开始的）
    current_labels = [index for index, value in enumerate(row) if value == 1]
    # 将当前样本的标签列表添加到y中
    y.append(current_labels)

# split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2)

# train model and predict
cost = COST_Classifier_multi()
b1 = 100
b2 = 10

# subspaces generation
X_train, X_valid, X_test = cost.feature_subset_generation(X_train, X_valid, X_test, y_train, b1, b2)

# predict: the training process in included in the predict function
regular, regular_reject, regular_refine, y_pred_reject_refine, _ = cost.predict(X_train, y_train, X_valid, y_valid, X_test, 6, 'rOP')
# convert the predicted labels to the same format with the testing samples, it means convert the predicted single label to a list, for example : [1,[2,3],-1] to [[1],[2,3],[-]]
pred_reject_refine_need = [item if isinstance(item, (list, np.ndarray)) else np.array([item]) for item in
                               y_pred_reject_refine]

score_reject_refine = 0

# compute the average JacAcc score
for pred, test in zip(pred_reject_refine_need, y_test):
    intersection = set(pred) & set(test)
    union = set(pred) | set(test)
    score_reject_refine += (len(intersection) / len(union))
print(score_reject_refine / len(y_test))


