import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import pandas as pd

ann = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Scenarios/China ML/ANN/ann_prob.csv')
dt = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Scenarios/China ML/Decision Tree Classification/dt_prob.csv')
knn = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Scenarios/China ML/K-Nearest Neighbors (K-NN)/knn_prob.csv')
lr = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Scenarios/China ML/Logistic Regression/lr_prob.csv')
rf = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Scenarios/China ML/Random Forest Classification/rf_prob.csv')

ann_t = ann.iloc[:, [0]]
ann_p = ann.iloc[:, [1]]

dt_t = dt.iloc[:, [0]]
dt_p = dt.iloc[:, [1]]

knn_t = knn.iloc[:, [0]]
knn_p = knn.iloc[:, [1]]

lr_t = lr.iloc[:, [0]]
lr_p = lr.iloc[:, [1]]

rf_t = rf.iloc[:, [0]]
rf_p = rf.iloc[:, [1]]

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
fpr1, tpr1, threshold1 = metrics.roc_curve(ann_t, ann_p)
roc_auc1 = metrics.auc(fpr1, tpr1)
fpr2, tpr2, threshold2 = metrics.roc_curve(dt_t, dt_p)
roc_auc2 = metrics.auc(fpr2, tpr2)
fpr3, tpr3, threshold1 = metrics.roc_curve(knn_t, knn_p)
roc_auc3 = metrics.auc(fpr3, tpr3)
fpr4, tpr4, threshold4 = metrics.roc_curve(lr_t, lr_p)
roc_auc4 = metrics.auc(fpr4, tpr4)
fpr5, tpr5, threshold5 = metrics.roc_curve(rf_t, rf_p)
roc_auc5 = metrics.auc(fpr5, tpr5)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'r', label = 'ANN(AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, 'g', label = 'DT(AUC = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, 'y', label = 'KNN(AUC = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, 'b', label = 'LR(AUC = %0.2f)' % roc_auc4)
plt.plot(fpr5, tpr5, 'm', label = 'RF(AUC = %0.2f)' % roc_auc5)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'c--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()