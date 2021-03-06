import numpy as nump
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression
from movie_classification import preprocess
import sklearn.metrics as metrics
import seaborn as sns


def logistic_regression(d, features, target, ts=300):
    x = d[:, features]
    y = d[:, target]
    x_train = x[:-ts]
    y = y[:-ts]
    y_train = nump.zeros(len(y))
    for i in range(len(y)):
        y_train[i] = y[i]
    y_train = y_train.astype('int')
    x_test = x[-ts:]
    y_test = y[-ts:]
    reg = LogisticRegression(multi_class='ovr', max_iter=100000).fit(x_train, y_train)
    pred = reg.predict(x_test)
    return pred


# Prediction of revenue
# features: popularity, budget, runtime, vote_count, vote_average
Features = [0, 3, 4, 5, 6]
# Features = [0]
# target: revenue
Target = 7
D = preprocess.pre_process_classed()
Ts = 500
Pred = logistic_regression(D, Features, Target, Ts)


# B9 = 3000000000
# B8 = 2000000000
# B7 = 700000000
# B6 = 600000000
# B5 = 500000000
B4 = 1000000000
B3 = 600000000
B2 = 300000000
B1 = 100000000
B0 = 50000000
TrueCluster = nump.zeros(Ts)
PredictedCluster = nump.zeros(Ts)
for i in range(Ts):
    true_reve = D[-Ts + i, 2]
    pred_reve = Pred[i]
    if true_reve <= B0:
        TrueCluster[i] = 0
    elif true_reve <= B1:
        TrueCluster[i] = 1
    elif true_reve <= B2:
        TrueCluster[i] = 2
    elif true_reve <= B3:
        TrueCluster[i] = 3
    elif true_reve <= B4:
        TrueCluster[i] = 4
    # elif true_reve <= B5:
    #     TrueCluster[i] = 5
    # elif true_reve <= B6:
    #     TrueCluster[i] = 6
    # elif true_reve <= B7:
    #     TrueCluster[i] = 7
    # elif true_reve <= B8:
    #     TrueCluster[i] = 8
    # else:
    #     TrueCluster[i] = 5

    if pred_reve <= B0:
        PredictedCluster[i] = 0
    elif pred_reve <= B1:
        PredictedCluster[i] = 1
    elif pred_reve <= B2:
        PredictedCluster[i] = 2
    elif pred_reve <= B3:
        PredictedCluster[i] = 3
    elif pred_reve <= B4:
        PredictedCluster[i] = 4
    # elif pred_reve <= B5:
    #     PredictedCluster[i] = 5
    # elif pred_reve <= B6:
    #     PredictedCluster[i] = 6
    # elif pred_reve <= B7:
    #     PredictedCluster[i] = 7
    # elif pred_reve <= B8:
    #     PredictedCluster[i] = 8
    # else:
    #     PredictedCluster[i] = 5

C = metrics.confusion_matrix(TrueCluster, PredictedCluster, labels=range(5))
sns.set()
sns.heatmap(C, annot=True)
plot.show()
print(metrics.classification_report(TrueCluster, PredictedCluster, labels=range(5)))


profit = D[:-Ts, 2] - D[:-Ts, 1]

max_profit = nump.max(profit)
min_profit = nump.min(profit)

pred_profit = Pred - D[-Ts:, 1]

# S = 0
# W1 = 50000000
# W2 = 150000000
# W3 = 250000000
# B0 = 0
# B1 = 0
# B2 = 0
# B3 = 0
# B4 = 0
# for i in range(Ts):
#     if Pred[i] < 0 and D[-Ts + i, 2] < 0:
#         B0 += 1
#         S += 1
#     elif 0 <= Pred[i] < W1 and 0 <= D[-Ts + i, 2] < W1:
#         B1 += 1
#         S += 1
#     elif W1 <= Pred[i] < W2 and W1 <= D[-Ts + i, 2] < W2:
#         B2 += 1
#         S += 1
#     elif W2 <= Pred[i] < W3 and W2 <= D[-Ts + i, 2] < W3:
#         B3 += 1
#         S += 1
#     elif Pred[i] >= W3 and D[-Ts + i, 2] >= W3:
#         B4 += 1
#         S += 1
#
# print("Success: " + str(S))
# print("Failure: " + str(Ts - S))
# print("Success Rate: " + str(S / Ts))
# print("Number of B0: " + str(B0))
# print("Number of B1: " + str(B1))
# print("Number of B2: " + str(B2))
# print("Number of B3: " + str(B3))
# print("Number of B4: " + str(B4))
