from sklearn.neural_network import MLPClassifier
from movie_classification import preprocess
import numpy as nump
import matplotlib.pyplot as plot
import sklearn.metrics as metrics
import seaborn as sns


def nn(d, features, target, ts):
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
    cla = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14, 9), random_state=1)
    cla.fit(x_train, y_train)
    pred = cla.predict(x_test)
    return pred


# Prediction of revenue
# features: popularity, budget, runtime, vote_count, vote_average
Features = [0, 3, 4, 5, 6]
# Features = [0]
# target: revenue
Target = 7
D = preprocess.pre_process_classed()
Ts = 500
Pred = nn(D, Features, Target, Ts)
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

# print(Pred)
# S = 0
# C = [0, 0, 0, 0]
# S1 = 0
# S2 = 0
# S3 = 0
# S4 = 0
# for i in range(Ts):
#     C[D[-Ts + i, 2]] += 1
#     if Pred[i] == 0 and D[-Ts + i, 2] == 0:
#         S1 += 1
#         S += 1
#     elif Pred[i] == 1 and D[-Ts + i, 2] == 1:
#         S2 += 1
#         S += 1
#     elif Pred[i] == 2 and D[-Ts + i, 2] == 2:
#         S3 += 1
#         S += 1
#     elif Pred[i] == 3 and D[-Ts + i, 2] == 3:
#         S4 += 1
#         S += 1
# T = [S1, S2, S3, S4]
# print("Total Success: " + str(S))
# print("Total Success Rate: " + str(S / Ts))
# for i in range(0, 4):
#     print("B" + str(i) + "Success: " + str(T[i]))
#     print("B" + str(i) + "Success Rate: " + str(T[i] / C[i]))
