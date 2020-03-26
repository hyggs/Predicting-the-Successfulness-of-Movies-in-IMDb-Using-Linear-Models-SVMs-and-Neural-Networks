import numpy as nump
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from movie_classification import preprocess
import matplotlib.pyplot as plot
import sklearn.metrics as metrics
import seaborn as sns


# linear regression based on feature, target and size of test data
def linear_regression(d, features, target, ts=300):
    x = d[:, features]
    y = d[:, target]
    x_train = x[:-ts]
    y_train = y[:-ts]
    x_test = x[-ts:]
    y_test = y[-ts:]
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    coef = reg.coef_
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, coef, mse, r2


D = preprocess.pre_process()
print(D)

# Prediction of revenue
# features: popularity, budget, runtime, vote_count, vote_average
Features = [0, 3, 4, 5, 6]
# Features = [0]
# Features = [0, 5]
# target: revenue
Target = 7
Ts = 500
Test, Pred, COEF, MSE, R2 = linear_regression(D, features=Features, target=Target, ts=Ts)

print(Pred)

print("coefficients:" + str(COEF))
print("mean squared error: " + str(MSE))
print("r2 score: " + str(R2))


L1 = nump.zeros(len(D))
L2 = nump.zeros(len(D))
plot.scatter(D[:-Ts, 1], D[:-Ts, 2], edgecolors='black')
plot.xlabel("Budget in Training Data")
plot.ylabel("Revenue in Training Data")
plot.title("Budget in Training Data v.s. Revenue in Training Data")
plot.show()

plot.scatter(D[-Ts:, 1], D[-Ts:, 2], edgecolors='black')
plot.xlabel("Budget in Test Data")
plot.ylabel("Revenue in Test Data")
plot.title("Budget in Test Data v.s. Revenue in Test Data")
plot.show()

plot.scatter(D[-Ts:, 1], Pred, edgecolors='black')
plot.xlabel("Budget in Test Data")
plot.ylabel("Predicted Revenue")
plot.title("Budget in Test Data v.s. Predicted Revenue")
plot.show()
#
# profit = D[:-Ts, 2] - D[:-Ts, 1]
#
# max_profit = nump.max(profit)
# min_profit = nump.min(profit)
# print(max_profit)
# print(min_profit)
#
# pred_profit = Pred - D[-Ts:, 1]
# max_pred_profit = nump.max(pred_profit)
# min_pred_profit = nump.min(pred_profit)
# print(max_pred_profit)
# print(min_pred_profit)
# print(nump.sum(pred_profit) / len(pred_profit))

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
    # #     TrueCluster[i] = 8
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
print(metrics.classification_report(TrueCluster, PredictedCluster, labels=range(5)))
sns.set()
sns.heatmap(C, annot=True)
plot.show()
print(TrueCluster)
print(PredictedCluster)

