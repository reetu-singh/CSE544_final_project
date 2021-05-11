import pandas as pd
import numpy as np

data = pd.read_csv("data/updated_9.csv")
data = data.iloc[:, 1:]

def get_august_data(input_data):
    ks_confirmed1 = input_data.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ks_confirmed = ks_confirmed1.loc[
                       (ks_confirmed1['Date'] >= '2020-08-01') & (ks_confirmed1['Date'] <= '2020-08-28')].iloc[:, 1]

    ks_deaths1 = input_data.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)
    ks_deaths = ks_deaths1.loc[(ks_deaths1['Date'] >= '2020-08-01') & (ks_deaths1['Date'] <= '2020-08-28')].iloc[:, 1]

    ky_confirmed1 = input_data.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ky_confirmed = ky_confirmed1.loc[
                       (ky_confirmed1['Date'] >= '2020-08-01') & (ky_confirmed1['Date'] <= '2020-08-28')].iloc[:, 1]

    ky_deaths1 = input_data.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)
    ky_deaths = ky_deaths1.loc[(ky_deaths1['Date'] >= '2020-08-01') & (ky_deaths1['Date'] <= '2020-08-28')].iloc[:, 1]

    return ks_confirmed, ks_deaths, ky_confirmed, ky_deaths


def auto_regression(x_train, y_train, x_test, y_test, sse, mape):
    m = np.ones((len(x_train), 1))
    X = np.matrix(x_train)
    X = np.concatenate((m, X), axis=1)
    X_transpose = X.T
    mul_inverse = np.linalg.inv(np.matmul(X_transpose, X))
    Y = np.matrix(y_train)
    Y = Y.T
    temp = np.matmul(mul_inverse, X_transpose)
    beta_val = np.matmul(temp, Y)
    mm = np.ones((1, 1))
    x_test = np.matrix(x_test)
    x_test = np.concatenate((mm, x_test), axis=1)
    predicted_y = np.matmul(x_test, beta_val)
    y_test = np.matrix(y_test).T
    print('predicted y : ' + str(predicted_y) + ", Actual y : " + str(y_test))

    error = y_test - predicted_y
    sse.append(error**2)
    mape.append((abs(error)/y_test)*100)


def EWMA(input_data, test_y, alpha, sse, mape):
    ans = 0
    for i in range(1, len(input_data)+1, 1):
        ans += ((1-alpha)**(i-1)) * (input_data[-i])
    predicted_y = alpha * ans
    print("Predicted Value : "+ str(predicted_y)+" ,Actual y : "+ str(test_y))
    error = test_y - predicted_y
    sse.append(error ** 2)
    mape.append((abs(error) / test_y) * 100)


def calc_ar(input_data, p):
    arr = input_data.values
    training_xdata = []
    training_ydata = []
    sse = []
    mape = []

    for i in range(0, len(arr) - p, 1):
        training_xdata.append(list(arr[i:i + p]))
        training_ydata.append(arr[i + p])

    for i in range(7):
        train_x = training_xdata[0: 21 - p + i + 1]
        train_y = training_ydata[0: 21 - p + i + 1]

        test_x = training_xdata[21 - p + i: 21 - p + i + 1]
        test_y = training_ydata[21 - p + i: 21 - p + i + 1]

        auto_regression(train_x, train_y, test_x, test_y, sse, mape)

    print("\nMSE : "+"{:5.2f}".format(np.mean(sse)))
    print("MAPE : "+"{:5.2f}".format(np.mean(mape))+"%")


def calc_ewma(input_data, alpha):
    arr = input_data.values
    sse = []
    mape = []
    for i in range(7):
        train_y = arr[:21 + i]
        test_y = arr[21 + i]
        EWMA(train_y, test_y, alpha, sse, mape)
    print("\nMSE : " + "{:5.2f}".format(np.mean(sse)))
    print("MAPE : " + "{:5.2f}".format(np.mean(mape)) + "%")

p = 3
print('--------------with p=3 ----------------')
cases_ks, deaths_ks, cases_ky, deaths_ky = get_august_data(data)
print('For KS confirmed cases : \n')
calc_ar(cases_ks, p)
print('--------------------------------------------')

print('For KS death cases : \n')
calc_ar(deaths_ks, p)
print('--------------------------------------------')

print('For KY confirmed cases : \n')
calc_ar(cases_ky, p)
print('--------------------------------------------')

print('For KY death cases : \n')
calc_ar(deaths_ky, p)
print('--------------------------------------------')


p = 5
print('--------------with p=5 ----------------')
print('For KS confirmed cases : \n')
calc_ar(cases_ks, p)
print('--------------------------------------------')

print('For KS death cases : \n')
calc_ar(deaths_ks, p)
print('--------------------------------------------')

print('For KY confirmed cases : \n')
calc_ar(cases_ky, p)
print('--------------------------------------------')

print('For KY death cases : \n')
calc_ar(deaths_ky, p)
print('--------------------------------------------')

print('-----------------------EWMA(0.5)---------------------------')
alpha = 0.5
print('For KS confirmed cases : \n')
calc_ewma(cases_ks, alpha)
print('--------------------------')
print('For KS confirmed cases : \n')
calc_ewma(deaths_ks, alpha)
print('--------------------------')
print('For KS confirmed cases : \n')
calc_ewma(cases_ky, alpha)
print('--------------------------')
print('For KS confirmed cases : \n')
calc_ewma(deaths_ky, alpha)
print('--------------------------')





