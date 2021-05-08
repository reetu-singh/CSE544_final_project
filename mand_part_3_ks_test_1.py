import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from decimal import Decimal

new_data = pd.read_csv("data/updated_9.csv")


def poisson_lambda(input_data):
    X_bar = input_data.mean()
    return X_bar


def geometric_p(input_data):
    X_bar = input_data.mean()
    p_mme = 1 / X_bar
    return p_mme


def binomial_n_p(input_data):
    input_data = np.array(input_data)
    n = len(input_data)
    X_bar = input_data.mean()
    summation = 0
    for i in range(n):
        a = input_data[i] * input_data[i]

    p_mme = X_bar + 1 - (1 / (n * X_bar)) * summation
    n_mme = X_bar / p_mme
    return p_mme, np.round(n_mme)


def get_data_oct_dec(input):
    ks_confirmed = input.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ks_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)
    ky_confirmed = input.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ky_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)

    cases_ks = ks_confirmed.loc[(ks_confirmed['Date'] >= '2020-10-01') & (ks_confirmed['Date'] <= '2020-12-31')].iloc[:,
               2]
    deaths_ks = ks_deaths.loc[(ks_deaths['Date'] >= '2020-10-01') & (ks_deaths['Date'] <= '2020-12-31')].iloc[:, 2]
    cases_ky = ky_confirmed.loc[(ky_confirmed['Date'] >= '2020-10-01') & (ky_confirmed['Date'] <= '2020-12-31')].iloc[:,
               2]
    deaths_ky = ky_deaths.loc[(ky_deaths['Date'] >= '2020-10-01') & (ky_deaths['Date'] <= '2020-12-31')].iloc[:, 2]
    print(poisson_lambda(cases_ks))
    return cases_ks, deaths_ks, cases_ky, deaths_ky


get_data_oct_dec(new_data)


#
# a = np.array(ks_confirmed_OD)
# a[0]
#
# binomial_n_p(ks_confirmed_OD)
#
# ks_confirmed_OD.mean()


def poisson_cdf(lmda, x):
    summation = 0
    for i in range(x + 1):
        summation += (lmda ** i) / Decimal(math.factorial(i))

    return (summation) * (math.exp(-1 * lmda))
    # return math.exp(x - lmda)


def geometric_cdf(p, x):
    return 1 - ((1 - p) ** x)


def binomial_cdf(n, p, x):
    summation = 0.0
    for i in range(x + 1):
        n_C_i = math.comb(n, i)
        summation += n_C_i * (p ** i) * ((1 - p) ** (n - i))
    return summation

# lambdaa = poisson_lambda(ks_confirmed_OD)
# print(lambdaa)
#
# sorted_ks_confirmed = np.sort(ks_confirmed_OD)
# delta = 0.1
# X = [sorted_ks_confirmed[0] - delta]
# Y = [0]
#
# for i in range(len(ks_confirmed_OD)):
#     ecdf_poisson = poisson_cdf(lambdaa, sorted_ks_confirmed[i])
#     X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
#     Y = Y + [ecdf_poisson, ecdf_poisson]
#
# X = X + [np.max(ks_confirmed_OD) + delta]
# Y = Y + [1]
#
# sorted_ky_confirmed = np.sort(ky_confirmed_OD)
# delta2 = 0.1
#
# X2 = [sorted_ky_confirmed[0] - delta2]
# Y2 = [0]
#
# for i in range(len(ky_confirmed_OD)):
#     X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
#     Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]
#
# X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
# Y2 = Y2 + [1]
#
# max_ks = int(np.max(ks_confirmed_OD))
# max_ky = int(np.max(ky_confirmed_OD))
# max_cases = max(max_ks, max_ky)
#
# maximum_difference = 0
# for i in range(max_cases + 1):
#     d = abs(np.interp(i, X, Y) - np.interp(i, X2, Y2))
#     if maximum_difference < d:
#         maximum_difference = d
#         x_max_diff = i
#
# print("maximum difference is : " + str(maximum_difference))
# if maximum_difference > threshold:
#     print("Since, max difference is greater than 0.05, therefore, Null hypothesis is rejected")
#
# else:
#     print("Since, max difference is less than 0.05, therefore, Null hypothesis is accepted")
#
# plt.annotate('Max diff x=' + str(x_max_diff),
#              xy=(x_max_diff, 0.2),
#              xytext=(x_max_diff, 0.5),
#              arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))
#
# plt.plot(X, Y, label="eCDF of Males ages")
# plt.plot(X2, Y2, label="eCDF of Females ages")
# plt.xlabel('x')
# plt.ylabel('Pr[X<=x]')
# plt.title("eCDF of male and female ages")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# threshold = 0.05
#
# pmme = geometric_p(ks_confirmed_OD)
# print(pmme)
#
# sorted_ks_confirmed = np.sort(ks_confirmed_OD)
# delta = 0.1
# X = [sorted_ks_confirmed[0] - delta]
# Y = [0]
#
# for i in range(len(ks_confirmed_OD)):
#     ecdf_geometric = gemetric_cdf(pmme, sorted_ks_confirmed[i])
#     X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
#     Y = Y + [ecdf_geometric, ecdf_geometric]
#
# X = X + [np.max(ks_confirmed_OD) + delta]
# Y = Y + [1]
#
# sorted_ky_confirmed = np.sort(ky_confirmed_OD)
# delta2 = 0.1
#
# X2 = [sorted_ky_confirmed[0] - delta2]
# Y2 = [0]
#
# for i in range(len(ky_confirmed_OD)):
#     X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
#     Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]
#
# X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
# Y2 = Y2 + [1]
#
# max_ks = int(np.max(ks_confirmed_OD))
# max_ky = int(np.max(ky_confirmed_OD))
# max_cases = max(max_ks, max_ky)
#
# maximum_difference = 0
# for i in range(max_cases + 1):
#     d = abs(np.interp(i, X, Y) - np.interp(i, X2, Y2))
#     if maximum_difference < d:
#         maximum_difference = d
#         x_max_diff = i
#
# print("maximum difference is : " + str(maximum_difference))
# if maximum_difference > threshold:
#     print("Since, max difference is greater than 0.05, therefore, Null hypothesis is rejected")
#
# else:
#     print("Since, max difference is less than 0.05, therefore, Null hypothesis is accepted")
#
# plt.annotate('Max diff x=' + str(x_max_diff),
#              xy=(x_max_diff, 0.2),
#              xytext=(x_max_diff, 0.5),
#              arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))
#
# plt.plot(X, Y, label="eCDF of Males ages")
# plt.plot(X2, Y2, label="eCDF of Females ages")
# plt.xlabel('x')
# plt.ylabel('Pr[X<=x]')
# plt.title("eCDF of male and female ages")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
#
# threshold = 0.05
#
# pmme = geometric_p(ks_deaths_OD)
# print(pmme)
#
# sorted_ks_deaths = np.sort(ks_deaths_OD)
# delta = 0.1
# X = [sorted_ks_deaths[0] - delta]
# Y = [0]
#
# for i in range(len(ks_deaths_OD)):
#     ecdf_geometric = gemetric_cdf(pmme, sorted_ks_deaths[i])
#     X = X + [sorted_ks_deaths[i], sorted_ks_deaths[i]]
#     Y = Y + [ecdf_geometric, ecdf_geometric]
#
# X = X + [np.max(ks_deaths_OD) + delta]
# Y = Y + [1]
#
# sorted_ky_deaths = np.sort(ky_deaths_OD)
# delta2 = 0.1
#
# X2 = [sorted_ky_deaths[0] - delta2]
# Y2 = [0]
#
# for i in range(len(ky_deaths_OD)):
#     X2 = X2 + [sorted_ky_deaths[i], sorted_ky_deaths[i]]
#     Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_deaths_OD)]
#
# X2 = X2 + [np.max(ky_deaths_OD) + delta2]
# Y2 = Y2 + [1]
#
# max_ks = int(np.max(ks_deaths_OD))
# max_ky = int(np.max(ky_deaths_OD))
# max_deaths = max(max_ks, max_ky)
#
# maximum_difference = 0
# for i in range(max_deaths + 1):
#     d = abs(np.interp(i, X, Y) - np.interp(i, X2, Y2))
#     if maximum_difference < d:
#         maximum_difference = d
#         x_max_diff = i
#
# print("maximum difference is : " + str(maximum_difference))
# if maximum_difference > threshold:
#     print("Since, max difference is greater than 0.05, therefore, Null hypothesis is rejected")
#
# else:
#     print("Since, max difference is less than 0.05, therefore, Null hypothesis is accepted")
#
# plt.annotate('Max diff x=' + str(x_max_diff),
#              xy=(x_max_diff, 0.2),
#              xytext=(x_max_diff, 0.5),
#              arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))
#
# plt.plot(X, Y, label="eCDF of Males ages")
# plt.plot(X2, Y2, label="eCDF of Females ages")
# plt.xlabel('x')
# plt.ylabel('Pr[X<=x]')
# plt.title("eCDF of male and female ages")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
#
# threshold = 0.05
#
# pmme, nmme = binomial_n_p(ks_confirmed_OD)
# nmme = int(nmme)
# # pmme=listt[0]
# # nmme=listt[1]
# print('----')
# print(pmme)
# print(nmme)
#
# sorted_ks_confirmed = np.sort(ks_confirmed_OD)
# delta = 0.1
# X = [sorted_ks_confirmed[0] - delta]
# Y = [0]
#
# for i in range(len(ks_confirmed_OD)):
#     ecdf_binomial = binomial_cdf(nmme, pmme, sorted_ks_confirmed[i])
#     X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
#     Y = Y + [ecdf_binomial, ecdf_binomial]
#
# X = X + [np.max(ks_confirmed_OD) + delta]
# Y = Y + [1]
#
# sorted_ky_confirmed = np.sort(ky_confirmed_OD)
# delta2 = 0.1
#
# X2 = [sorted_ky_confirmed[0] - delta2]
# Y2 = [0]
#
# for i in range(len(ky_confirmed_OD)):
#     X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
#     Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]
#
# X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
# Y2 = Y2 + [1]
#
# max_ks = int(np.max(ks_confirmed_OD))
# max_ky = int(np.max(ky_confirmed_OD))
# max_cases = max(max_ks, max_ky)
#
# maximum_difference = 0
# for i in range(max_cases + 1):
#     d = abs(np.interp(i, X, Y) - np.interp(i, X2, Y2))
#     if maximum_difference < d:
#         maximum_difference = d
#         x_max_diff = i
#
# print("maximum difference is : " + str(maximum_difference))
# if maximum_difference > threshold:
#     print("Since, max difference is greater than 0.05, therefore, Null hypothesis is rejected")
#
# else:
#     print("Since, max difference is less than 0.05, therefore, Null hypothesis is accepted")
#
# plt.annotate('Max diff x=' + str(x_max_diff),
#              xy=(x_max_diff, 0.2),
#              xytext=(x_max_diff, 0.5),
#              arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))
#
# plt.plot(X, Y, label="eCDF of Males ages")
# plt.plot(X2, Y2, label="eCDF of Females ages")
# plt.xlabel('x')
# plt.ylabel('Pr[X<=x]')
# plt.title("eCDF of male and female ages")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
