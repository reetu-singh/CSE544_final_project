import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from decimal import Decimal
import scipy.stats as stats

# reading daily data for the states
new_data = pd.read_csv("updated_9.csv")
new_data['KS confirmed'].mask(new_data['KS confirmed'] < 0, 0, inplace=True)
new_data['KS deaths'].mask(new_data['KS deaths'] < 0, 0, inplace=True)
new_data['KY confirmed'].mask(new_data['KY confirmed'] < 0, 0, inplace=True)
new_data['KY deaths'].mask(new_data['KY deaths'] < 0, 0, inplace=True)
# print(new_data.shape)
# index_names = new_data[new_data['KS confirmed'] < 0].index
# new_data.drop(index_names, inplace=True)
# print(new_data.shape)
# index_names = new_data[new_data['KY confirmed'] < 0].index
# new_data.drop(index_names, inplace=True)
# print(new_data.shape)
# index_names = new_data[new_data['KS deaths'] < 0].index
# new_data.drop(index_names, inplace=True)
# print(new_data.shape)
# index_names = new_data[new_data['KY deaths'] < 0].index
# new_data.drop(index_names, inplace=True)
# print(new_data.shape)


# function for calculating MME for Poisson distribution
def poisson_lambda(input_data):
    X_bar = input_data.mean()
    return X_bar


# function for calculating MME for geometric distribution
def geometric_p(input_data):
    X_bar = input_data.mean()
    p_mme = 1 / X_bar
    return p_mme


# function for calculating MME for binomial distribution
def binomial_n_p(input_data):
    input_data = np.array(input_data)
    n = len(input_data)
    X_bar = input_data.mean()
    summation = 0
    for i in range(n):
        summation += input_data[i] * input_data[i]

    p_mme = X_bar + 1 - (1 / (n * X_bar)) * summation
    n_mme = X_bar / p_mme
    return p_mme, n_mme


# function for filtering the data of months of October, November, December from the read data
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
    return cases_ks, deaths_ks, cases_ky, deaths_ky


# getting separate data for confirmed cases and deaths for the states of KS and KY for the months of OCt-Dec
ks_confirmed_OD, ks_deaths_OD, ky_confirmed_OD, ky_deaths_OD = get_data_oct_dec(new_data)

# sorting the data for use ahead such as calculating cdf
ks_confirmed_OD_sort = ks_confirmed_OD.sort_values(ascending=True)
ky_confirmed_OD_sort = ky_confirmed_OD.sort_values(ascending=True)
ks_deaths_OD_sort = ks_deaths_OD.sort_values(ascending=True)
ky_deaths_OD_sort = ky_deaths_OD.sort_values(ascending=True)


# main function for one sample ks test
def checking_distribution_using_ks_test(distribution_name, ks_confirmed_OD_sort, ks_deaths_OD_sort, ky_confirmed_OD_sort, ky_deaths_OD_sort):
    cdf_ks_confirmed, cdf_ks_deaths, ecdf_ky_confirmed_negative, ecdf_ky_confirmed_positive, \
    ecdf_ky_deaths_negative, ecdf_ky_deaths_positive = [], [], [], [], [], []
    if distribution_name == "Poisson":
        ks_confirmed_mme = poisson_lambda(ks_confirmed_OD_sort)
        ks_deaths_mme = poisson_lambda(ks_deaths_OD_sort)
        for i in ks_confirmed_OD_sort:
            cdf_ks_confirmed.append(stats.poisson.cdf(i, ks_confirmed_mme))
        for i in ks_deaths_OD_sort:
            cdf_ks_deaths.append((stats.poisson.cdf(i, ks_deaths_mme)))
        # print(cdf_ks_confirmed[45])
        # print(ks_confirmed_mme)

    elif distribution_name == "Geometric":
        ks_confirmed_mme = geometric_p(ks_confirmed_OD_sort)
        ks_deaths_mme = geometric_p(ks_deaths_OD_sort)
        for i in ks_confirmed_OD_sort:
            cdf_ks_confirmed.append(stats.geom.cdf(i, ks_confirmed_mme))
        for i in ks_deaths_OD_sort:
            cdf_ks_deaths.append((stats.geom.cdf(i, ks_deaths_mme)))
        # print(cdf_ks_confirmed[45])
        # print(ks_confirmed_mme)

    elif distribution_name == "Binomial":
        ks_confirmed_mme_p, ks_confirmed_mme_n = binomial_n_p(ks_confirmed_OD_sort)
        ks_deaths_mme_p, ks_deaths_mme_n = binomial_n_p(ks_deaths_OD_sort)
        for i in ks_confirmed_OD_sort:
            cdf_ks_confirmed.append(stats.binom.cdf(i, ks_confirmed_mme_n, ks_deaths_mme_p))
        for i in ks_deaths_OD_sort:
            cdf_ks_deaths.append((stats.binom.cdf(i, ks_deaths_mme_n, ks_deaths_mme_p)))
        # print((cdf_ks_confirmed[45]))
        # print(ks_confirmed_mme_p, ks_confirmed_mme_n)

    # calculating ecdf at left and right of the point for confirmed cases and deaths
    ecdf_ky_deaths_negative.append(0)
    ecdf_ky_confirmed_negative.append(0)
    for i in range(1, len(ky_confirmed_OD_sort)):
        ecdf_ky_confirmed_negative.append(i / len(ky_confirmed_OD_sort))
        ecdf_ky_deaths_negative.append(i / len(ky_deaths_OD_sort))
    # print(ecdf_ky_confirmed_negative, ecdf_ky_deaths_negative)

    for i in range(0, len(ky_confirmed_OD_sort)):
        ecdf_ky_confirmed_positive.append((i + 1) / len(ky_confirmed_OD_sort))
        ecdf_ky_deaths_positive.append((i + 1) / len(ky_confirmed_OD_sort))
    # print(ecdf_ky_confirmed_positive)
    # print(ecdf_ky_deaths_positive)

    cdf_diff_confirmed_negative, cdf_diff_confirmed_positive, cdf_diff_deaths_negative, cdf_diff_deaths_positive = [], [], [], []

# calculating the KS-statistic for confirmed cases and deaths
    for i in range(len(ks_confirmed_OD_sort)):
        cdf_diff_confirmed_negative.append(abs(cdf_ks_confirmed[i] - ecdf_ky_confirmed_negative[i]))
        cdf_diff_confirmed_positive.append(abs(cdf_ks_confirmed[i] - ecdf_ky_confirmed_positive[i]))
        cdf_diff_deaths_negative.append(abs(cdf_ks_deaths[i] - ecdf_ky_deaths_negative[i]))
        cdf_diff_deaths_positive.append(abs(cdf_ks_deaths[i] - ecdf_ky_deaths_positive[i]))

    # max value for confirmed cases
    confirmed_max = max(max(cdf_diff_confirmed_negative), max(cdf_diff_confirmed_positive))

    if confirmed_max > 0.05:
        print("KY confirmed cases data for the months of Oct-Dec 2020 doesn't follow " + distribution_name + " distribution.")
    else:
        print("KY confirmed cases data for the months of Oct-Dec 2020 follow " + distribution_name + " distribution.")

    # max value for deaths
    deaths_max = max(max(cdf_diff_deaths_negative), max(cdf_diff_deaths_positive))

    if deaths_max > 0.05:
        print("KY death cases data for the months of Oct-Dec 2020 doesn't follow " + distribution_name + " distribution.")
    else:
        print("KY death cases data for the months of Oct-Dec 2020 follow " + distribution_name + " distribution.")


checking_distribution_using_ks_test("Poisson", ks_confirmed_OD_sort, ks_deaths_OD_sort, ky_confirmed_OD_sort, ky_deaths_OD_sort )
print()
checking_distribution_using_ks_test("Geometric", ks_confirmed_OD_sort, ks_deaths_OD_sort, ky_confirmed_OD_sort, ky_deaths_OD_sort )
print()
checking_distribution_using_ks_test("Binomial", ks_confirmed_OD_sort, ks_deaths_OD_sort, ky_confirmed_OD_sort, ky_deaths_OD_sort )

# ----------------------code below not used----------------

# # ks distribution: Poisson
# cdf_ks_confirmed, cdf_ks_deaths, ecdf_ky_confirmed_negative, ecdf_ky_confirmed_positive,\
# ecdf_ky_deaths_negative, ecdf_ky_deaths_positive = [], [], [], [], [], []
# ks_confirmed_mme_poisson = poisson_lambda(ks_confirmed_OD_sort)
# ks_deaths_mme_poisson = poisson_lambda(ks_deaths_OD_sort)
# for i in ks_confirmed_OD_sort:
#     cdf_ks_confirmed.append(stats.poisson.cdf(i, ks_confirmed_mme_poisson))
# for i in ks_deaths_OD_sort:
#     cdf_ks_deaths.append((stats.poisson.cdf(i, ks_deaths_mme_poisson)))
#
# # ecdf lists
# # print(len(ky_confirmed_OD_sort), len(ky_deaths_OD_sort))
# ecdf_ky_deaths_negative.append(0)
# ecdf_ky_confirmed_negative.append(0)
# for i in range(1, len(ky_confirmed_OD_sort)):
#     ecdf_ky_deaths_negative.append(i/len(ky_confirmed_OD_sort))
#     ecdf_ky_confirmed_negative.append(i/len(ky_deaths_OD_sort))
# # print(ecdf_ky_confirmed_negative, ecdf_ky_deaths_negative)
#
# for i in range(0, len(ky_confirmed_OD_sort)):
#     ecdf_ky_confirmed_positive.append((i+1)/len(ky_confirmed_OD_sort))
#     ecdf_ky_deaths_positive.append((i+1)/len(ky_confirmed_OD_sort))
# # print(ecdf_ky_confirmed_positive)
# # print(ecdf_ky_deaths_positive)
#
# cdf_diff_confirmed_negative, cdf_diff_confirmed_positive, cdf_diff_deaths_negative, cdf_diff_deaths_positive = [], [], [], []
#
# for i in range(len(ks_confirmed_OD_sort)):
#     cdf_diff_confirmed_negative.append(abs(cdf_ks_confirmed[i]-ecdf_ky_confirmed_negative[i]))
#     cdf_diff_confirmed_positive.append(abs(cdf_ks_confirmed[i]-ecdf_ky_confirmed_positive[i]))
#     cdf_diff_deaths_negative.append(abs(cdf_ks_deaths[i]-ecdf_ky_deaths_negative[i]))
#     cdf_diff_deaths_positive.append(abs(cdf_ks_deaths[i]-ecdf_ky_deaths_positive[i]))
#
# print(type(cdf_diff_confirmed_negative), type(cdf_diff_confirmed_positive), type(cdf_diff_deaths_negative), type(cdf_diff_deaths_positive))
# # max value for confirmed cases
# confirmed_max = max(max(cdf_diff_confirmed_negative), max(cdf_diff_confirmed_positive))
#
# if confirmed_max > 0.05:
#     print("KY confirmed cases data for the months of Oct-Dec 2020 doesn't follow Poisson distribution.")
# else:
#     print("KY confirmed cases data for the months of Oct-Dec 2020 follow Poisson distribution.")
#
# # max value for deaths
# deaths_max = max(max(cdf_diff_deaths_negative), max(cdf_diff_deaths_positive))
#
# if deaths_max > 0.05:
#     print("KY death cases data for the months of Oct-Dec 2020 doesn't follow Poisson distribution.")
# else:
#     print("KY death cases data for the months of Oct-Dec 2020 follow Poisson distribution.")

# def poisson_cdf(lmda, x):
#     summation = 0
#     x = int(x)
#     for i in range(x + 1):
#         summation += (lmda ** i) / Decimal(math.factorial(i))
#
#     return (summation) * (math.exp(-1 * lmda))
#     # return math.exp(x - lmda)
#
#
# cdf = poisson_cdf(ks_confirmed_OD_sort, ks_deaths_OD_sort.iloc[45])
# print(cdf)
#
#
# def geometric_cdf(p, x):
#     return 1 - ((1 - p) ** x)
#
#
# def binomial_cdf(n, p, x):
#     summation = 0.0
#     for i in range(x + 1):
#         n_C_i = math.comb(n, i)
#         summation += n_C_i * (p ** i) * ((1 - p) ** (n - i))
#     return summation

# one sample ks test considering first state to be poisson distributed for confirmed cases and deaths


