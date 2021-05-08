import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

new_data = pd.read_csv("data/updated_9.csv")


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


def ks_2_sample_test(ks_data, ky_data, type):
    threshold = 0.05

    sorted_ks = np.sort(ks_data)
    delta = 0.1
    X = [sorted_ks[0] - delta]
    Y = [0]

    for i in range(len(ky_data)):
        X = X + [sorted_ks[i], sorted_ks[i]]
        Y = Y + [Y[-1], Y[-1] + 1 / len(ks_data)]

    X = X + [np.max(ky_data) + delta]
    Y = Y + [1]

    sorted_ky = np.sort(ky_data)
    delta2 = 0.1

    X2 = [sorted_ky[0] - delta2]
    Y2 = [0]

    for i in range(len(ky_data)):
        X2 = X2 + [sorted_ky[i], sorted_ky[i]]
        Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_data)]

    X2 = X2 + [np.max(ky_data) + delta2]
    Y2 = Y2 + [1]

    max_ks = int(np.max(ks_data))
    max_ky = int(np.max(ky_data))

    max_cases = max(max_ks, max_ky)

    min_ks = int(np.min(ks_data))
    min_ky = int(np.min(ky_data))

    min_cases = min(min_ks, min_ky)

    maximum_difference = 0

    for i in range(min_cases, max_cases + 1, 1):
        d = abs(np.interp(i, X, Y) - np.interp(i, X2, Y2))
        if maximum_difference < d:
            maximum_difference = d
            x_max_diff = i
    print("maximum difference is : " + str(maximum_difference))
    if maximum_difference > threshold:
        print("Since, max difference is greater than 0.05, therefore, Null hypothesis is rejected")

    else:
        print("Since, max difference is less than 0.05, therefore, Null hypothesis is accepted")

    plt.annotate('Max diff x=' + str(x_max_diff),
                 xy=(x_max_diff, 0.1),
                 xytext=(x_max_diff, 0.2),
                 arrowprops=dict(facecolor='black', linewidth=0.01, shrink=0.00001))

    plt.plot(X, Y, label="eCDF of ks " + type + " cases")
    plt.plot(X2, Y2, label="eCDF of ky " + type + " cases")
    plt.xlabel('x')
    plt.ylabel('Pr[X<=x]')
    plt.title("eCDF of " + type + " cases")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def part_2_c(input_data):
    ks_confirmed, ks_deaths, ky_confirmed, ky_deaths = get_data_oct_dec(input_data)
    print("\nFor confirmed cases in KS and KY: \n")
    ks_2_sample_test(ks_confirmed, ky_confirmed, 'confirmed')
    print("\nFor death cases in KS and KY: \n")
    ks_2_sample_test(ks_deaths, ky_deaths, 'death')


part_2_c(new_data)
