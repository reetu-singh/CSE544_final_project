import pandas as pd
import numpy as np

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
    # print(cases_ks)
    return cases_ks, deaths_ks, cases_ky, deaths_ky


def permutation_test(union, sizeA, sizeB):
    np.random.shuffle(union)
    listA = union[:sizeA]
    listB = union[-sizeB:]
    return abs(listA.mean() - listB.mean())


def calculate_pvalue(listA, listB, Tobs, n):
    union_list = np.hstack([listA, listB])
    Ti_values = []
    for i in range(n):
        Ti_values.append(permutation_test(union_list, len(listA), len(listB)))

    count_Ti = []
    for i in range(n):
        if Ti_values[i] > Tobs:
            count_Ti.append(1)
        else:
            count_Ti.append(0)

    calculated_pValue = count_Ti.count(1) / n
    return calculated_pValue


def do_permutation_test(input_data):
    threshold = 0.05
    n = 1000
    ks_confirmed_OD, ks_deaths_OD, ky_confirmed_OD, ky_deaths_OD = get_data_oct_dec(input_data)
    null_hypothesis_cases = "Distribution of ks and ky confirmed cases are same"
    T_observed = abs(ks_confirmed_OD.mean() - ky_confirmed_OD.mean())
    print("For confirmed cases in 2 cities :\n")
    print("Mean of ks confirmed cases : ", ks_confirmed_OD.mean())
    print("Mean of ky confirmed cases : ", ky_confirmed_OD.mean())

    pValue = calculate_pvalue(ks_confirmed_OD, ky_confirmed_OD, T_observed, n)
    print("p-value = " + str(pValue) + " for n = " + str(n))
    print('Given threshold is :', threshold)
    if pValue > threshold:
        print("Null hypothesis (" + null_hypothesis_cases + ") is accepted.")
    else:
        print("Null hypothesis (" + null_hypothesis_cases + ") is rejected.")

    print('---------------------------------------------------------------------------')

    null_hypothesis_deaths = "Distribution of ks and ky death cases are same"
    T_observed = abs(ks_deaths_OD.mean() - ky_deaths_OD.mean())
    print("For death cases in 2 cities :\n")
    print("Mean of ks death cases : ", ks_deaths_OD.mean())
    print("Mean of ky death cases : ", ky_deaths_OD.mean())

    pValue = calculate_pvalue(ks_deaths_OD, ky_deaths_OD, T_observed, n)
    print("p-value = " + str(pValue) + " for n = " + str(n))
    print('Given threshold is :', threshold)
    if pValue > threshold:
        print("Null hypothesis (" + null_hypothesis_deaths + ") is accepted.")
    else:
        print("Null hypothesis (" + null_hypothesis_deaths + ") is rejected.")


do_permutation_test(new_data)
