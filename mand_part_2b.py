import pandas as pd
import numpy as np
import csv

# reading data for Feb '21 and March '21 from the updated_9 file

data = pd.read_csv("data/updated_9.csv")
data = data.iloc[:, 1:]
# print(data)

def getEntireDataColumn(data):
    first_state_confirmed = []
    second_state_confirmed = []
    first_state_deaths = []
    second_state_deaths = []

    rows = data[data.columns[0]].count()
    for i in range(rows):
        first_state_confirmed.append(data.iloc[i][1])
        second_state_confirmed.append(data.iloc[i][2])
        first_state_deaths.append(data.iloc[i][3])
        second_state_deaths.append(data.iloc[i][4])

    return first_state_confirmed, second_state_confirmed, first_state_deaths, second_state_deaths


def data_for_particular_month(start_date, end_date):
    start = data.Date[data.Date == start_date].index.tolist()
    end = data.Date[data.Date == end_date].index.tolist()
    clipped_data = data.iloc[start[0] : end[0]+1, : ]

    first_state_confirmed_table = clipped_data.drop(['Date', 'KY confirmed', 'KS deaths', 'KY deaths'], axis=1)
    second_state_confirmed_table = clipped_data.drop(['Date', 'KS confirmed', 'KS deaths', 'KY deaths'], axis=1)
    first_state_deaths_table = clipped_data.drop(['Date', 'KY confirmed', 'KS confirmed', 'KY deaths'], axis=1)
    second_state_deaths_table = clipped_data.drop(['Date', 'KY confirmed', 'KS confirmed', 'KS deaths'], axis=1)

    first_state_confirmed = []
    second_state_confirmed = []
    first_state_deaths = []
    second_state_deaths = []
    rows = first_state_confirmed_table[first_state_confirmed_table.columns[0]].count()
    for i in range(rows):
        first_state_confirmed.append(first_state_confirmed_table.iloc[i][0])
        second_state_confirmed.append(second_state_confirmed_table.iloc[i][0])
        first_state_deaths.append(first_state_deaths_table.iloc[i][0])
        second_state_deaths.append(second_state_deaths_table.iloc[i][0])

    # i = 0
    # while data.iloc[i][0] != start_date:
    #     i += 1
    # first_state_confirmed = []
    # second_state_confirmed = []
    # first_state_deaths = []
    # second_state_deaths = []
    # while data.iloc[i][0] != end_date:
    #     first_state_confirmed.append(data.iloc[i][1])
    #     second_state_confirmed.append(data.iloc[i][2])
    #     first_state_deaths.append(data.iloc[i][3])
    #     second_state_deaths.append(data.iloc[i][4])
    #     i += 1
    # i += 1
    # first_state_confirmed.append(data.iloc[i][1])
    # second_state_confirmed.append(data.iloc[i][2])
    # first_state_deaths.append(data.iloc[i][3])
    # second_state_deaths.append(data.iloc[i][4])
    return first_state_confirmed, second_state_confirmed, first_state_deaths, second_state_deaths


ks_feb_confirmed, ky_feb_confirmed, ks_feb_deaths, ky_feb_deaths = \
    data_for_particular_month("2021-02-01", "2021-02-28")
ks_march_confirmed, ky_march_confirmed, ks_march_deaths, ky_march_deaths = \
    data_for_particular_month("2021-03-01", "2021-03-31")
#
# print(len(ks_feb_confirmed))
# print(len(ky_feb_confirmed))
# print(len(ky_feb_deaths))
# print(len(ky_feb_deaths))
# print(len(ks_march_confirmed))
# print(len(ky_march_confirmed))
# print(len(ky_march_deaths))
# print(len(ky_march_deaths))

# print(ks_feb_confirmed)
# print(ky_feb_confirmed)
# print(ks_feb_deaths)
# print(ky_feb_deaths)


def mean_of_data_points(data_points):
    return sum(data_points)/len(data_points)


ks_feb_mean_confirmed_cases = mean_of_data_points(ks_feb_confirmed)
ks_feb_mean_deaths = mean_of_data_points(ks_feb_deaths)
ks_march_mean_confirmed_cases = mean_of_data_points(ks_march_confirmed)
ks_march_mean_deaths = mean_of_data_points(ks_march_deaths)


def standard_error(no_of_data_points, variance):
    return np.sqrt(variance/no_of_data_points)

def standardDeviation(data, data_mean):
    v = 0
    for d in data:
        v += (d - data_mean)**2

    std = (v / len(data) - 1)**0.5
    return std

def walds_one_sample_test(theta_hat, theta_guess, se_hat_theta_hat):
    w = (theta_hat - theta_guess)/se_hat_theta_hat
    # 1.96 is the z_(alpha/2) value where alpha = 0.05
    if abs(w) > 1.96:
        return 0
    return 1

def ZTestOneSamleTest(sample_mean, mean_guess, true_std, total_values):
    z = (sample_mean - mean_guess)/(true_std/total_values**0.5)

    return abs(z)

# for Wald's test we have to consider that daily data is Poisson distributed
# using MLE as estimator for Wald's test
# MLE for Poisson distribution is sample mean
# Variance of Poisson distribution is equal to mean

# Wald's one sample test for first state(KS)'s confirmed cases and deaths
# using mean daily values of confirmed cases and deaths of Feb '21
# as the guess for mean daily values for March '21 i.e checking if mean
# confirmed deaths and cases in March '21 are same or different from Feb '21

# Default hypothesis is mean confirmed cases of March equals mean confirmed
# cases of Feb and mean deaths of March equals means deaths of Feb

# wald's one sample test for daily mean confirmed cases in March'21 for first state(KS)

print("Wald's one sample test on total confirmed cases of state 1(KS): ")

se_hat_mean_confirmed_cases_ks = standard_error(len(ks_march_confirmed), ks_march_mean_confirmed_cases)
if walds_one_sample_test(ks_march_mean_confirmed_cases,
                         ks_feb_mean_confirmed_cases,se_hat_mean_confirmed_cases_ks) == 0:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

# wald's one sample test for daily mean deaths in March '21 for first state(KS)

print("###################################################")
print("Wald's one sample test on total deaths of state 1(KS): ")
se_hat_mean_deaths_ks = standard_error(len(ks_march_deaths), ks_march_mean_deaths)
if walds_one_sample_test(ks_march_mean_deaths, ks_feb_mean_deaths, se_hat_mean_deaths_ks) == 0:
    print("Null hypothesis mu1 = mu2 i.e. mean deaths "
          "in state 1 for the month of March are equal to mean deaths"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean deaths "
          "in state 1 for the month of March are equal to mean deaths"
          " for the month of Feb is accepted.")

# Wald's one sample test for first second(KY)'s confirmed cases and deaths
# using mean daily values of confirmed cases and deaths of Feb '21
# as the guess for mean daily values for March '21 i.e checking if mean
# confirmed deaths and cases in March '21 are same or different from Feb '21

# Default hypothesis is mean confirmed cases of March equals mean confirmed
# cases of Feb and mean deaths of March equals means deaths of Feb

ky_feb_mean_confirmed_cases = mean_of_data_points(ky_feb_confirmed)
ky_feb_mean_deaths = mean_of_data_points(ky_feb_deaths)
ky_march_mean_confirmed_cases = mean_of_data_points(ky_march_confirmed)
ky_march_mean_deaths = mean_of_data_points(ky_march_deaths)

# wald's one sample test for daily mean confirmed cases in March'21 for second state(KY)

print("###################################################")
print("Wald's one sample test on total confirmed cases of state 2(KY): ")
se_hat_mean_confirmed_cases_ky = standard_error(len(ky_march_confirmed), ky_march_mean_confirmed_cases)
if walds_one_sample_test(ky_march_mean_confirmed_cases,
                         ky_feb_mean_confirmed_cases, se_hat_mean_confirmed_cases_ky) == 0:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 2 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 2 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

# wald's one sample test for daily mean deaths in March '21 for second state(KY)
print("###################################################")
print("Wald's one sample test on total deaths of state 2(KY): ")

se_hat_mean_deaths_ky = standard_error(len(ky_march_deaths), ky_march_mean_deaths)
if walds_one_sample_test(ky_march_mean_deaths, ky_feb_mean_deaths, se_hat_mean_deaths_ky) == 0:
    print("Null hypothesis mu1 = mu2 i.e. mean deaths "
          "in state 2 for the month of March are equal to mean deaths"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean deaths "
          "in state 1 for the month of March are equal to mean deaths"
          " for the month of Feb is accepted.")










# Fetching the entre dataset into separate lists of state1 confirmed cases, state 2 confirmed cases, sate 1 total
# deaths, and state 2 total deaths.
ks_confirmed, ky_confirmed, ks_deaths, ky_deaths = getEntireDataColumn(data)

# Computing true mean
true_mean_ks_confirmed, true_mean_ks_confirmed, true_mean_ks_deaths, true_mean_ky_deaths = \
    mean_of_data_points(ks_confirmed), mean_of_data_points(ky_confirmed), mean_of_data_points(ks_deaths), mean_of_data_points(ky_deaths)


# computing ture standard deviation
std_ks_confirmed = standardDeviation(ks_confirmed, true_mean_ks_confirmed)
std_ky_confirmed = standardDeviation(ky_confirmed, true_mean_ks_confirmed)
std_ks_deaths = standardDeviation(ks_deaths, true_mean_ks_deaths)
std_ky_deaths = standardDeviation(ky_deaths, true_mean_ky_deaths)

# Performing one-sample Z-Tests
Ztest_ks_confirmed = ZTestOneSamleTest(ks_march_mean_confirmed_cases, ks_feb_mean_confirmed_cases, std_ks_confirmed, len(ks_march_confirmed))
Ztest_ky_confirmed = ZTestOneSamleTest(ky_march_mean_confirmed_cases, ky_feb_mean_confirmed_cases, std_ky_confirmed, len(ky_march_confirmed))
Ztest_ks_deaths = ZTestOneSamleTest(ks_march_mean_deaths, ks_feb_mean_deaths, std_ks_deaths, len(ks_march_deaths))
Ztest_ky_deaths = ZTestOneSamleTest(ky_march_mean_deaths, ky_feb_mean_deaths, std_ky_deaths, len(ky_march_deaths))

# print(Ztest_ks_confirmed, Ztest_ky_confirmed, Ztest_ks_deaths, Ztest_ky_deaths)

print("###################################################")
print("Z-Test on total confirmed cases of state 1(KS): ")
if Ztest_ks_confirmed > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Z-Test on total deaths of state 1(KS): ")
if Ztest_ks_deaths > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Z-Test on total confirmed cases of state 2(KY): ")
if Ztest_ks_deaths > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Z-Test on total deaths of state 2(KY): ")
if Ztest_ks_deaths > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

# Computing standard deviation for T-test   ks_march_confirmed, ky_march_confirmed, ks_march_deaths, ky_march_deaths
# ky_march_mean_confirmed_cases , ky_march_mean_deaths

sd_T_ks_confirmed =  standardDeviation(ks_march_confirmed, ks_march_mean_confirmed_cases)
sd_T_ky_confirmed =  standardDeviation(ky_march_confirmed, ky_march_mean_confirmed_cases)
sd_T_ks_deaths =  standardDeviation(ks_march_deaths, ks_march_mean_deaths)
sd_T_ky_deaths =  standardDeviation(ky_march_deaths, ky_march_mean_deaths)

# Performing 1 sampled T-test
def TTestOneSamleTest(sample_mean, mean_guess, std, total_values):
    t = (sample_mean - mean_guess)/(std/total_values**0.5)

    return abs(t)

Ttest_ks_confirmed = TTestOneSamleTest(ks_march_mean_confirmed_cases, ks_feb_mean_confirmed_cases, sd_T_ks_confirmed, len(ks_march_confirmed))
Ttest_ky_confirmed = TTestOneSamleTest(ky_march_mean_confirmed_cases, ky_feb_mean_confirmed_cases, sd_T_ky_confirmed, len(ky_march_confirmed))
Ttest_ks_deaths = TTestOneSamleTest(ks_march_mean_deaths, ks_feb_mean_deaths, sd_T_ks_deaths, len(ks_march_deaths))
Ttest_ky_deaths = TTestOneSamleTest(ky_march_mean_deaths, ky_feb_mean_deaths, sd_T_ky_deaths, len(ky_march_deaths))

print("T one sample: ", Ttest_ks_confirmed, Ttest_ky_confirmed, Ttest_ks_deaths, Ttest_ky_deaths)

print("###################################################")
print("One sample T-Test on total confirmed cases of state 1(KS): ")
if Ttest_ks_confirmed > 1.697261:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("One sample T-Test on total confirmed cases of state 2(KY): ")
if Ttest_ky_confirmed > 1.697261:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("One sample T-Test on total deaths of state 1(KS): ")
if Ttest_ks_deaths > 1.697261:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("One sample T-Test on total deaths of state 2(KY): ")
if Ttest_ky_deaths > 1.697261:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

# Two sampled waled test

def standardErrorTwoSample(no_of_data_points_feb, no_of_data_points_march, variance_feb, variance_march):
    return np.sqrt((variance_feb/no_of_data_points_feb) + (variance_march/no_of_data_points_march))

def waldsTwoSampleTest(state_1_mean, state_2_mean, se_hat):
    w = (state_1_mean - state_2_mean) / se_hat
    # 1.96 is the z_(alpha/2) value where alpha = 0.05
    return w

se_hat_mean_confirmed_cases_ks_2Sample = standardErrorTwoSample(len(ks_feb_confirmed), len(ks_march_confirmed), ks_feb_mean_confirmed_cases, ks_march_mean_confirmed_cases)
se_hat_mean_confirmed_cases_ky_2Sample = standardErrorTwoSample(len(ky_feb_confirmed), len(ky_march_confirmed), ky_feb_mean_confirmed_cases, ky_march_mean_confirmed_cases)
se_hat_mean_deaths_ks_2Sample = standardErrorTwoSample(len(ks_feb_deaths), len(ks_march_deaths), ks_feb_mean_deaths, ks_march_mean_deaths)
se_hat_mean_deaths_ky_2Sample = standardErrorTwoSample(len(ky_feb_deaths), len(ky_march_deaths), ky_feb_mean_deaths, ky_march_mean_deaths)


Wald_ks_confirmed_2Sample = waldsTwoSampleTest(ks_feb_mean_confirmed_cases, ks_march_mean_confirmed_cases, se_hat_mean_confirmed_cases_ks_2Sample)
Wald_ky_confirmed_2Sample = waldsTwoSampleTest(ky_feb_mean_confirmed_cases, ky_march_mean_confirmed_cases, se_hat_mean_confirmed_cases_ky_2Sample)
Wald_ks_deaths_2Sample = waldsTwoSampleTest(ks_feb_mean_deaths, ks_march_mean_deaths, se_hat_mean_deaths_ks_2Sample)
Wald_ky_deaths_2Sample = waldsTwoSampleTest(ky_feb_mean_deaths, ky_march_mean_deaths, se_hat_mean_deaths_ky_2Sample)

print(Wald_ks_confirmed_2Sample, Wald_ky_confirmed_2Sample, Wald_ks_deaths_2Sample, Wald_ky_deaths_2Sample)


print("###################################################")
print("Two sample Wald's Test on total confirmed cases of state 1(KS): ")
if Wald_ks_confirmed_2Sample > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample Wald's Test on total confirmed cases of state 2(KY): ")
if Wald_ky_confirmed_2Sample > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample Wald's Test on total deaths of state 1(KS): ")
if Wald_ks_deaths_2Sample > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample Wald's Test on total deaths of state 2(KY): ")
if Wald_ky_deaths_2Sample > 1.96:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")
# Two sampled unpaired T test

def TtestTwoSample(feb_mean, march_mean, sample_pooled_std):
    t = (feb_mean - march_mean)/sample_pooled_std

    return abs(t)


def standardDeviationPooled(feb_data, march_data, feb_mean, march_mean):
    vfeb = 0
    for d in feb_data:
        vfeb += (d - feb_mean)**2

    std_feb = (vfeb / len(feb_data) - 1)**0.5

    vmar = 0
    for d in march_data:
        vmar += (d - march_mean) ** 2

    std_mar = (vfeb / len(march_data) - 1) ** 0.5

    std = np.sqrt((std_feb/len(feb_data)) + (std_mar/len(march_data)))
    return std

# Computing sample pooled standard deviation


std_ks_confirmed_pooled = standardDeviationPooled(ks_feb_confirmed, ks_march_confirmed, ks_feb_mean_confirmed_cases, ks_march_mean_confirmed_cases)
std_ky_confirmed_pooled = standardDeviationPooled(ky_feb_confirmed, ky_march_confirmed, ky_feb_mean_confirmed_cases, ky_march_mean_confirmed_cases)
std_ks_deaths_pooled = standardDeviationPooled(ks_feb_deaths, ks_march_deaths, ks_feb_mean_deaths, ks_march_mean_deaths)
std_ky_deaths_pooled = standardDeviationPooled(ky_feb_deaths, ky_march_deaths, ky_feb_mean_deaths, ky_march_mean_deaths)
# print(std_ks_confirmed_pooled)

t_test_two_sampled_ks_confirmed = TtestTwoSample(ks_feb_mean_confirmed_cases, ks_march_mean_confirmed_cases, std_ks_confirmed_pooled)
t_test_two_sampled_ky_confirmed = TtestTwoSample(ky_feb_mean_confirmed_cases, ky_march_mean_confirmed_cases, std_ky_confirmed_pooled)
t_test_two_sampled_ks_deaths = TtestTwoSample(ks_feb_mean_deaths, ks_march_mean_deaths, std_ks_deaths_pooled)
t_test_two_sampled_ky_deaths = TtestTwoSample(ky_feb_mean_deaths, ky_march_mean_deaths, std_ky_deaths_pooled)
print(t_test_two_sampled_ks_confirmed, t_test_two_sampled_ky_confirmed, t_test_two_sampled_ks_deaths, t_test_two_sampled_ky_deaths)

# t value  1.672029

print("###################################################")
print("Two sample unpaired T-Test on total confirmed cases of state 1(KS): ")
if t_test_two_sampled_ks_confirmed > 1.672029:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample unpaired T-Test on total confirmed cases of state 2(KY): ")
if t_test_two_sampled_ky_confirmed > 1.672029:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample unpaired T-Test on total deaths of state 1(KS): ")
if t_test_two_sampled_ks_deaths > 1.672029:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")

print("###################################################")
print("Two sample unpaired T-Test on total deaths of state 2(KY): ")
if t_test_two_sampled_ky_deaths > 1.672029:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is rejected.")
else:
    print("Null hypothesis mu1 = mu2 i.e. mean confirmed cases "
          "in state 1 for the month of March are equal to mean confirmed cases"
          " for the month of Feb is accepted.")