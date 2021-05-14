import pandas as pd
import numpy as np
import statistics as st
import csv
import matplotlib.pyplot as plt



borderdata = pd.read_csv("data/exploratory_dataset.csv")
data = borderdata.iloc[78:, 2:]

tx_bordercrossing_total = []
for i in range(27):
    tx_bordercrossing_total.append(int(data.iloc[0][i]))


def variance(data, data_mean):
    v = 0
    for d in data:
        v += (d - data_mean)**2

    return v


tx_confirm = pd.read_csv("data/tx_confirm_monthly.csv")
tx_confirm = tx_confirm.iloc[:, 1:]
tx_death = pd.read_csv("data/tx_death_monthly.csv")
tx_death = tx_death.iloc[:, 1:]
# print(tx_death)


tx_confirm_ls = []
for i in range(15):
    tx_confirm_ls.append(int(tx_confirm.iloc[i][1]))

tx_death_ls = []
for i in range(15):
    tx_death_ls.append(int(tx_death.iloc[i][1]))

date_ls = []
for i in range(15):
    date_ls.append(tx_death.iloc[i][0])

tx_bordercrossing_total = tx_bordercrossing_total[12:]
# print(tx_bordercrossing_total)
# print(tx_confirm_ls)
# print(tx_death)


#Performing pearson test
#computing standard deviation


def pearsonTest(x, y):
    var_x = variance(x, st.mean(x))
    var_y = variance(y, st.mean(y))
    x_bar = st.mean(x)
    y_bar = st.mean(y)
    lenth = len(x)

    n = 0
    for i in range(lenth):
        n += (x[i] - x_bar) * (y[i] - y_bar)

    d = (var_x * var_y)**0.5
    p = n/d
    return p


p_confirm = pearsonTest(tx_bordercrossing_total, tx_confirm_ls)
p_death = pearsonTest(tx_bordercrossing_total, tx_death_ls)
print("Pearson's statistic for texas COVID confirm cases and border crossing through the state of texas: ",abs(p_confirm),"\nPearson's statistic for texas COVID death cases and border crossing through the state of texas: ", abs(p_death))

if abs(p_confirm) <= 0.3:
    print("Border crossing data and COVID confirm cases for the state of texas has no correlation.")
elif abs(p_confirm) > 0.3:
    if p_confirm > 0.3:
        print("Border crossing data and COVID confirm cases for the state of texas has a positive linearly correlation.i.e both follow the same trend")
    elif p_confirm < -0.3:
        print("Border crossing data and COVID confirm cases for the state of texas has a negative linearly correlation.i.e both follow the opposite trend")

if abs(p_death) <= 0.3:
    print("Border crossing data and COVID death for the state of texas has no correlation.")
elif abs(p_death) > 0.3:
    if p_death> 0.3:
        print("Border crossing data and COVID death for the state of texas has a positive linearly correlation.i.e both follow the same trend")
    elif p_death < -0.3:
        print("Border crossing data and COVID death for the state of texas has a negative linearly correlation.i.e both follow the opposite trend")

# print(date_ls)
# print(tx_confirm_ls)
# print(tx_death_ls)
# print(tx_bordercrossing_total)

# plt.figure("Pearson's Test")
# plt.subplot(1, 2, 1)
# # newList = [x / 100 for x in tx_bordercrossing_total]
# plt.plot(date_ls, tx_bordercrossing_total ,label='Border crossing data.')
# plt.plot(date_ls, tx_confirm_ls ,label='Texas - Confirm cases.')
# plt.xlabel('Month')
# plt.ylabel('Values')
# plt.title("Pearson's Test linearity graph.")
# plt.legend(loc="upper left")
# plt.grid()
#
# plt.subplot(1, 2, 2)
# plt.plot(date_ls, tx_bordercrossing_total ,label='Border crossing data.')
# plt.plot(date_ls, tx_death_ls ,label='Texas - Death cases.')
# plt.xlabel('Month')
# plt.ylabel('Values')

# Performing permutation test
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


def do_permutation_test(x, y, s):
    threshold = 0.05
    n = 1000
    # ks_confirmed_OD, ks_deaths_OD, ky_confirmed_OD, ky_deaths_OD = get_data_oct_dec(input_data)
    null_hypothesis_cases = "Distribution of ks and ky confirmed cases are same"
    T_observed = abs(x.mean() - y.mean())
    print("\n\nFor border-crossing data and the {} cases for the state of Texas :\n".format(s))
    print("Mean of border-crossing : ", x.mean())
    print("Mean of {} cases : ".format(s), y.mean())

    pValue = calculate_pvalue(x, y, T_observed, n)
    print("p-value = " + str(pValue) + " for n = " + str(n))
    print('Given threshold is :', threshold)
    if pValue > threshold:
        print("Null hypothesis (" + null_hypothesis_cases + ") is accepted.")
    else:
        print("Null hypothesis (" + null_hypothesis_cases + ") is rejected.")

do_permutation_test(np.array(tx_bordercrossing_total), np.array(tx_confirm_ls), 'Confirm')
do_permutation_test(np.array(tx_bordercrossing_total), np.array(tx_death_ls), 'death')

# chi square test

data = pd.read_csv('data/exploratory_dataset.csv')
data.rename(columns={'Unnamed: 0': 'Port name'}, inplace=True)
data.rename(columns={'Unnamed: 1': 'Mode of crossing'}, inplace=True)


# getting the list of month-wise border crossing by pedestrians and personal vehicles
def monthly_lst_mode_crossing(mode_of_crossing):
    monthly_crossings = data.loc[data['Mode of crossing'] == mode_of_crossing]
    mode_lst = []
    for i in range(2, monthly_crossings.shape[1]):
        lst = monthly_crossings.iloc[:, i]
        lst = lst.tolist()
        lst1 = []
        for i in lst:
            if isinstance(i, str):
                lst1.append(float(i.replace(',', '')))
            else:
                lst1.append((float(i)))
        mode_lst.append(sum(lst1))
    return mode_lst


monthly_crossing_pedestrians = monthly_lst_mode_crossing("Pedestrians")
monthly_crossing_personal_vehicles = monthly_lst_mode_crossing("Personal Vehicles")
monthly_crossing_pedestrians = [x for x in monthly_crossing_pedestrians if np.isnan(x) == False]
monthly_crossing_personal_vehicles = [x for x in monthly_crossing_personal_vehicles if np.isnan(x) == False]
# print((monthly_crossing_pedestrians))
# print((monthly_crossing_personal_vehicles))

pedestrians_total_crossings_19 = sum(monthly_crossing_pedestrians[0:12])
pedestrians_total_crossings_20 = sum(monthly_crossing_pedestrians[12:24])
personal_vehicles_crossings_19 = sum(monthly_crossing_personal_vehicles[0:12])
personal_vehicles_crossings_20 = sum(monthly_crossing_personal_vehicles[12:24])

# chi square test for independence/association
# does mode of crossing(Pedestrian, Personal vehicles) depend of Covid-19
# null hypothesis(H_o) is mode of crossing is independent of covid-19
# alternate hypothesis(H_1) is mode of crossing is dependent on covid-19

total_pedestrian_crossings = pedestrians_total_crossings_19+pedestrians_total_crossings_20
total_pers_veh_crossings = personal_vehicles_crossings_19+personal_vehicles_crossings_20
chi_table = np.array([[pedestrians_total_crossings_19, pedestrians_total_crossings_20, total_pedestrian_crossings],
                     [personal_vehicles_crossings_19, personal_vehicles_crossings_20, total_pers_veh_crossings],
                     [pedestrians_total_crossings_19+personal_vehicles_crossings_19,
                      pedestrians_total_crossings_20+personal_vehicles_crossings_20,
                      total_pedestrian_crossings+total_pers_veh_crossings]])
e11 = (chi_table[2][0]/chi_table[2][2]) * chi_table[0][2]
e12 = (chi_table[2][1]/chi_table[2][2]) * chi_table[0][2]
e21 = (chi_table[2][0]/chi_table[2][2]) * chi_table[1][2]
e22 = (chi_table[2][1]/chi_table[2][2]) * chi_table[1][2]

expected_chi_table = np.array([[e11, e12], [e21, e22]])

q_obs = 0
# calculating Q_obs
for i in range(0, 2):
    for j in range(0, 2):
        q_obs += (expected_chi_table[i][j]-chi_table[i][j])**2 / expected_chi_table[i][j]

# degrees of freedom(df) = no_of_rows-1 * no_of_columns-1
df = (len(expected_chi_table[0])-1) * len(expected_chi_table)-1
print("Q_obs is:" + str(q_obs))
print("Degrees of freedom are:" + str(df))
print("From the look up table we find out that the p-value is < 0.00001 i.e p-value < alpha = 0.05\nTherefore"
      " the null hypothesis that mode of transport for border crossing(pedestrian and personal vehicles)"
      " and emergence of covid-19 are independent\n"
      "is rejected and we get that mode of transport for border crossing and emergence of covid-19 are dependent.")

# Performing Linear Regression


date_int_x = []
for i in range(len(date_ls)):
    date_int_x.append(i+1)

date_int_x = np.array(date_int_x)
tx_confirm_ls = np.array(tx_confirm_ls)
tx_death_ls = np.array(tx_death_ls)
#
# # average
# av_date_int_x = np.mean(date_int_x)
# av_tx_confirm_ls = np.mean(tx_confirm_ls)
# av_tx_death_ls = np.mean(tx_death_ls)
#
# # summations
#
# dateTimesConfirmSum = np.sum(np.multiply(tx_confirm_ls, date_int_x))
# dateTimesDeathSum = np.sum(np.multiply(tx_death_ls, date_int_x))
#
# squaresumDate = np.sum(np.square(date_int_x))

def linearRegression(date_int_x, y):
    date_int_x = np.array(date_int_x)
    y = np.array(y)
    # tx_death_ls = np.array(tx_death_ls)

    # average
    av_date_int_x = np.mean(date_int_x)
    av_y = np.mean(y)
    # av_tx_death_ls = np.mean(tx_death_ls)

    # summations

    dateTimesConfirmSum = np.sum(np.multiply(y, date_int_x))
    # dateTimesDeathSum = np.sum(np.multiply(tx_death_ls, date_int_x))

    squaresumDate = np.sum(np.square(date_int_x))
    B_1_US = (dateTimesConfirmSum - len(date_int_x) * av_date_int_x * av_y) / (squaresumDate - len(date_int_x) * (av_date_int_x) ** 2)

    B_0_US = av_y - B_1_US * av_date_int_x


    US_RegressionFit = []
    for y in date_int_x:
        US_RegressionFit.append(B_0_US + B_1_US * y)

    regressionFit = np.array(US_RegressionFit)

    return regressionFit, B_0_US, B_1_US

fitC, b0, b1 = linearRegression(date_int_x, tx_confirm_ls)
print("Predicting confirm cases for April 2021, May 2021, June 2021, July 2021, Aug 2021: \n")
l = len(date_int_x)
for i in range(5):
    print("Prediction of the confirm cases in Texas during {}/2021: {}".format(i+4, b0 + b1*(l +(i+1))))
fitD, b0, b1 = linearRegression(date_int_x, tx_death_ls)

print("Predicting confirm cases for April 2021, May 2021, June 2021, July 2021, Aug 2021: \n")
l = len(date_int_x)
for i in range(5):
    print("Prediction of the deaths in Texas during {}/2021: {}".format(i+4, b0 + b1*(l +(i+1))))

fitB, b0, b1 = linearRegression(date_int_x, tx_bordercrossing_total)
print("Predicting confirm cases for April 2021, May 2021, June 2021, July 2021, Aug 2021: \n")
l = len(date_int_x)
for i in range(5):
    print("Prediction of the border-crossing in Texas during {}/2021: {}".format(i+4, b0 + b1*(l +(i+1))))

plt.figure("Linear Regression")
plt.subplot(1, 3, 1)
# newList = [x / 100 for x in tx_bordercrossing_total]
plt.scatter(date_ls, tx_confirm_ls, color='goldenrod', marker='*' ,label='Texas - Confirm')
plt.plot(date_ls, fitC ,label='Regression  Fit')
plt.xlabel('Month')
plt.ylabel('Confirm Cases')
plt.title("Linear Regression for COVID confirm cases in Texas.")
plt.legend(loc="upper left")
plt.grid()

plt.subplot(1, 3, 2)
plt.scatter(date_ls, tx_death_ls, color='goldenrod', marker='*' ,label='Texas - Death')
plt.plot(date_ls, fitD ,label='Regression Fit')
plt.xlabel('Month')
plt.ylabel('Deaths')
plt.title("Linear Regression for COVID deaths in Texas.")
plt.legend(loc="upper left")
plt.grid()

plt.subplot(1, 3, 3)
plt.scatter(date_ls, tx_bordercrossing_total, color='goldenrod', marker='*' ,label='Texas - Border-crossings')
plt.plot(date_ls, fitB ,label='Regression Fit')
plt.xlabel('Month')
plt.ylabel('Boder-crossing values')
plt.title("Linear Regression for Border=Crossing in Texas.")
plt.legend(loc="upper left")
plt.grid()

plt.show()
