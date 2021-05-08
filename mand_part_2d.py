import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

new_data = pd.read_csv("data/updated_9.csv")
new_data = new_data.iloc[:, 1:]


# week_4data=ksy_deaths.iloc[0:28,3]
# x = np.linspace(0, 10000, 100)
# y = stats.expon.pdf(x, scale=1/alpha_mme)
# label=plt.plot(x, y)

def get_data_from_june(input_data):
    ksy_confirmed = input_data.drop(['KY deaths', 'KS deaths'], axis=1)
    ksy_deaths = input_data.drop(['KS confirmed', 'KY confirmed'], axis=1)

    combined_cases = ksy_confirmed['KS confirmed'] + ksy_confirmed['KY confirmed']
    ksy_confirmed['Combined cases'] = combined_cases

    combined_deaths = ksy_deaths['KS deaths'] + ksy_deaths['KY deaths']
    ksy_deaths['Combined deaths'] = combined_deaths

    ksy_confirmed = ksy_confirmed.loc[(ksy_confirmed['Date'] >= '2020-06-01')]
    ksy_deaths = ksy_deaths.loc[(ksy_deaths['Date'] >= '2020-06-01')]

    return ksy_confirmed, ksy_deaths


def bayesian_inference(week, data, alpha_g, beta_g, n, input_type):
    #     print('before calculation')
    #     print(alpha_g)
    #     print(beta_g)
    #     print('--------')
    X_bar = np.mean(data)
    # n = len(data)
    print(len(data))
    alpha_g = alpha_g + (n * X_bar)
    beta_g = beta_g + n
    #     print('after calculation')
    #     print(alpha_g)
    #     print(beta_g)
    #     print('*********************')

    lambda_MAP = (alpha_g + (n * X_bar)) / (beta_g + n)
    print("Week-" + str(week) + ":  Alpha - " + str(alpha_g) + ",  Beta - " + str(beta_g))
    print("MAP value is at lambda : " + str(lambda_MAP) + '\n')

    if input_type == 'death':
        x = np.linspace(740, 825, 100)
    else:
        x = np.linspace(25000, 29500, 100)
    y = stats.gamma.pdf(x, a=alpha_g, scale=1 / beta_g)
    label = plt.plot(x, y, label="Week" + str(week))
    return alpha_g, beta_g


def plot_death_cases():
    cases_KSY, deaths_KSY = get_data_from_june(new_data)
    data_june4weeks = deaths_KSY.iloc[0:28, :]
    mean = data_june4weeks.iloc[:, 3].mean()
    alpha_mme = 1 / mean
    alpha_g = 1
    beta_g = alpha_mme

    week = 4
    print("Week-4 : " + "Alpha - " + str(alpha_g) + ", Beta - " + str(beta_g) + '\n')
    week5_data = deaths_KSY.iloc[0:28 + 7, 3]
    alpha_g, beta_g = bayesian_inference(5, week5_data, alpha_g, beta_g, len(week5_data), 'death')

    week6_data = deaths_KSY.iloc[0:28 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(6, week6_data, alpha_g, beta_g, len(week6_data), 'death')

    week7_data = deaths_KSY.iloc[0:28 + 7 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(7, week7_data, alpha_g, beta_g, len(week7_data), 'death')

    week8_data = deaths_KSY.iloc[0:28 + 7 + 7 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(8, week8_data, alpha_g, beta_g, len(week8_data), 'death')

    plt.title("Posterior distributions")
    plt.xlabel("lambda")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()


def plot_confirmed_cases():
    cases_KSY, deaths_KSY = get_data_from_june(new_data)
    data_june4weeks = cases_KSY.iloc[0:28, :]
    mean = data_june4weeks.iloc[:, 3].mean()
    alpha_mme = 1 / mean
    alpha_g = 1
    beta_g = alpha_mme

    week = 4
    print("Week-4 : " + "Alpha - " + str(alpha_g) + ", Beta - " + str(beta_g) + '\n')
    week5_data = cases_KSY.iloc[0:28 + 7, 3]
    alpha_g, beta_g = bayesian_inference(5, week5_data, alpha_g, beta_g, len(week5_data), 'confirmed')

    week6_data = cases_KSY.iloc[0:28 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(6, week6_data, alpha_g, beta_g, len(week6_data), 'confirmed')

    week7_data = cases_KSY.iloc[0:28 + 7 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(7, week7_data, alpha_g, beta_g, len(week7_data), 'confirmed')

    week8_data = cases_KSY.iloc[0:28 + 7 + 7 + 7 + 7, 3]
    alpha_g, beta_g = bayesian_inference(8, week8_data, alpha_g, beta_g, len(week8_data), 'confirmed')

    plt.title("Posterior distributions")
    plt.xlabel("lambda")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()


plot_death_cases()
plot_confirmed_cases()