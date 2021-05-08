import pandas as pd
import numpy as np

data = pd.read_csv("data/9.csv")
columns = data.columns


def check_null_values(input):
    ans = input.isnull().values.any()
    return ans


def get_data_columns():
    ks_confirmed = data.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ks_deaths = data.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)

    ky_confirmed = data.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ky_deaths = data.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)
    return ks_confirmed, ks_deaths, ky_confirmed, ky_deaths


def outlier_detection(data1):
    df = data1.iloc[:, 1]
    n = df.size
    df = df.sort_values(ascending=True)
    q1 = df[int(np.ceil(0.25 * n))]
    q3 = df[int(np.ceil(0.75 * n))]
    iqr = q3 - q1
    alpha = 1.5
    upper_limit = q3 + alpha * iqr
    lower_limit = q1 - alpha * iqr
    data1 = data1.loc[((df < lower_limit) | (df > upper_limit))]
    return data1.iloc[:, 0]


# outlier_detection(ky_deaths)


def remove_outliers(ks_confirmed, ks_deaths, ky_confirmed, ky_deaths):
    outliers = []
    if len(outlier_detection(ky_deaths)):
        outliers.append(outlier_detection(ky_deaths))
    if len(outlier_detection(ks_confirmed)):
        outliers.append(outlier_detection(ks_confirmed))
    if len(outlier_detection(ky_confirmed)):
        outliers.append(outlier_detection(ky_confirmed))
    if len(outlier_detection(ks_deaths)):
        outliers.append(outlier_detection(ks_deaths))

    outliers = np.array(outliers)

    print("\nOutliers present in the dataset are :\n", outliers)

    indexes = data[data['Date'].isin(outliers[0])].index
    data.drop(indexes, inplace=True)


def mand_task_1(input):
    if not check_null_values(input):
        print("No missing values are present in the data.\n")
    else:
        print("some missing values are present in the data.\n")

    ks_confirmed = input.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ks_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)

    ky_confirmed = input.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ky_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)

    print('Before removing outliers, shape of dataset : ', data.shape)
    remove_outliers(ks_confirmed, ks_deaths, ky_confirmed, ky_deaths)
    print('\nAfter removing outliers, shape of dataset : ', data.shape)
    print("--------------------------------------------------------------------------------")
    print("Saving dataset after deleting outliers into new csv file named: 'updated_9.csv'")
    data.to_csv('updated_9.csv')


mand_task_1(data)

# ks_confirmed = data.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
# ks_deaths = data.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)
#
# ky_confirmed = data.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
# ky_deaths = data.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)
