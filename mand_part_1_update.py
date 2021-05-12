import pandas as pd
import numpy as np

data = pd.read_csv("9.csv")


def converting_cumulative_to_daily(cumulative_data):
    df = pd.DataFrame()
    df['Date'], df['KS confirmed'], df['KY confirmed'], \
    df['KS deaths'], df['KY deaths'] = [], [], [], [], []
    df.at[0] = cumulative_data.iloc[0]
    for i in range(1, cumulative_data.shape[0]):
        df.at[i, 'Date'] = cumulative_data.iloc[i]['Date']
        df.at[i, "KS confirmed"] \
            = cumulative_data.iloc[i]['KS confirmed'] - cumulative_data.iloc[i-1]['KS confirmed']
        df.at[i, 'KY confirmed'] = cumulative_data.iloc[i]['KY confirmed'] - cumulative_data.iloc[i-1]['KY confirmed']
        df.at[i, 'KS deaths'] = cumulative_data.iloc[i]['KS deaths'] - cumulative_data.iloc[i-1]['KS deaths']
        df.at[i, 'KY deaths'] = cumulative_data.iloc[i]['KY deaths'] - cumulative_data.iloc[i-1]['KY deaths']
    # print(df)
    return df


daily_data = converting_cumulative_to_daily(data)
# daily_data.to_csv('9_daily_cases.csv')


def check_null_values(input):
    ans = input.isnull().values.any()
    return ans


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
    if len(data1):
        print("\nOutliers present in the dataset are :\n", data1)
    return data1.iloc[:, 0]


def remove_outliers(ks_confirmed, ks_deaths, ky_confirmed, ky_deaths):
    outliers = []
    l1 = outlier_detection(ky_deaths)
    l2 = outlier_detection(ks_confirmed)
    l3 = outlier_detection(ky_confirmed)
    l4 = outlier_detection(ks_deaths)
    if len(l1):
        outliers.append(l1)
    if len(l2):
        outliers.append(l2)
    if len(l3):
        outliers.append(l3)
    if len(l4):
        outliers.append(l4)

    outliers = np.array(outliers)
    print('\nNumber of outliers present in the dataset :', len(outliers[0]))

    indexes = daily_data[daily_data['Date'].isin(outliers[0])].index
    daily_data.drop(indexes, inplace=True)


def mand_task_1(input):
    if not check_null_values(input):
        print("No missing values are present in the data.\n")
    else:
        print("some missing values are present in the data.\n")

    ks_confirmed = input.drop(['KY confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ks_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KY deaths'], axis=1)

    ky_confirmed = input.drop(['KS confirmed', 'KY deaths', 'KS deaths'], axis=1)
    ky_deaths = input.drop(['KS confirmed', 'KY confirmed', 'KS deaths'], axis=1)

    print('Before removing outliers, shape of dataset : ', daily_data.shape)
    remove_outliers(ks_confirmed, ks_deaths, ky_confirmed, ky_deaths)
    print('\nAfter removing outliers, shape of dataset : ', daily_data.shape)
    print("--------------------------------------------------------------------------------")
    daily_data.to_csv('updated_9.csv')
    print("Data saved to new csv file named 'updated_9.csv' after deleting outliers.")


mand_task_1(daily_data)
