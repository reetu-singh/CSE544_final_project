
def get_data_oct_dec():
    # ks_confirmed_OD
    cases_ks = ks_confirmed.loc[
                   (ks_confirmed['Date'] >= '2020-10-01') & (ks_confirmed['Date'] <= '2020-12-31')].iloc[:, 1]
    deaths_ks = ks_deaths.loc[(ks_deaths['Date'] >= '2020-10-01') & (ks_deaths['Date'] <= '2020-12-31')].iloc[:, 1]
    cases_ky = ky_confirmed.loc[
                   (ky_confirmed['Date'] >= '2020-10-01') & (ky_confirmed['Date'] <= '2020-12-31')].iloc[:, 1]
    deaths_ky = ky_deaths.loc[(ky_deaths['Date'] >= '2020-10-01') & (ky_deaths['Date'] <= '2020-12-31')].iloc[:, 1]
    return cases_ks, deaths_ks, cases_ky, deaths_ky


def permutation_test(union, sizeA, sizeB):
    np.random.shuffle(union)
    listA = union[:sizeA]
    listB = union[-sizeB:]
    return abs(listA.mean() - listB.mean())


def calculate_pValue(listA, listB, Tobs, n):
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


def q2_part_c_i(n):  # n is no of permutations.
    threshold = 0.05
    null_hypothesis = "Distribution of ks and ky confirmed cases are same"
    ks_confirmed_OD, ks_deaths_OD, ky_confirmed_OD, ky_deaths_OD = get_data_oct_dec()
    Tobserved = abs(ks_confirmed_OD.mean() - ky_confirmed_OD.mean())

    print("Mean of ks confirmed cases : " + str(ks_confirmed_OD.mean()))
    print("Mean of ky confirmed cases : " + str(ky_confirmed_OD.mean()))

    pValue = calculate_pValue(ks_confirmed_OD, ky_confirmed_OD, Tobserved, n)
    print("p-value = " + str(pValue) + " when n = " + str(n))
    if pValue > threshold:
        print("Null hypothesis (" + null_hypothesis + ") is accepted.")

    else:
        print("Null hypothesis (" + null_hypothesis + ") is rejected.")


q2_part_c_i(1000)


def q2_part_c_ii(n):  # n is no of permutations.
    threshold = 0.05
    null_hypothesis = "Distribution of ks and ky death cases are same"

    Tobserved = abs(ks_deaths_OD.mean() - ky_deaths_OD.mean())

    print("Mean of ks death cases : " + str(ks_deaths_OD.mean()))
    print("Mean of ky death cases : " + str(ky_deaths_OD.mean()))

    pValue = calculate_pValue(ks_deaths_OD, ky_deaths_OD, Tobserved, n)
    print("p-value = " + str(pValue) + " when n = " + str(n))
    if pValue > threshold:
        print("Null hypothesis (" + null_hypothesis + ") is accepted.")

    else:
        print("Null hypothesis (" + null_hypothesis + ") is rejected.")


q2_part_c_ii(1000)


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


a = np.array(ks_confirmed_OD)
a[0]

binomial_n_p(ks_confirmed_OD)

ks_confirmed_OD.mean()

from decimal import Decimal


def poisson_CDF(lmda, x):
    #     summation = 0
    #     for i in range(x+1):
    #         summation+=((lmda**i))/Decimal(math.factorial(i))

    #     return (summation) * (math.exp(-1*lmda))
    return (math.exp(x - lmda))


# * (math.exp(x))

def gemetric_CDF(p, x):
    return 1 - ((1 - p) ** x)


def binomial_CDF(n, p, x):
    summation = 0.0
    for i in range(x + 1):
        n_C_i = math.comb(n, i)
        summation += n_C_i * (p ** i) * ((1 - p) ** (n - i))
    return summation


# In[30]:


import matplotlib.pyplot as plt

lambdaa = poisson_lambda(ks_confirmed_OD)
print(lambdaa)

sorted_ks_confirmed = np.sort(ks_confirmed_OD)
delta = 0.1
X = [sorted_ks_confirmed[0] - delta]
Y = [0]

for i in range(len(ks_confirmed_OD)):
    ecdf_poisson = poisson_CDF(lambdaa, sorted_ks_confirmed[i])
    X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
    Y = Y + [ecdf_poisson, ecdf_poisson]

X = X + [np.max(ks_confirmed_OD) + delta]
Y = Y + [1]

sorted_ky_confirmed = np.sort(ky_confirmed_OD)
delta2 = 0.1

X2 = [sorted_ky_confirmed[0] - delta2]
Y2 = [0]

for i in range(len(ky_confirmed_OD)):
    X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
    Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]

X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
Y2 = Y2 + [1]

max_ks = int(np.max(ks_confirmed_OD))
max_ky = int(np.max(ky_confirmed_OD))
max_cases = max(max_ks, max_ky)

maximum_difference = 0
for i in range(max_cases + 1):
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
             xy=(x_max_diff, 0.2),
             xytext=(x_max_diff, 0.5),
             arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))

plt.plot(X, Y, label="eCDF of Males ages")
plt.plot(X2, Y2, label="eCDF of Females ages")
plt.xlabel('x')
plt.ylabel('Pr[X<=x]')
plt.title("eCDF of male and female ages")
plt.legend(loc="upper left")
plt.grid()
plt.show()

# In[33]:


import matplotlib.pyplot as plt

threshold = 0.05

pmme = geometric_p(ks_confirmed_OD)
print(pmme)

sorted_ks_confirmed = np.sort(ks_confirmed_OD)
delta = 0.1
X = [sorted_ks_confirmed[0] - delta]
Y = [0]

for i in range(len(ks_confirmed_OD)):
    ecdf_geometric = gemetric_CDF(pmme, sorted_ks_confirmed[i])
    X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
    Y = Y + [ecdf_geometric, ecdf_geometric]

X = X + [np.max(ks_confirmed_OD) + delta]
Y = Y + [1]

sorted_ky_confirmed = np.sort(ky_confirmed_OD)
delta2 = 0.1

X2 = [sorted_ky_confirmed[0] - delta2]
Y2 = [0]

for i in range(len(ky_confirmed_OD)):
    X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
    Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]

X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
Y2 = Y2 + [1]

max_ks = int(np.max(ks_confirmed_OD))
max_ky = int(np.max(ky_confirmed_OD))
max_cases = max(max_ks, max_ky)

maximum_difference = 0
for i in range(max_cases + 1):
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
             xy=(x_max_diff, 0.2),
             xytext=(x_max_diff, 0.5),
             arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))

plt.plot(X, Y, label="eCDF of Males ages")
plt.plot(X2, Y2, label="eCDF of Females ages")
plt.xlabel('x')
plt.ylabel('Pr[X<=x]')
plt.title("eCDF of male and female ages")
plt.legend(loc="upper left")
plt.grid()
plt.show()

# In[70]:


import matplotlib.pyplot as plt

threshold = 0.05

pmme = geometric_p(ks_deaths_OD)
print(pmme)

sorted_ks_deaths = np.sort(ks_deaths_OD)
delta = 0.1
X = [sorted_ks_deaths[0] - delta]
Y = [0]

for i in range(len(ks_deaths_OD)):
    ecdf_geometric = gemetric_CDF(pmme, sorted_ks_deaths[i])
    X = X + [sorted_ks_deaths[i], sorted_ks_deaths[i]]
    Y = Y + [ecdf_geometric, ecdf_geometric]

X = X + [np.max(ks_deaths_OD) + delta]
Y = Y + [1]

sorted_ky_deaths = np.sort(ky_deaths_OD)
delta2 = 0.1

X2 = [sorted_ky_deaths[0] - delta2]
Y2 = [0]

for i in range(len(ky_deaths_OD)):
    X2 = X2 + [sorted_ky_deaths[i], sorted_ky_deaths[i]]
    Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_deaths_OD)]

X2 = X2 + [np.max(ky_deaths_OD) + delta2]
Y2 = Y2 + [1]

max_ks = int(np.max(ks_deaths_OD))
max_ky = int(np.max(ky_deaths_OD))
max_deaths = max(max_ks, max_ky)

maximum_difference = 0
for i in range(max_deaths + 1):
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
             xy=(x_max_diff, 0.2),
             xytext=(x_max_diff, 0.5),
             arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))

plt.plot(X, Y, label="eCDF of Males ages")
plt.plot(X2, Y2, label="eCDF of Females ages")
plt.xlabel('x')
plt.ylabel('Pr[X<=x]')
plt.title("eCDF of male and female ages")
plt.legend(loc="upper left")
plt.grid()
plt.show()

# In[72]:


import matplotlib.pyplot as plt

threshold = 0.05

pmme, nmme = binomial_n_p(ks_confirmed_OD)
nmme = int(nmme)
# pmme=listt[0]
# nmme=listt[1]
print('----')
print(pmme)
print(nmme)

sorted_ks_confirmed = np.sort(ks_confirmed_OD)
delta = 0.1
X = [sorted_ks_confirmed[0] - delta]
Y = [0]

for i in range(len(ks_confirmed_OD)):
    ecdf_binomial = binomial_CDF(nmme, pmme, sorted_ks_confirmed[i])
    X = X + [sorted_ks_confirmed[i], sorted_ks_confirmed[i]]
    Y = Y + [ecdf_binomial, ecdf_binomial]

X = X + [np.max(ks_confirmed_OD) + delta]
Y = Y + [1]

sorted_ky_confirmed = np.sort(ky_confirmed_OD)
delta2 = 0.1

X2 = [sorted_ky_confirmed[0] - delta2]
Y2 = [0]

for i in range(len(ky_confirmed_OD)):
    X2 = X2 + [sorted_ky_confirmed[i], sorted_ky_confirmed[i]]
    Y2 = Y2 + [Y2[-1], Y2[-1] + 1 / len(ky_confirmed_OD)]

X2 = X2 + [np.max(ky_confirmed_OD) + delta2]
Y2 = Y2 + [1]

max_ks = int(np.max(ks_confirmed_OD))
max_ky = int(np.max(ky_confirmed_OD))
max_cases = max(max_ks, max_ky)

maximum_difference = 0
for i in range(max_cases + 1):
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
             xy=(x_max_diff, 0.2),
             xytext=(x_max_diff, 0.5),
             arrowprops=dict(facecolor='black', linewidth=2, shrink=0.001))

plt.plot(X, Y, label="eCDF of Males ages")
plt.plot(X2, Y2, label="eCDF of Females ages")
plt.xlabel('x')
plt.ylabel('Pr[X<=x]')
plt.title("eCDF of male and female ages")
plt.legend(loc="upper left")
plt.grid()
plt.show()