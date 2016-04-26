import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn import linear_model
import warnings

df = pd.read_fwf("salary.txt", header=None,
                 names=["Sex", "Rank", "Year", "Degree", "YSdeg", "Salary"])


def initialize_linreg_sd():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex', 'Rank', 'Year',
                                'Degree', 'YSdeg']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd()


def output_sd_task_one(regression):
    print("Line of best fit: ", end='')
    print('Y = {0:.2f}sex'.format(regression.coef_[0]), end='')
    print(' + {0:.2f}rank'.format(regression.coef_[1]), end='')
    print(' + {0:.2f}year'.format(regression.coef_[2]), end='')
    print(' + {0:.2f}degree'.format(regression.coef_[3]), end='')
    print(' + {0:.2f}ysdeg'.format(regression.coef_[4]), end='')
    print(' + {0:.2f}'.format(regression.intercept_))


def output_sd_task_two(regression, transposed_x, y):
    print('Variance score all vars: {0:.2f}'.format(
            regression.score(transposed_x, y)))

    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex', 'Rank', 'Year', 'Degree']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    print('Variance score less YSdeg: {0:.2f}'.format(
            regression.score(transposed_x, y)))

    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex', 'Rank', 'Year', 'YSdeg']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    print('Variance score less Degree: {0:.2f}'.format(
            regression.score(transposed_x, y)))

    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex', 'Rank', 'Degree', 'YSdeg']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    print('Variance score less Year: {0:.2f}'.format(
            regression.score(transposed_x, y)))

    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex', 'Rank', 'Degree', 'YSdeg']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    print('Variance score less Rank: {0:.2f}'.format(
            regression.score(transposed_x, y)))

    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Rank', 'Year', 'Degree', 'YSdeg']].values)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    print('Variance score less Sex: {0:.2f}'.format(
            regression.score(transposed_x, y)))


def output_sd_task_three(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary Compared with Employment Measures', fontsize=20)
    plt.xticks((14, 15, 16, 17, 18, 19, 20))
    plt.xlim((13.9, 20.3))
    plt.xlabel('Other variables', fontsize=14)
    plt.yticks((65, 70, 75, 80, 85, 90, 95, 100))
    plt.ylim((66, 97))
    plt.ylabel('Salary', fontsize=14)
    plt.show()


def initialize_linreg_sd_SEX():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Sex']].values).reshape(-1, 1)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd_SEX()


def graph_sd_three_SEX(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary vs Gender', fontsize=20)
    plt.xticks((0, 1))
    plt.xlim((-0.5, 1.5))
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Dollars', fontsize=14)
    plt.show()


def initialize_linreg_sd_RANK():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Rank']].values).reshape(-1, 1)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd_RANK()


def graph_sd_three_RANK(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary vs Employment Rank', fontsize=20)
    plt.xticks((1, 2, 3))
    plt.xlim((0.5, 3.5))
    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Dollars', fontsize=14)
    plt.show()


def initialize_linreg_sd_YEAR():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Year']].values).reshape(-1, 1)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd_YEAR()


def graph_sd_three_YEAR(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary vs Years of Work', fontsize=20)
    plt.xticks((0, 5, 10, 15, 20, 25))
    plt.xlim((-2, 27))
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Dollars', fontsize=14)
    plt.show()


def initialize_linreg_sd_DEGREE():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['Degree']].values).reshape(-1, 1)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd_DEGREE()


def graph_sd_three_DEGREE(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary vs Degree', fontsize=20)
    plt.xticks((0, 1))
    plt.xlim((-0.5, 1.5))
    plt.xlabel('Degree Category', fontsize=14)
    plt.ylabel('Dollars', fontsize=14)
    plt.show()


def initialize_linreg_sd_YSdeg():
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df[['YSdeg']].values).reshape(-1, 1)
    y = df['Salary'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialize_linreg_sd_YSdeg()


def graph_sd_three_YSdeg(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Salary vs Years Since Degree', fontsize=20)
    plt.xticks((0, 5, 10, 15, 20, 25, 30, 35))
    plt.xlim((-2, 37))
    plt.xlabel('Years', fontsize=14)
    plt.ylabel('Dollars', fontsize=14)
    plt.show()
