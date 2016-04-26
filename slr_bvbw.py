import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn import linear_model
import warnings


df = pd.read_fwf("brain_body.txt", header=0)
df = df[df.Brain < 1000]
df = df[df.Body < 1000]


def initialize_linreg_bvbw():
    warnings.warn("deprecated", DeprecationWarning)
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df['Brain'].values).reshape(-1, 1)
    y = df['Body'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    initialize_linreg_bvbw()


def output_bvbw_task_one(regression):
    print("Line of best fit: ", end='')
    print('Y = {0:.2f}X'.format(regression.coef_[0]), end='')
    print(' + {0:.2f}'.format(regression.intercept_))


def output_bvbw_task_two(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x), color='blue',
             linewidth=2)
    plt.title('Brain vs Body Weight', fontsize=20)
    plt.xticks((0, 100, 200, 300, 400, 500))
    plt.xlim((-25, 550))
    plt.xlabel('Brain Weight', fontsize=14)
    plt.yticks((0, 100, 200, 300, 400, 500, 600, 700))
    plt.ylim((-50, 750))
    plt.ylabel('Body Weight', fontsize=14)
    plt.show()


def output_bvbw_task_three(regression, transposed_x, y):
    print('Variance score: %.2f' % regression.score(transposed_x, y))
