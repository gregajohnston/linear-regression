import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn import linear_model
import warnings


ground_cricket_data = {"Chirps/Second":
                       [20.0, 16.0, 19.8, 18.4, 17.1, 15.5, 14.7,
                        15.7, 15.4, 16.3, 15.0, 17.2, 16.0, 17.0,
                        14.4],
                       "Ground Temperature":
                       [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7,
                        71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5,
                        76.3]}

df = pd.DataFrame(ground_cricket_data)


def initialize_linreg_gcd():
    warnings.warn("deprecated", DeprecationWarning)
    regression = linear_model.LinearRegression()
    transposed_x = np.array(df['Chirps/Second'].values).reshape(-1, 1)
    y = df['Ground Temperature'].values
    regression.fit(transposed_x, y)
    return regression, transposed_x, y

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    initialize_linreg_gcd()


def output_gcd_task_one(regression):
    print("Line of best fit: ", end='')
    print('Y = {0:.2f}X'.format(regression.coef_[0]), end='')
    print(' + {0:.2f}'.format(regression.intercept_))


def output_gcd_task_two(regression, transposed_x, y):
    plt.scatter(transposed_x, y,  color='black')
    plt.plot(transposed_x, regression.predict(transposed_x),
             color='blue', linewidth=2)
    plt.title('Cricket Noise Compared to Temperature', fontsize=20)
    plt.xticks((14, 15, 16, 17, 18, 19, 20))
    plt.xlim((13.9, 20.3))
    plt.xlabel('Chirps per Second', fontsize=14)
    plt.yticks((65, 70, 75, 80, 85, 90, 95, 100))
    plt.ylim((66, 97))
    plt.ylabel('Ground Temperature', fontsize=14)
    plt.show()


def output_gcd_task_three(regression, transposed_x, y):
    print('Variance score: {0:.2f}'.format(regression.score(transposed_x, y)))


def output_gcd_task_four(g_temp, regression):
    temp_to_cricket = (g_temp - regression.intercept_) / regression.coef_[0]
    print('Approximate Chirps/Second: {0:.2f}'.format(temp_to_cricket))


def output_gcd_task_five(chrps, regression):
    cricket_to_temp = (chrps * regression.coef_[0]) + regression.intercept_
    print('Approximate Ground Temperature: {0:.2f}'.format(cricket_to_temp))
