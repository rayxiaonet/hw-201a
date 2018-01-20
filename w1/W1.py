import matplotlib.pyplot as plt
import xlrd
from sklearn import datasets, linear_model
import numpy as np

datasource = xlrd.open_workbook('w1-data.xlsx')

worksheet = datasource.sheet_by_name('Sheet1')

data = [[worksheet.cell_value(r, c) for r in range(1,worksheet.nrows)] for c in range(worksheet.ncols)]
#data[0]:year
#data[1]:A
#data[2]:B
#data[3]:C
#data[4]:D

max_year = max(data[0][:-1])
min_year = min(data[0][:-1])


def plot(plt,data,originalDataIndex):

    f, axarr = plt.subplots(2, 2)
    plots=['',axarr[0, 0],axarr[0, 1],axarr[1, 0],axarr[1,1]]

    for graph in range(1,5):
        min_y_A=min(data[graph][:-1])
        max_y_A=max(data[graph][:-1])

        regr = linear_model.LinearRegression()
        regr.fit(np.transpose(np.matrix(data[0][:originalDataIndex])), np.transpose(np.matrix(data[graph][:originalDataIndex])))
        expression= ' y = {0:.4f} * x + {1:.4f}'.format(regr.coef_[0][0].astype(float), regr.intercept_[0].astype(float))
        y=[regr.coef_[0][0] * min_year + regr.intercept_[0], regr.coef_[0][0] * max_year + regr.intercept_[0]]

        plots[graph].set_title(worksheet.cell_value(0, graph) + ': ' + expression)
        plots[graph].axis([1940, 1985, min_y_A, max_y_A])

        plots[graph].scatter(data[0][0:originalDataIndex], data[graph][0:originalDataIndex], color='red')
        plots[graph].plot([min_year, max_year], y, 'r--')

        plots[graph].scatter(data[0][originalDataIndex:len(data[graph])], data[graph][originalDataIndex:len(data[graph])], color='orange')
    return

data[1].pop(-1)

plot(plt,[data[0][1::2],data[1][1::2],data[2][1::2],data[3][1::2],data[4][1::2]],13)

plot(plt,data,26)


plt.show()

