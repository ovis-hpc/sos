'''
Created on Feb 2, 2017

@author: nichamon
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from pandas import DataFrame, Series

from numpy.core.numeric import arange


class Bollinger_band(object):

    def __init__(self, attr_name = None, window = 60, multi_sd = 2):
        '''
        Constructor of Bolling_band
        
        Arguments:
        --- attr_name: name of the data point to be calculated.
        --- window: window width. The default window width is 60.
        --- multi_sd: multiple of rolling standard deviation. The default of multi_sd is 2.
        '''
        self.window = window
        self.multi_sd = multi_sd
        self.attr_name = attr_name

    def set_attr_name(self, attr_name):
        '''
        Set the attr_name
        
        Positional Argument:
        --- attr_name: name of the data point to be calculated.
                       This is used in plot() function to construct the plot title.
        '''
        self.attr_name = attr_name

    def outliers(self, x, y, ub, lb):
        '''
        Calcuate the outliers of the data. 
        
        Any y values that are greater than the upper bound or
        less than the lower bound are outliers.
        
        Positional Arguments:
        --- x: the x-axis data
        --- y: the y-axis data to be calucated for outliers
        --- ub: Upper bound of the y values.
        --- lb: Lower bound of the y values.
        
        Returns:
        --- list of 2 numpy.ndarray's (x values corresponding to outliers, outliers)
            
        '''
        
        if (len(x) != len(ub)) or (len(x) != len(lb)):
            raise Exception("Length of 'x', 'y', 'ub', and 'lb' must be equal.")
        
        cond = np.logical_or(y > ub, y < lb)
        return(x[cond], y[cond])

    def calculate(self, x, y, window = None, multi_sd = None, is_outliers = True):
        '''
        Calculate the moving average (rolling average), rolling standard deviation,
        upper bound, and lower bounds of each time unit.
        
        The outliers could be included in the result by set is_outliers = True.
        
        Positional Arguments:
        --- x: a numpy.ndarray of the x-values.
        --- y: a numpy.ndarray of the y-values.
            The result are of the y-values.
        
        Optional Arguments:
        --- window: window width used to calculate the rolling average and standard deviation.
            If this is None, self.window is used. See the Constructor.
        --- multi_sd: Multiply of the rolling standard deviation.
            If this is None, self.multi_sd is used. See the Constructor.
        --- is_outliers: the result includes the outliers if it is true. The default is true.
        
        Returns:
        --- A dictionary x
            x['ma'] is the numpy.ndarray of rolling average
            x['std'] is the numpy.ndarray of rolling standard deviation
            x['upperband'] is the numpy.ndarray of the upperbound
            x['lowerband'] is the numpy.ndarray of the lowerbound
            x['window'] is the window width used in the calculation.
            x['multi_sd'] is the multiply of the rolling standard deviation used in the calculation.
            x['outlier'] is a list of 2 numpy.ndarray. See the Returns of 'outliers'
        '''
        try:
            if type(x) is not np.ndarray:
                raise Exception("'x' must be an numpy.ndarray")
                
            if type(y) is not np.ndarray:
                raise Exception("'y' must be an numpy.ndarray")
            
            df = DataFrame({'x': x, 'y': y})
            
            if window is None:
                window = self.window
            if multi_sd is None:
                multi_sd = self.multi_sd
            
            sma = np.array(df.y.rolling(window=window, center = False).mean()[window:])
            roll_std = np.array(df.y.rolling(window=window, center=False).std()[window:])
            ub = sma + (roll_std * multi_sd)
            lb = sma - (roll_std * multi_sd)
            
            result = {'ma': sma, 'std': roll_std,
                      'upperband': ub, 'lowerband': lb,
                      'window': window, 'multi_sd': multi_sd}
            
            if is_outliers:
                result['outliers'] = self.outliers(x[window:], y[window:], ub, lb)
            
            return (result)
        except:
            raise

    def plot(self, x, y, window = None, multi_sd = None, bb = None,
             xlabel = "timestamp", ylabel = None, title = None):
        '''
        Plot the Bollinger Band of the y value.

        Positional Arguments:
        --- x: x-value
        --- y: y-value
        
        Optional Arguments:
        --- window: window width used to calculate the rolling average and standard deviation.
            If this is None, self.window is used. See the Constructor.
        --- multi_sd: Multiply of the rolling standard deviation.
            If this is None, self.multi_sd is used. See the Constructor.
        --- bb: a result of the 'calculate' function.
        --- xlabel: The label of the x-axis. The default is 'timestamp'.
        --- ylabel: The label of the y-axis.
        --- title: The title of the plot. If it is None and self.attr_name is not None,
            the title is automatically constructed.
        '''
        
        if multi_sd is None:
            multi_sd = self.multi_sd
        if window is None:
            window = self.window

        if bb is None:
            bb = self.calculate(x, y, window, multi_sd)

        ma = bb['ma']
        ub = bb['upperband']
        lb = bb['lowerband']
        window = bb['window']
        
        fig = plt.figure()
        ax = fig.add_axes([0.075, 0.225, 0.875, 0.65], axisbg='w')
        ax.tick_params(labelsize = 10, direction = 'out')
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M:%S'))
        ax.fmt_xdata = DateFormatter('%m/%d %H:%M:%S')
        fig.autofmt_xdate()
        
        plt.plot(x[window:], ma, label='middle band', linewidth = 0.3, alpha=0.95)
        plt.plot(x[window:], ub, label='upper band', linewidth = 0.3, alpha=0.95)
        plt.plot(x[window:], lb, label='lower band', linewidth = 0.3, alpha=0.95)
        plt.fill_between(x[window:], lb, ub, facecolor = 'grey', alpha = 0.7)
        
        plt.plot(x[window:], y[window:], label='plot', linewidth = 1.3)
        
        plt.xlim(x[window + 1], x[-1])
        
        ts, outliers = bb['outliers']
        plt.scatter(ts, outliers, label='outliers', color='red', alpha=0.5)            
        
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        else:
            if self.attr_name:
                plt.title("{0} ({1}xsd ".format(self.attr_name, multi_sd) + 
                          "and window of {0})".format(window))
        
        ax.legend(loc='best')
        plt.grid(True)
        
        plt.show()