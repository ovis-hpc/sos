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
        Constructor
        '''
        self.window = window
        self.multi_sd = multi_sd
        self.attr_name = attr_name

    def set_attr_name(self, attr_name):
        self.attr_name = attr_name

    def outliers(self, x, y, ub, lb):
        outliers = []
        ts = []
        for i in range(len(x)):
            if (y[i] > ub[i]) or (y[i] < lb[i]):
                outliers.append(y[i])
                ts.append(x[i])
        return((np.array(ts), np.array(outliers)))

    def calculate(self, x, y, window = None, multi_sd = None, is_outliers = True):
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
            
            sma = df.y.rolling(window=window, center = False).mean()
            roll_std = df.y.rolling(window=window, center=False).std()
            ub = sma + (roll_std * multi_sd)
            lb = sma - (roll_std * multi_sd)
            
            result = {'ma': sma[window:], 'std': roll_std[window:],
                      'upperband': ub[window:], 'lowerband': lb[window:],
                      'timestamp': x[window:], 
                      'window': window, 'multi_sd': multi_sd}
            
            if is_outliers:
                result['outliers'] = self.outliers(x[window:], y[window:], ub, lb)
            
            return (result)
        except:
            raise

    def plot(self, x, y, window = None, multi_sd = None, bb = None, is_outliers = True,
             xlabel = "timestamp", ylabel = None, title = None, 
             is_filled = True, is_show = True):
        
        if multi_sd is None:
            multi_sd = self.multi_sd
        if window is None:
            window = self.window

        if bb is None:
            if not is_outliers:
                bb = self.calculate(x, y, window, multi_sd, is_outliers = False)
            else:
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
        if is_filled:
            plt.fill_between(x[window:], lb, ub, facecolor = 'grey', alpha = 0.7)
        
        plt.plot(x[window:], y[window:], label='plot', linewidth = 1.3)
        
        plt.xlim(x[window + 1], x[-1])
        
        if is_outliers and ('outliers' not in bb.keys()):
            ts, outliers = self.outliers(x, y, bb)
        else:
            ts, outliers = bb['outliers']
        plt.scatter(ts, outliers, label='outliers', color='red', alpha=0.5)            
        
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        else:
            plt.title("{0} ({1}xsd ".format(self.attr_name, multi_sd) + 
                  "and window of {0})".format(window))
        
        ax.legend(loc='best')
        plt.grid(True)
        
        if is_show:
            plt.show()