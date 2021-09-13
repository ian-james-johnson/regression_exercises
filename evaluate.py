import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pydataset import data



def plot_residuals (x, y):
    '''This function takes x column and y column as arguments. It creates a plot of residuals.'''
    ax1 = sns.residplot(x , y)
    #ax1.set(ylabel='Tips ($)', xlabel='Total Bill ($)')
    return



def regression_errors(y, yhat):
    '''This functions takes y and yhat and returns regression error values.'''

    # MSE2
    mse2 = mean_squared_error(y, yhat)
    print('MSE is ',round(mse2,2))

    # RMSE
    rmse = mse2 ** 0.5
    print('RMSE is  ',round(rmse,2))

    # SSE, sum of squared error
    sse2 = mean_squared_error(y, yhat)*len(y)
    print('SSE is ',round(sse2,2))

    # ESS
    ess = sum((yhat -y.mean())**2)
    print('ESS is ',round(ess,2))

    return mse2, rmse, sse2, ess



def baseline_mean_errors(y):
    '''This function takes in y column and returns the error statistics.'''

    baseline = y.mean()
    residuals = baseline - y
    residuals_squared = sum(residuals**2)
    
    SSE = residuals_squared
    
    MSE = SSE/len(y)
    
    RMSE = MSE ** 0.5
    
    d =  {'SSE': [round(SSE,2)],
      'MSE': [round(MSE,2)],
      'RMSE': [round(RMSE,2)]}
    
    baseline_errors_df=pd.DataFrame(d,index=['Baseline Reggression Errors'])
    return baseline_errors_df.T

    def better_than_baseline(y,yhat):
        baseline = y.mean()
        residuals_baseline = baseline - y
        residuals_squared_baseline = sum(residuals_baseline**2)
        SSE_baseline = residuals_squared_baseline
    
        MSE_baseline = SSE_baseline/len(y)
    
        RMSE_baseline = MSE_baseline ** 0.5
    
        residuals = yhat - y
        residuals_squared = sum(residuals**2)
        SSE = residuals_squared
    
        MSE = sklearn.metrics.mean_squared_error(y,yhat)
    
        RMSE = (sklearn.metrics.mean_squared_error(y,yhat)) ** 0.5
    
        if RMSE < RMSE_baseline:
            return True
        else: 
            return False
