import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


def linear_fit(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, std_err, r_value, intercept

def power_fit(x, y):
    x = np.array(x)
    y = np.array(y)
    ind = (x>0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[ind], np.log(y[ind]))
    return slope, std_err, r_value, np.exp(intercept)

def power_fit(x, y):
    x = np.array(x)
    y = np.array(y)
    ind = (x>0) & (y>0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(x[ind]), np.log(y[ind]))
    return slope, std_err, r_value, np.exp(intercept)


def loglog_regplot(x,y,confidence=0.95):
    xs = np.array([x_ for x_, _ in sorted(zip(x, y))])
    ys = np.array([y_ for _, y_ in sorted(zip(x, y))])
    X = sm.add_constant(np.log(xs))
    res = sm.OLS(np.log(ys), X).fit()

    st, data, ss2 = summary_table(res, alpha=1-confidence)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    y_fit  = np.exp(fittedvalues)
    y_low  = np.exp(predict_mean_ci_low)
    y_high = np.exp(predict_mean_ci_upp)
    stats  = {'intercept':res.params[1], 'exponent': res.params[1], 'R2':res.rsquared, 
    'half_confidence_interval_exponent_0.975':res.params[1] - res.conf_int()[1][0]}

    return xs, ys, y_fit, y_low, y_high, stats


def logx_regplot(x,y,confidence=0.95):
    xs = np.array([x_ for x_, _ in sorted(zip(x, y))])
    ys = np.array([y_ for _, y_ in sorted(zip(x, y))])
    
    X = sm.add_constant(np.log(xs))
    res = sm.OLS(ys, X).fit()

    st, data, ss2 = summary_table(res, alpha=1-confidence)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    y_fit  = fittedvalues
    y_low  = predict_mean_ci_low
    y_high = predict_mean_ci_upp
    stats  = {'intercept':res.params[1], 'exponent': res.params[1], 'R2':res.rsquared, 
    'half_confidence_interval_exponent_0.975':res.params[1] - res.conf_int()[1][0]}

    return xs, ys, y_fit, y_low, y_high, stats


def linear_regplot(x,y,confidence=0.95):
    xs = np.array([x_ for x_, _ in sorted(zip(x, y))])
    ys = np.array([y_ for _, y_ in sorted(zip(x, y))])
    
    X = sm.add_constant(xs)
    res = sm.OLS(ys, X).fit()

    st, data, ss2 = summary_table(res, alpha=1-confidence)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    y_fit  = np.exp(fittedvalues)
    y_low  = np.exp(predict_mean_ci_low)
    y_high = np.exp(predict_mean_ci_upp)
    stats  = {'intercept':res.params[1], 'slope': res.params[0], 'R2':res.rsquared, 
    'half_confidence_interval_slope_0.975':res.params[0] - res.conf_int()[0][0]}

    return xs, ys, y_fit, y_low, y_high, stats

"""
from rg_histograms import cumulative
from scipy.optimize import curve_fit

def doubleExp(x, a, b, c, d):
    return np.log(a * np.exp(-x/b) + c * np.exp(-x/d))

def singleExp(x, a, b):
    return a * x + b
"""

