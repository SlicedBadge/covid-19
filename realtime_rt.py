'''
Module provides functions for updated rt over time
'''

import os
import pandas as pd
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d


#adding this to filepaths should allow script to reference other files relative to where it is
dir_path = os.path.dirname(__file__)

def highest_density_interval(pmf, p=.9, debug=False):
    '''
    TODO: add better documentation
    '''

    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])

def smooth_cases(cases, cutoff=25, smoothing_param = 7, return_original = True):
    '''
    Prepares a pandas series `cases` defining the cumulative case counts over time by applying a rolling average
    and cutting off the time series before there were a certain number of cases (`cutoff`)
    
    Parameters
    -----------
    cases : pd.Series
        Time series of cumulative case counts per day
    
    cutoff : int
        Time series starts after cutoff level is reached (everything before is dropped). Note that this cutoff is applied AFTER smoothing.
        
    smoothing_param : int
        The number of days to include in the smoothing window (passed into pd.Series.rolling) reccomend this number be odd so it is an even window
        
    return_original : bool
        True if you want the return to be a tuple (original data, smoothed data) where both have been truncated according to `cutoff`
        False if you just want smoothed data returned
    
    Outputs
    --------
    tuple of pd.Series if return_original is True
    
    pd.Series if return_original is false
    '''
    #turn cumulative cases into new cases per day
    new_cases = cases.diff()
    #replace negative values with 0
    new_cases[new_cases < 0] = 0
    
    #smooth the time series using a gaussian filter (the std of the filter is the smoothing param/3, default is 2) this is somewhat arbitrary
    smoothed = new_cases.rolling(smoothing_param,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=(smoothing_param - (smoothing_param%2)) / 3).round() #modulo deals with odd numbers so we get a nice even std
    
    #get the index of the first day where we are at the cutoff value
    idx_start = np.searchsorted(smoothed, cutoff)
    
    #filter the time series so that it only includes data after that day
    smoothed = smoothed.iloc[idx_start:]
    
    #return results
    if return_original:
        original = new_cases.loc[smoothed.index]
        return original, smoothed
    else:
        return smoothed


def get_posteriors(sr, sigma=0.15):
    '''
    Calculates the 
    
    '''
    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                            scale=sigma
                            ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood