'''
This module provide feature computation functions.
'''
from datetime import datetime
import numpy as np
from scipy import stats, fftpack
import utils

MAX_VAL = 1e6
MIN_VAL = 1e-6

def isWeekday(timestamp, offset):
    day = (datetime.utcfromtimestamp(timestamp) 
          + datetime.timedelta(second=offset))
    return int(day.isoweekday() < 6)


def act_type_one_hot(act_type):
    encoding = [0] * utils.NUM_ACT_TYPE
    encoding[act_type] = 1
    return encoding


# Time domain features start
def mean(data):
    '''
    Returns the mean value of the data. 
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    val = np.mean(data)
    
    return val


def mad(data):
    '''
    Returns the median absolute deviation (MAD)
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    
    val = np.median(np.abs(data - np.median(data)))
    
    return val


def raw_energy(data):
    '''
    Returns the total energy of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    
    total = 0
    for i in range(len(data)):
        total += data[i] ** 2
 
    val=total / len(data)

    return val


def mini(data):
    '''
    Returns the minimum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.min(data)
    
    return val


def maxi(data):
    '''
    Returns the maximum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.max(data)

    return val


def median(data):
    '''
    Returns the median value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.median(data)
    
    return val


def var(data):
    '''
    Returns the variance of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.var(data)
    
    return val


def std(data):
    '''
    Returns the standard deviation of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.std(data)
    
    return val


def ran(data):
    '''
    Returns the range of the data.
    '''    
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.max(data) - np.min(data)
    
    return val


def abs_mean(data):
    '''
    Returns the average of the absolute values of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    
    val = np.mean(np.absolute(data))
    
    return val


def coeff_var(data):
    '''
    Returns the coefficient of variation. 
    Measures signal dispersion.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    
    val = stats.variation(data)

    return val


def skewness(data):
    '''
    Returns the skewness (3rd moment) of the data. 
    Measures asymmetry of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = stats.skew(data)
    
    return val


def kurtosis(data):
    '''
    Returns the kurtosis (4th moment) of the data. 
    Measures peakedness of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = stats.kurtosis(data)
    
    return val

    
def quartile1(data):
    '''Returns the first quartile of the data.
    ''' 
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.percentile(data, 25)
    
    return val
   

def quartile3(data):
    '''
    Returns the third quartile of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.percentile(data, 75)
    
    return val


def iqr(data):
    '''
    Returns the difference between the 3rd and 1st quartile, also known as the 
    inter quartile range. Measures dispersion.
    '''    
    if not isinstance(data, np.ndarray):
        return np.nan

    val = quartile3(data) - quartile1(data)
    
    return val


def mcr(data):
    '''
    Returns the number of times the data crosses the mean value.
    Measures how often the signal varies.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    mean = np.mean(data)
    crossed = 0
    below = data[0] <= mean
    for i in data:
        if i <= mean and not below:
            below = True
            crossed += 1
        elif i > mean and below:
            below = False
            crossed += 1
            
    return crossed


def abs_area(data):
    '''
    Returns the absolute area, or the absolute sum of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.sum(np.abs(data))
    
    return val
    

def signal_mag_area(datax, datay, dataz):
    '''
    Returns the signal magnitude area across all three axes.
    '''
    if not isinstance(datax, np.ndarray):
        return np.nan
    if not isinstance(datay, np.ndarray):
        return np.nan
    if not isinstance(dataz, np.ndarray):
        return np.nan
    return abs_area(datax) + abs_area(datay) + abs_area(dataz)


def signal_vec_mag(datax, datay, dataz):
    '''
    Returns the signal vector magnitude across all three axes.
    '''
    if not isinstance(datax, np.ndarray):
        return np.nan
    if not isinstance(datay, np.ndarray):
        return np.nan
    if not isinstance(dataz, np.ndarray):
        return np.nan
    val = np.mean(np.sqrt(datax**2 + datay**2 + dataz**2))
    return val


def percentile(data, perc):
    '''
    Returns the given percentile of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = np.percentile(data, perc)
    
    return val


def rms(data):
    '''
    Returns the root mean square of the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    val = np.sqrt(np.mean(data**2))
    
    return val
    

def slope(data):
    '''
    Returns the slope between the first and last point.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    val = float(data[len(data) - 1] - data[0]) / (len(data) - 1)
    
    return val


def integral(data, dt=1):
    ''' Returns the integral of data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    val = np.trapz(data, dt)

    return val


def correlation(data1, data2):
    ''' Return pearson coefficient.
    '''
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        return np.nan

    val = np.corrcoef(data1, data2)

    return val

# Time domain features end


# Frequency domain features start

def dc_component(data, sampling_rate):
    '''
    Find dc component of data.
    Sampling_rate: number of samples per second
    '''

    X = fftpack.fft(data)
    val = np.abs(X[0])
    return val 


def bands_energy(data):
    '''
    Return energy accross frequency bands.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    fft_data = fftpack.fft(data)
    n = 64
    timestep = 0.01
    freq = fftpack.fftfreq(n, d=timestep)
    energy1to8 = np.sum(freq[0:8]**2)
    energy9to16 = np.sum(freq[8:16]**2)
    energy17to24 = np.sum(freq[16:24]**2)
    energy25to32 = np.sum(freq[25:32]**2)
    energy33to40 = np.sum(freq[32:40]**2)
    energy41to48 = np.sum(freq[40:48]**2)
    energy49to56 = np.sum(freq[48:56]**2)
    energy57to64 = np.sum(freq[57:64]**2)
    energy1to16 = np.sum(freq[0:16]**2)
    energy17to32 = np.sum(freq[16:32]**2)
    energy33to48 = np.sum(freq[32:48]**2)
    energy49to64 = np.sum(freq[48:64]**2)
    energy1to24 = np.sum(freq[0:24]**2)
    energy25to48 = np.sum(freq[24:48]**2)

    return [energy1to8, energy9to16, energy17to24, energy25to32, energy33to40, 
            energy41to48, energy49to56, energy57to64, energy1to16, energy17to32,
            energy33to48, energy49to64, energy1to24, energy25to48]


def energy(data):
    '''
    Calculated within the frequency domain. 
    Returns the total energy of the data in all frequencies.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    fft_data = fftpack.fft(data)
    fft_data = np.absolute(fft_data)
    
    half = int((len(fft_data) + 1) / 2)
    total = 0
    for i in range(half):
        total += fft_data[i] ** 2
    
    val = total
    
    return val


def entropy(data):
    '''
    Calculate within the frequency domain. Returns the impurity within the data.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    X = np.abs(fftpack.fft(data))
    
    psd = X**2 / len(data)
    
    psd = psd / np.sum(psd) # Normalize psd
    
    ln_psd = np.log(psd)

    val = np.sum(psd * psd_ln) * - 1

    return val
    
    # for i in fft_data:
    #     i = np.absolute(i)
    #     psd.append(i ** 2 / len(fft_data))
    # psd_tot = np.sum(psd)
    # pdf = []
    # for i in psd:
    #     pdf.append(i / psd_tot)
    # total = 0
    # for i in pdf:
    #     i = i 
    #     total += i * np.log(i)
    # 
    # val = -total
    
    # return val


def dom_freq_ratio(data):
    '''
    Calculated within the frequency domain. Returns the ratio between the 
    largest FFT coefficient and all FFT coefficients.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan

    fft_data = fftpack.fft(data)
    fft_data = np.abs(fft_data)
    div = np.sum(fft_data)
    if div == 0:
        return 0
    
    val = np.max(fft_data) / div
    
    return val

# Frequency domain feature ends
