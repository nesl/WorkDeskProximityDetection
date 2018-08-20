'''
This module provide feature computation functions.
'''
from datetime import datetime, timedelta
import numpy as np
from scipy import stats, fftpack
import utils

MAX_VAL = 1e6
MIN_VAL = 1e-6

def outlier_check(val):
    if np.isnan(val):
        print('val is nan')
        import pdb
        pdb.set_trace()
        return np.nan
    elif val > MAX_VAL:
        return MAX_VAL
    elif val < MIN_VAL:
        return MIN_VAL
    else:
        return val

def is_weekday(timestamp, offset):
    day = (datetime.utcfromtimestamp(timestamp) 
           + timedelta(seconds=offset))
    val = int(day.isoweekday() < 6)
    return outlier_check(val)


def act_type_one_hot(data):
    if not isinstance(data, np.ndarray):
        print('act_type_one_hot:', 'input is not numpy array')
        return np.nan

    most_common_type = int(stats.mode(data).mode[0])
    if most_common_type > 7 or most_common_type <0:
        print("Act type is output range.")
    encoding = [0] * utils.NUM_ACT_TYPE
    encoding[most_common_type] = 1
    return encoding


# Time domain features start
def mean(data):
    '''
    Returns the mean value of the data. 
    '''
    if not isinstance(data, np.ndarray):
        print('mean:', 'input is not numpy array')
        return np.nan
    val = np.mean(data)
    
    return outlier_check(val)


def mad(data):
    '''
    Returns the median absolute deviation (MAD)
    '''
    if not isinstance(data, np.ndarray):
        print('mad:', 'input is not numpy array')
        return np.nan
    
    val = np.median(np.abs(data - np.median(data)))
    
    return outlier_check(val)


def raw_energy(data):
    '''
    Returns the total energy of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('raw_energy:', 'input is not numpy array')
        return np.nan
    
    total = 0
    for i in range(len(data)):
        total += data[i] ** 2
 
    val = total / len(data)

    return outlier_check(val)


def mini(data):
    '''
    Returns the minimum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('mini:', 'input is not numpy array')
        return np.nan

    val = np.min(data)
    
    return outlier_check(val)


def maxi(data):
    '''
    Returns the maximum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('maxi:', 'input is not numpy array')
        return np.nan

    val = np.max(data)

    return outlier_check(val)


def median(data):
    '''
    Returns the median value of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('median:', 'input is not numpy array')
        return np.nan

    val = np.median(data)
    
    return outlier_check(val)


def var(data):
    '''
    Returns the variance of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('var:', 'input is not numpy array')
        return np.nan

    val = np.var(data)
    
    return outlier_check(val)


def std(data):
    '''
    Returns the standard deviation of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('std:', 'input is not numpy array')
        return np.nan

    val = np.std(data)
    
    return outlier_check(val)


def ran(data):
    '''
    Returns the range of the data.
    '''    
    if not isinstance(data, np.ndarray):
        print('ran:', 'input is not numpy array')
        return np.nan

    val = np.max(data) - np.min(data)
    
    return outlier_check(val)


def abs_mean(data):
    '''
    Returns the average of the absolute values of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('abs_mean:', 'input is not numpy array')
        return np.nan
    
    val = np.mean(np.absolute(data))
    
    return outlier_check(val)


def coeff_var(data):
    '''
    Returns the coefficient of variation. 
    Measures signal dispersion.
    '''
    if not isinstance(data, np.ndarray):
        print('coeff_var:', 'input is not numpy array')
        return np.nan
    
    val = stats.variation(data)

    if np.isnan(val):
        val = MAX_VAL

    return outlier_check(val)


def skewness(data):
    '''
    Returns the skewness (3rd moment) of the data. 
    Measures asymmetry of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('skewness:', 'input is not numpy array')
        return np.nan

    val = stats.skew(data)
    
    return outlier_check(val)


def kurtosis(data):
    '''
    Returns the kurtosis (4th moment) of the data. 
    Measures peakedness of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('kurtosis:', 'input is not numpy array')
        return np.nan

    val = stats.kurtosis(data)
    
    return outlier_check(val)

    
def quartile1(data):
    '''Returns the first quartile of the data.
    ''' 
    if not isinstance(data, np.ndarray):
        print('quartile1:', 'input is not numpy array')
        return np.nan

    val = np.percentile(data, 25)
    
    return outlier_check(val)
   

def quartile3(data):
    '''
    Returns the third quartile of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('quartile3:', 'input is not numpy array')
        return np.nan

    val = np.percentile(data, 75)
    
    return outlier_check(val)


def iqr(data):
    '''
    Returns the difference between the 3rd and 1st quartile, also known as the 
    inter quartile range. Measures dispersion.
    '''    
    if not isinstance(data, np.ndarray):
        print('iqr:', 'input is not numpy array')
        return np.nan

    val = quartile3(data) - quartile1(data)
    
    return outlier_check(val)


def mcr(data):
    '''
    Returns the number of times the data crosses the mean value.
    Measures how often the signal varies.
    '''
    if not isinstance(data, np.ndarray):
        print('mcr:', 'input is not numpy array')
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
            
    return outlier_check(crossed)


def abs_area(data):
    '''
    Returns the absolute area, or the absolute sum of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('abs_area:', 'input is not numpy array')
        return np.nan

    val = np.sum(np.abs(data))
    
    return outlier_check(val)
    

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
    val = abs_area(datax) + abs_area(datay) + abs_area(dataz)
    return outlier_check(val)

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
    return outlier_check(val)


def percentile(data, perc):
    '''
    Returns the given percentile of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('percentile:', 'input is not numpy array')
        return np.nan

    val = np.percentile(data, perc)
    
    return outlier_check(val)


def rms(data):
    '''
    Returns the root mean square of the data.
    '''
    if not isinstance(data, np.ndarray):
        print('rms:', 'input is not numpy array')
        return np.nan
    val = np.sqrt(np.mean(data**2))
    
    return outlier_check(val)
    

def slope(data):
    '''
    Returns the slope between the first and last point.
    '''
    if not isinstance(data, np.ndarray):
        print('slope:', 'input is not numpy array')
        return np.nan

    val = float(data[len(data) - 1] - data[0]) / (len(data) - 1)
    
    return outlier_check(val)


def integral(data, dt=1):
    ''' Returns the integral of data.
    '''
    if not isinstance(data, np.ndarray):
        print('integral:', 'input is not numpy array')
        return np.nan
    val = np.trapz(data, dx=dt)

    return outlier_check(val)


def correlation(data1, data2):
    ''' Return pearson coefficient.
    '''
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        return np.nan

    val = np.corrcoef(data1, data2)

    return outlier_check(val)

# Time domain features end


# Frequency domain features start

def dc_component(data, sampling_rate):
    '''
    Find dc component of data.
    Sampling_rate: number of samples per second
    '''
    if not isinstance(data, np.ndarray):
        print('dc_component:', 'input is not numpy array')
        return np.nan
    
    X = fftpack.fft(data)
    val = np.abs(X[0])
    return outlier_check(val)


def bands_energy(data, sampling_rate):
    '''
    Return energy accross frequency bands.
    '''
    # TODO: This function has a bug. The computation is not correct.
    if not isinstance(data, np.ndarray):
        return np.nan

    fft_data = fftpack.fft(data)
    n = 64
    timestep = 1/samping_rate
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
        print('energy:', 'input is not numpy array')
        return np.nan

    fft_data = fftpack.fft(data)
    fft_data = np.absolute(fft_data)
    
    half = int((len(fft_data) + 1) / 2)
    total = 0
    for i in range(half):
        total += fft_data[i] ** 2
    
    val = total
    
    return outlier_check(val)


def entropy(data):
    '''
    Calculate within the frequency domain. Returns the impurity within the data.
    '''
    if not isinstance(data, np.ndarray):
        print('entropy:', 'input is not numpy array')
        return np.nan

    X = np.abs(fftpack.fft(data))
    
    psd = X**2 / X.size

    div = np.sum(psd)
    if div == 0.0:
        div = MIN_VAL
    psd = psd / div # Normalize psd
    psd[np.where(psd==0.0)] = MIN_VAL
    ln_psd = np.log(psd)

    val = np.sum(psd * ln_psd) * - 1

    return outlier_check(val)


def dom_freq_ratio(data):
    '''
    Calculated within the frequency domain. Returns the ratio between the 
    largest FFT coefficient and all FFT coefficients.
    '''
    if not isinstance(data, np.ndarray):
        print('dom_freq_ratio:', 'input is not numpy array')
        return np.nan

    fft_data = fftpack.fft(data)
    fft_data = np.abs(fft_data)
    div = np.sum(fft_data)
    if div == 0.0:
        div = MIN_VAL
    
    val = np.max(fft_data) / div
    
    return outlier_check(val)

# Frequency domain feature ends
