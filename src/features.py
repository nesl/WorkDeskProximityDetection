import numpy as np
import scipy.stats as st
from scipy.fftpack import fft

max_val=1e6
min_val=1e-6


##Features used from UCI dataset 
def mean(data):
    '''
    Returns the mean value of the data. 
    '''
    if not isinstance(data, np.ndarray):
        return None
    val = np.mean(data)
#    if val>max_val:
#        val=max_val
#    if val<min_val:
#        val=min_val
    return val


def mad(data):
    '''
    Returns the median absolute deviation (MAD)
    '''
    if not isinstance(data, np.ndarray):
        return None
    
    val = np.median(np.abs(data - np.median(data)))
#    if val>max_val:
#        val=max_val
#    if val<min_val:
#        val=min_val
    
    return val
   
 
def rawenergy(data):
    '''
    Returns the total energy of the data .
    '''
    if not isinstance(data, np.ndarray):
        return None
    
    total = 0
    for i in range(len(data)):
        total += data[i] ** 2
        
    val=total / len(data)
#    if val>max_val:
#        val=max_val
#    if val<min_val:
#        val=min_val

    return val


def meanFreq(data):
    if not isinstance(data, np.ndarray):
        return None

    fft_data = fft(data)
    fft_data = fft_data.real
    val=np.mean(fft_data)
#    if val>max_val:
#        val=max_val
#    if val<min_val:
#        val=min_val
    
    return val


def bandsenergy(data):
    '''
    Calculated within the frequency domain.
    Returns the total energy of the data in all frequencies.
    '''
    if not isinstance(data, np.ndarray):
        return None

    fft_data = fft(data)
    n = 64
    timestep = 0.01
    freq = np.fft.fftfreq(n, d=timestep)
    energy1to8 = sum(freq[0:8]**2)
    energy9to16 = sum(freq[8:16]**2)
    energy17to24 = sum(freq[16:24]**2)
    energy25to32 = sum(freq[25:32]**2)
    energy33to40 = sum(freq[32:40]**2)
    energy41to48 = sum(freq[40:48]**2)
    energy49to56 = sum(freq[48:56]**2)
    energy57to64 = sum(freq[57:64]**2)
    energy1to16 = sum(freq[0:16]**2)
    energy17to32 = sum(freq[16:32]**2)
    energy33to48 = sum(freq[32:48]**2)
    energy49to64 = sum(freq[48:64]**2)
    energy1to24 = sum(freq[0:24]**2)
    energy25to48 = sum(freq[24:48]**2)

    return [energy1to8, energy9to16, energy17to24, energy25to32, energy33to40, 
            energy41to48, energy49to56, energy57to64, energy1to16, energy17to32,
            energy33to48, energy49to64, energy1to24, energy25to48]



##############

def mini(data):
    '''
    Returns the minimum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val = np.min(data)
    
#    if val>max_val:
#        val=max_val
#    if val<min_val:
#        val=min_val
    
    return val


def maxi(data):
    '''
    Returns the maximum value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val = np.max(data)
   # if val>max_val:
   #     val=max_val
   # if val<min_val:
   #     val=min_val
    
    return val


def median(data):
    '''
    Returns the median value of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val = np.median(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val

def var(data):
    '''
    Returns the variance of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val=np.var(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def std(data):
    '''Returns the standard deviation of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val=np.std(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def ran(data):
    '''
    Returns the range of the data.
    '''    
    if not isinstance(data, np.ndarray):
        return None

    val=np.max(data) - np.min(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def absmean(data):
    '''
    Returns the average of the absolute values of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    data = np.absolute(data)
    
    if(len(data))==0:
        return 0
    
    val=np.sum(data) / len(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def cv(data):
    '''
    Returns the ratio between standard deviation and the mean times 100. 
    Measures signal dispersion.
    '''
    if not isinstance(data, np.ndarray):
        return None

    div=np.mean(data)
    
    if div==0:
        return 0
    
    val=np.sum(data) / (div * 100)

    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def skew(data):
    '''
    Returns the skewness (3rd moment) of the data. 
    Measures asymmetry of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val=st.skew(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def kurtosis(data):
    '''
    Returns the kurtosis (4th moment) of the data. 
    Measures peakedness of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val=st.kurtosis(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val

    
def quartile1(data):
    '''Returns the first quartile of the data.
    ''' 
    if not isinstance(data, np.ndarray):
        return None

    val= np.percentile(data, 25)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val
    
   

def quartile3(data):
    '''
    Returns the third quartile of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val= np.percentile(data, 75)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val

def iqr(data):
    '''
    Returns the difference between the 3rd and 1st quartile, also known as the 
    inter quartile range. Measures dispersion.
    '''    
    if not isinstance(data, np.ndarray):
        return None

    val= quartile3(data) - quartile1(data)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def mcr(data):
    '''
    Returns the number of times the data crosses the mean value.
    Measures how often the signal varies.
    '''
    if not isinstance(data, np.ndarray):
        return None

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


def absarea(data):
    '''
    Returns the absolute area, or the absolute sum of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    total = 0
    for i in data:
        total += np.abs(i)
        
    val=total
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val
    
    

def totalabsarea(datax, datay, dataz):
    '''
    Returns the absolute area across all three axes.
    '''
    if not isinstance(datax, np.ndarray):
        return None
    if not isinstance(datay, np.ndarray):
        return None
    if not isinstance(dataz, np.ndarray):
        return None


    return absarea(datax) + absarea(datay) + absarea(dataz)


def totalsvm(datax, datay, dataz):
    '''
    Returns the signal magnitude across all three axes.
    '''
    if not isinstance(datax, np.ndarray):
        return None
    if not isinstance(datay, np.ndarray):
        return None
    if not isinstance(dataz, np.ndarray):
        return None

    total = 0
    for i in range(len(datax)):
        total += np.sqrt(datax[i] ** 2 + datay[i] ** 2 + dataz[i] ** 2)
    return float(total) / len(datax)


def energy(data):
    '''
    Calculated within the frequency domain. 
    Returns the total energy of the data in all frequencies.
    '''
    if not isinstance(data, np.ndarray):
        return None

    fft_data = fft(data)
    fft_data = np.absolute(fft_data)
    
    half = int((len(fft_data) + 1) / 2)
    total = 0
    for i in range(half):
        total += fft_data[i] ** 2
    
    val=total
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def entropy(data):
    '''
    Calculate within the frequency domain. Returns the impurity within the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    fft_data = fft(data)

    psd = []
    for i in fft_data:
        i = np.absolute(i)
        psd.append(i ** 2 / len(fft_data))
    psd_tot = sum(psd)
    pdf = []
    for i in psd:
        pdf.append(i / psd_tot)
    total = 0
    for i in pdf:
        i = i + min_val
        total += i * np.log(i)
    
    val=-total
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def domfreqratio(data):
    '''
    Calculated within the frequency domain. Returns the ratio between the 
    largest FFT coefficient and all FFT coefficients.
    '''
    if not isinstance(data, np.ndarray):
        return None

    fft_data = fft(data)
    fft_data = np.absolute(fft_data)
    div = sum(fft_data)
    if div == 0:
        return 0
    
    val=max(fft_data) / div
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val


def percentile(data, perc):
    '''
    Returns the given percentile of the data.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val = np.percentile(data, perc)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val

def rms(data):
    '''
    Returns the root mean square of the data.
    '''

    if not isinstance(data, np.ndarray):
        return None

    total = 0
    for i in data:
        total += i ** 2
    
    val=np.sqrt(total / len(data))
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val
    


def slope(data):
    '''
    Returns the slope between the first and last point.
    '''
    if not isinstance(data, np.ndarray):
        return None

    val=float(data[len(data) - 1] - data[0]) / (len(data) - 1)
    
    #if val>max_val:
    #    val=max_val
    #if val<min_val:
    #    val=min_val
    
    return val
