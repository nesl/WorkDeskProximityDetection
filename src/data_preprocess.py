# Standard library imports.
from datetime import timedelta, datetime
import pprint
import pickle
import json

# Related third party imports.
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# Local application/library specific imports.
from cerebralcortex.cerebralcortex import CerebralCortex
import utils

def data_filter(input_array: np.ndarray, start_hr: int, end_hr: int):
    'Keep data rows which are between start_hr and end_hr'
    output_array = list()
    for row in input_array:
        ts = row[0]
        offset = row[2]
        dt = datetime.utcfromtimestamp(ts) + timedelta(seconds=offset)
        if dt.hour >= start_hr and dt.hour < end_hr:
            output_array.append(row)
    return np.vstack(output_array)


def resample(x, ts_new):
    'Resamples x with freq from start_t to end_t'
    
    f_new = interp1d(x[:, 0], x[:, 1:], kind='zero', axis=0)
    
    x_new = np.hstack((ts_new.reshape((-1, 1)), f_new(ts_new)))
    return x_new

# Set important paths
CONFIG_PATH = '../config/'
DATA_PATH = '../data/'

# Load all user IDs
with open(CONFIG_PATH+'users.json', 'r') as f:
    USR_IDS = json.load(f)

# Load all users' work days
with open(DATA_PATH+'usr_work_days.pkl', 'rb') as f:
    USR_WORK_DAYS = pickle.load(f)
    
# Load all users' groundthruths
with open(DATA_PATH+'at_desk_groundtruth.pkl', 'rb') as f:
    AT_DESK_TIMES = pickle.load(f)


for usr_id in USR_WORK_DAYS:
    usr_path = DATA_PATH+usr_id+'/'
    for day in USR_WORK_DAYS[usr_id]:

        accel = np.load(usr_path+'accel'+day+'.npz')['arr_0']
        act_type = np.load(usr_path+'act_type'+day+'.npz')['arr_0']
        gyro = np.load(usr_path+'gyro'+day+'.npz')['arr_0']
        step_cnt = np.load(usr_path+'step_cnt'+day+'.npz')['arr_0']

        # Find the latest start time among all sensors
        start_t = max([accel[0][0], act_type[0][0], 
                       gyro[0][0], step_cnt[0][0]])
        
        # Find the earliest end time among all sensors
        end_t = min([accel[-1][0], act_type[-1][0], 
                     gyro[-1][0], step_cnt[-1][0]])
        
        # Generate new timestamps
        ts_new = np.arange(start_t, end_t, 1.0/utils.INTERP_FREQ)
 
        # Prepare labels for each sensor data point
        df = pd.DataFrame({'val' : np.zeros(ts_new.shape)},
                          index=pd.to_datetime(ts_new, unit='s'))
  
        for start_t, end_t in AT_DESK_TIMES[usr_id]:
            df[start_t : end_t] = 1
        
        
        accel = data_filter(resample(accel, ts_new), 8, 20)
        offset = accel[:, 2]
        act_type = data_filter(resample(act_type, ts_new), 8, 20)
        gyro = data_filter(resample(gyro, ts_new), 8, 20)
        step_cnt  = data_filter(resample(step_cnt, ts_new), 8, 20)
        
        
        ts_new = accel[:, 0]
        # Prepare labels for each sensor data point
        df = pd.DataFrame({'val' : np.zeros(ts_new.shape)},
                          index=pd.to_datetime(ts_new, unit='s'))
        for start_t, end_t in AT_DESK_TIMES[usr_id]:
            df[start_t : end_t] = 1
        labels = np.asarray(df.values).reshape(-1)
        
        assert(ts_new.shape[0] == accel.shape[0])
        assert(ts_new.shape[0] == act_type.shape[0])
        assert(ts_new.shape[0] == gyro.shape[0])
        assert(ts_new.shape[0] == step_cnt.shape[0])
        assert(ts_new.shape[0] == labels.shape[0])

        output_fn = usr_path + 'data' + day + '.npz'
        print('Saving '+output_fn+'...')

        np.savez(output_fn,
                 ts=ts_new,
                 offset=offset,
                 accel=accel[:, 3:],
                 act_type=act_type[:, 3:],
                 gyro=gyro[:, 3:],
                 step_cnt=step_cnt[:, 3:],
                 labels=labels)
