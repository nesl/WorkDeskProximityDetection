# Standard library imports.
from datetime import timedelta, datetime
import pickle
import json

# Related third party imports.
import numpy as np
import pandas as pd

# Local application/library specific imports.
import features
import utils


# Set important paths
CONFIG_PATH = '../config/'
DATA_PATH = '../data/'

WIN_SIZE = 10 # 10s
OVERLAP = 0 # Zero overlapping between sliding windows

def feature_vector(win, freq_feat=True):
    vec = list()
    # Time domain features
    vec.append(features.mean(win))
    vec.append(features.mad(win))       
    vec.append(features.mini(win))
    vec.append(features.maxi(win))
    vec.append(features.median(win))
    vec.append(features.var(win))
    vec.append(features.std(win))
    vec.append(features.ran(win))
    vec.append(features.abs_mean(win))
    vec.append(features.coeff_var(win))
    vec.append(features.skewness(win))
    vec.append(features.kurtosis(win))
    vec.append(features.quartile1(win))
    vec.append(features.quartile3(win))
    vec.append(features.iqr(win))
    vec.append(features.mcr(win))
    vec.append(features.rms(win))
    vec.append(features.slope(win))
    vec.append(features.integral(win))
    # frequency domain features
    if freq_feat:
        vec.append(features.dc_component(win, utils.INTERP_FREQ))
        vec.append(features.energy(win))
        vec.append(features.entropy(win))
        vec.append(features.dom_freq_ratio(win))
    
    return vec


def time_features(ts, offset, sample_freq, t_win, overlap):
    win_len = sample_freq * t_win
    overlap_len = sample_freq * overlap * t_win
    ts_wins = utils.generate_wins(ts, win_len, overlap_len)
    offset_wins = utils.generate_wins(ts, win_len, overlap_len)
    ts_features = list()
    for ts_win, offset_win in zip(ts_wins, offset_wins):
        ts_features.append(features.is_weekday(ts_win[0], offset_win[0]))
    return np.array(ts_features)


def accel_features(accel, sample_freq, t_win, overlap):
    accel_x = accel[:, 0]
    accel_y = accel[:, 1]
    accel_z = accel[:, 2]

    win_len = sample_freq * t_win
    overlap_len = sample_freq * overlap * t_win
    accel_x_wins = utils.generate_wins(accel_x, win_len, overlap_len)
    accel_y_wins = utils.generate_wins(accel_y, win_len, overlap_len)
    accel_z_wins = utils.generate_wins(accel_z, win_len, overlap_len)
    accel_features = list() 
    for x_win, y_win, z_win in zip(accel_x_wins, accel_y_wins, accel_z_wins):
        feature_vec = feature_vector(x_win) + feature_vector(y_win) + feature_vector(z_win)
        feature_vec.append(features.signal_vec_mag(x_win, y_win, z_win))
        feature_vec.append(features.signal_mag_area(x_win, y_win, z_win))

        accel_features.append(feature_vec)
    return np.array(accel_features)


def gyro_features(gyro, sample_freq, t_win, overlap):
    gyro_x = gyro[:, 0]
    gyro_y = gyro[:, 1]
    gyro_z = gyro[:, 2]

    win_len = sample_freq * t_win
    overlap_len = sample_freq * overlap * t_win
    gyro_x_wins = utils.generate_wins(gyro_x, win_len, overlap_len)
    gyro_y_wins = utils.generate_wins(gyro_y, win_len, overlap_len)
    gyro_z_wins = utils.generate_wins(gyro_z, win_len, overlap_len)
    gyro_features = list() 
    for x_win, y_win, z_win in zip(gyro_x_wins, gyro_y_wins, gyro_z_wins):
        feature_vec = feature_vector(x_win) + feature_vector(y_win) + feature_vector(z_win)
        gyro_features.append(feature_vec)
    return np.array(gyro_features)



def act_type_features(act_type, sample_freq, t_win, overlap):
    win_len = sample_freq * t_win
    overlap_len = sample_freq * overlap * t_win
    act_type_wins = utils.generate_wins(act_type, win_len, overlap_len)
    act_features = list()
    for win in act_type_wins:
        act_features.append(features.act_type_one_hot(win))
    return np.array(act_features)


def step_cnt_features(step_cnt, sample_freq, t_win, overlap):

    win_len = sample_freq * t_win
    overlap_len = sample_freq * overlap * t_win
    step_cnt_wins = utils.generate_wins(step_cnt[:,0], win_len, overlap_len)
    step_cnt_features = list() 
    for win in step_cnt_wins:
        feature_vec = feature_vector(win, False)
        step_cnt_features.append(feature_vec)
    return np.array(step_cnt_features)


with open(CONFIG_PATH+'users.json', 'r') as f:
    USR_IDS = json.load(f)

# Load all users' work days
with open(DATA_PATH+'usr_work_days.pkl', 'rb') as f:
    USR_WORK_DAYS = pickle.load(f)
    
# Load all users' groundtruths
with open(DATA_PATH+'at_desk_groundtruth.pkl', 'rb') as f:
    AT_DESK_TIMES = pickle.load(f)


data_path = '/home/mperf/sandeep/Codes/data/03996723-2411-4167-b14b-eb11dfc33124/'

with open(data_path+'data20180102.npz', 'rb') as f:
    data = np.load(f)

    print(time_features(data['ts'], data['offset'], 20, WIN_SIZE, OVERLAP))
    print(accel_features(data['accel'], 20, WIN_SIZE, OVERLAP))
    print(gyro_features(data['gyro'], 20, WIN_SIZE, OVERLAP))
    
    print(step_cnt_features(data['step_cnt'], 20, WIN_SIZE, OVERLAP))
# for usr_id in USR_WORK_DAYS:
#     usr_path = DATA_PATH+usr_id+'/'
#     for day in USR_WORK_DAYS[usr_id]:
#         data_fn = usr_path+'data'+day+'.npz'
#         data = np.load(data_fn)
#         ts = data['ts']
#         offset = data['offset']
#         accel = data['accel']
#         gyro = data['gyro']
#         set_cnt = data['step_cnt']
#         act_type = data['act_type']
