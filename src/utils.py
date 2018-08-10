'''
This module contains utility constatns and helper functions.
'''
from datetime import timedelta, datetime
from typing import List

import numpy as np
from cerebralcortex.core.datatypes.datastream import DataPoint


# Sensor sample range
MAX_ACCEL = 5.0
MIN_ACCEL = -5.0

MAX_GYRO = 5.0
MIN_GYRO = -5.0

MAX_LOC_LAT = 90.0
MIN_LOC_LAT = -90.0
MAX_LOC_LON = 180.0
MIN_LOC_LON = -180.0
MAX_LOC_ALT = 1000.0
MIN_LOC_ALT = 0
MAX_LOC_SPEED = 500.0
MIN_LOC_SPEED = 0
MAX_LOC_BEAR = 360.0
MIN_LOC_BEAR = 0
MAX_LOC_ACCUR = 100.0
MIN_LOC_ACCUR = 0

MAX_ACT_TYPE = 7
MIN_ACT_TYPE = 0
MAX_CONFIDENCE = 100.0
MIN_CONFIDENCE = 0

MAX_LIGHT_INTENSITY = 250.0
MIN_LIGHT_INTENSITY = 0

MAX_PROXIMITY = 10
MIN_PROXIMITY = 0

MAX_BATT_LEVEL = 100.0
MIN_BATT_LEVEL = 0
MAX_BATT_VOLT = 5000.0
MIN_BATT_VOLT = 0
MAX_BATT_TEMP = 100.0
MIN_BATT_TEMP = -50.0

MAX_BLE_DIS = 100.0
MIN_BLE_DIS = 0

MAX_BLE_RSSI = 100.0
MIN_BLE_RSSI = -100.0

MAX_BLE_TX = 100.0
MIN_BLE_TX = -100.0

MAX_STEP_CNT = 50.0
MIN_STEP_CNT = 0


def extract_matched_labels(labels: List, keywords: List)->List:
    """
    Extract sensor stream labels which contain all the keywords.
    """
    results = list()
    for label in labels:
        matched = True
        for keyword in keywords:
            matched = matched and (keyword in label)
        if matched:
            results.append(label)
    return results


def extract_all_data(CC, usr_id: str, stream_label: str)->List[DataPoint]:
    """
    Extract all sensory data with in a stream for a user.
    """
    data = list()
    usr_streams = CC.get_user_streams(usr_id)
    try:
        target_stream = usr_streams[stream_label]
    except KeyError:
        print(usr_id + " does not have stream " + stream_label)
        return data
    # Enumerate stream id in target each  stream
    for stream_id in target_stream['stream_ids']:
        stream_days = CC.get_stream_days(stream_id)
        # Get data for all stream days
        for i, stream_day in enumerate(stream_days):
            ds = CC.get_stream(stream_id, usr_id, stream_day)
            data.extend(ds.data)
    # Sort all datapoints in chronological order
    data.sort()
    return data



def fill_missing_values(datapoints: List[DataPoint], freq: float) -> List[DataPoint]:
    
    """
    Introperlate the datapoints based on assigned frequency.
    """
    
    if not datapoints:
        return datapoints
    if freq == 0.0:
        return datapoints
    
    # Convert frequency to time intveral in second.
    time_interval = 1.0/freq
    new_datapoints = list()
    start_t  = datapoints[0].start_time
    end_t = datapoints[-1].start_time
    
    # Create a new list of timestamps with adjacent timestamp separated by time_invertal.
    t = start_t
    new_ts = list()
    
    # Interpolate the data list
    while t <= end_t:
        new_ts.append(t)
        t += timedelta(seconds=time_interval)
    #print(start_t, end_t)
    #print('# of new dp:', len(new_ts))

    j = 0
    for i in range(len(new_ts)):
        if new_ts[i] >= datapoints[j].start_time:
            new_datapoints.append(DataPoint(new_ts[i], None, datapoints[j].offset, datapoints[j].sample))
            j += 1
        else:
            new_datapoints.append(DataPoint(new_ts[i], None, datapoints[j-1].offset, datapoints[j-1].sample))
    return new_datapoints


def dp_to_list(dp: DataPoint)-> List:
    """
    Convert DataPoint to a list.
    """
    result = list()
    if dp is None:
        return result
    if dp.start_time is not None:
        result.append(dp.start_time.timestamp())
    else:
        result.append(np.nan)
        
    if dp.end_time is not None:
        result.append(dp.end_time.timestamp())
    else:
        result.append(np.nan)
    result.append(dp.offset)
    result.extend(list(dp.sample))
    return result
    

def to_numpy_array(datapoints: List[DataPoint]):
    """
    Convert a list of DataPoints to numpy array.
    """
    data = [dp_to_list(dp) for dp in datapoints]
    return np.array(data)
