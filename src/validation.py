'''
This module provides functions to do data validation check.
'''
import numbers
from typing import List
from cerebralcortex.core.datatypes.datastream import DataPoint
import utils

def is_valid_phone_accelerometer(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone accelerometer data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 3:
        return False
    val = datapoint.sample[0]
    valid_x = (isinstance(val, numbers.Real)
               and val >= utils.MIN_ACCEL
               and val <= utils.MAX_ACCEL)
    val = datapoint.sample[1]
    valid_y = (isinstance(val, numbers.Real)
               and val >= utils.MIN_ACCEL
               and val <= utils.MAX_ACCEL)
    val = datapoint.sample[2]
    valid_z = (isinstance(val, numbers.Real)
               and val >= utils.MIN_ACCEL
               and val <= utils.MAX_ACCEL)

    return valid_x and valid_y and valid_z


def is_valid_phone_gyroscope(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone accelerometer data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 3:
        return False

    val = datapoint.sample[0]
    valid_x = (isinstance(val, numbers.Real)
               and val >= utils.MIN_GYRO
               and val <= utils.MAX_GYRO)
    val = datapoint.sample[1]
    valid_y = (isinstance(val, numbers.Real)
               and val >= utils.MIN_GYRO
               and val <= utils.MAX_GYRO)
    val = datapoint.sample[2]
    valid_z = (isinstance(val, numbers.Real)
               and val >= utils.MIN_GYRO
               and val <= utils.MAX_GYRO)

    return valid_x and valid_y and valid_z


def is_valid_phone_location(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone location data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 6:
        return False

    val = datapoint.sample[0]
    valid_lat = (isinstance(val, numbers.Real)
                 and val >= utils.MIN_LOC_LAT
                 and val <= utils.MAX_LOC_LAT)
    val = datapoint.sample[1]
    valid_lon = (isinstance(val, numbers.Real)
                 and val >= utils.MIN_LOC_LON
                 and val <= utils.MAX_LOC_LON)
    val = datapoint.sample[2]
    valid_alt = (isinstance(val, numbers.Real)
                 and val >= utils.MIN_LOC_ALT
                 and val <= utils.MAX_LOC_ALT)
    val = datapoint.sample[3]
    valid_speed = (isinstance(val, numbers.Real)
                   and val >= utils.MIN_LOC_SPEED
                   and val <= utils.MAX_LOC_SPEED)
    val = datapoint.sample[4]
    valid_bearing = (isinstance(val, numbers.Real)
                     and val >= utils.MIN_LOC_BEAR
                     and val <= utils.MAX_LOC_BEAR)
    val = datapoint.sample[5]
    valid_accuracy = (isinstance(val, numbers.Real)
                      and val >= utils.MIN_LOC_ACCUR
                      and val <= utils.MAX_LOC_ACCUR)

    return (valid_lat
            and valid_lon
            and valid_alt
            and valid_speed
            and valid_bearing
            and valid_accuracy)

def is_valid_phone_activity_type(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone activity type data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 2:
        return False
    val = datapoint.sample[0]
    valid_type = (isinstance(val, numbers.Real)
                  and val >= utils.MIN_ACT_TYPE
                  and val <= utils.MAX_ACT_TYPE)
    val = datapoint.sample[1]
    valid_confidence = (isinstance(val, numbers.Real)
                        and val >= utils.MIN_CONFIDENCE
                        and val <= utils.MAX_CONFIDENCE)

    return valid_type and valid_confidence

def is_valid_phone_ambient_light(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone ambient light data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 1:
        return False

    val = datapoint.sample[0]
    return (isinstance(val, numbers.Real)
            and val >= utils.MIN_LIGHT_INTENSITY
            and val <= utils.MAX_LIGHT_INTENSITY)

def is_valid_phone_proximity(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone proximity data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 1:
        return False

    val = datapoint.sample[0]
    return (isinstance(val, numbers.Real)
            and val >= utils.MIN_PROXIMITY
            and val <= utils.MAX_PROXIMITY)

def is_valid_phone_battery(datapoint: DataPoint):
    '''
    Return True if input data point is a valid phone battery data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 3:
        return False
    val = datapoint.sample[0]
    valid_level = (isinstance(val, numbers.Real)
                   and val >= utils.MIN_BATT_LEVEL
                   and val <= utils.MAX_BATT_LEVEL)
    val = datapoint.sample[1]
    valid_volt = (isinstance(val, numbers.Real)
                  and val >= utils.MIN_BATT_VOLT
                  and val <= utils.MAX_BATT_VOLT)
    val = datapoint.sample[2]
    valid_temp = (isinstance(val, numbers.Real)
                  and val >= utils.MIN_BATT_TEMP
                  and val <= utils.MAX_BATT_TEMP)

    return valid_level and valid_volt and valid_temp

def is_valid_beacon(datapoint: DataPoint):
    '''
    Return True if input data point is a valid beacon data point.
    Otherwise return False.
    '''
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 3:
        return False

    val = datapoint.sample[0]
    valid_dis = (isinstance(val, numbers.Real)
                 and val >= utils.MIN_BLE_DIS
                 and val <= utils.MAX_BLE_DIS)
    val = datapoint.sample[1]
    valid_rssi = (isinstance(val, numbers.Real)
                  and val >= utils.MIN_BLE_RSSI
                  and val <= utils.MAX_BLE_RSSI)
    val = datapoint.sample[2]
    valid_tx = (isinstance(val, numbers.Real)
                and val >= utils.MIN_BLE_TX
                and val <= utils.MAX_BLE_TX)

    return valid_dis and valid_rssi and valid_tx

def is_valid_step_count(datapoint: DataPoint):
    if not isinstance(datapoint.sample, List) or len(datapoint.sample) != 1:
        return False
    
    val = datapoint.sample[0]
    valid_cnt = (isinstance(val, numbers.Real)
                 and val >= utils.MIN_STEP_CNT
                 and val <= utils.MAX_STEP_CNT)
    return valid_cnt


def validate_location(loc_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate phone location data stream
    :param loc_data:
    :return: valid_loc_data
    '''
    valid_loc_data = []
    for datapoint in loc_data:
        if is_valid_phone_location(datapoint):
            valid_loc_data.append(datapoint)

    return valid_loc_data

def validate_activity_type(act_type_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate phone activity type data stream
    :param act_type_data:
    :return: valid_act_type_data
    '''
    if act_type_data is None:
        return None
    valid_act_type_data = []
    for datapoint in act_type_data:
        if is_valid_phone_activity_type(datapoint):
            valid_act_type_data.append(datapoint)

    return valid_act_type_data

def validate_ambient_light(light_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate phone ambient light data stream
    :param light_data:
    :return: valid_light_data
    '''
    valid_light_data = []
    for datapoint in light_data:
        if is_valid_phone_ambient_light(datapoint):
            valid_light_data.append(datapoint)

    return valid_light_data

def validate_proximity(proximity_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate phone proximity data stream
    :param light_data:
    :return: valid_proximity_data
    '''
    valid_proximity_data = []
    for datapoint in proximity_data:
        if is_valid_phone_proximity(datapoint):
            valid_proximity_data.append(datapoint)

    return valid_proximity_data

def validate_battery(batt_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate phone battery data stream
    :param batt_data:
    :return: valid_batt_data
    '''
    valid_batt_data = []
    for datapoint in batt_data:
        if is_valid_phone_battery(datapoint):
            valid_batt_data.append(datapoint)

    return valid_batt_data

def validate_beacon(ble_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate beacon data stream
    :param ble_data:
    :return: valid_ble_data
    '''
    valid_ble_data = []
    for datapoint in ble_data:
        if is_valid_beacon(datapoint):
            valid_ble_data.append(datapoint)

    return valid_ble_data

def validate_step_count(step_cnt_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate step count data stream
    :param step_cnt_data:
    :return: valid_step_cnt_data
    '''
    valid_step_cnt_data = []
    for datapoint in step_cnt_data:
        if is_valid_step_count(datapoint):
            valid_step_cnt_data.append(datapoint)

    return valid_step_cnt_data


def validate_accelerometer(accel_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate accelerometer data stream
    :param accel_data:
    :return: valid_accel_data
    '''
    valid_accel_data = []
    for datapoint in accel_data:
        if is_valid_phone_accelerometer(datapoint):
            valid_accel_data.append(datapoint)

    return valid_accel_data


def validate_gyroscope(gyro_data: List[DataPoint]) -> List[DataPoint]:
    '''
    validate gyroscope data stream
    :param gyro_data:
    :return: valid_gyro_data
    '''
    valid_gyro_data = []
    for datapoint in gyro_data:
        if is_valid_phone_gyroscope(datapoint):
            valid_gyro_data.append(datapoint)

    return valid_gyro_data


