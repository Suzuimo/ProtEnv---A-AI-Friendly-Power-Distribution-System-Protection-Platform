import numpy as np
import pandas as pd
from datetime import datetime


# convert Cartesian to Polar for complex numbers
def cart_to_pol(arr):
    # the format of input array is:
    # [Re1, Im1, Re2, Im2, ...]
    dim = int(len(arr) / 2)
    mag = np.zeros(dim)
    angle = np.zeros(dim)
    for i in range(dim):
        re = arr[i*2]
        im = arr[i*2 + 1]
        cplx = re + 1j*im
        mag[i] = np.abs(cplx)
        angle[i] = np.angle(cplx)

    return mag, angle

## Get the number of a string, ignoring other characters
def getNum(string):
    length = len(string)
    i = 0
    while i < length:
        if (ord(string[i]) == 46) or (ord(string[i]) >= 48 and ord(string[i]) <= 57):
            pass
        else:
            string = string.replace(string[i],'')
            length = len(string)
            i = i - 1
        i = i + 1
    if length > 2:
        if ord(string[length-1]) == 46:
            string = string[0:(length-1)]
    try:
        num = float(string)
        return num
    # return None if conversion fails
    except Exception:
        return None

# normalize a dataframe
def normalize(df):
    result = df.copy()
    min_val = df.min()
    max_val = df.max()
    result = (df - min_val) / (max_val - min_val)

    return result


# fault class for DSS
class fault():
    def __init__(self, buses, phases, ts):
        self.bus = self.rand_bus(buses, phases)
        self.phases = self.rand_phase(buses, phases)
        self.R = self.rand_resistance()
        self.T = self.rand_time(ts)
        self.cmd = self.get_cmd_string()

    # location of fault        
    def rand_bus(self, buses, phases):
        # return a random bus in the system
        self.bus_idx = np.random.choice(range(len(buses)))
        while not (len(phases[self.bus_idx])==1 or len(phases[self.bus_idx])==3):
            self.bus_idx = np.random.choice(range(len(buses)))
        return buses[self.bus_idx]

    # return a fault type
    def rand_phase(self, buses, phases):
        p = phases[self.bus_idx]

        # if 1p line, only SLG  possible 
        if len(p) == 1:
            self.type = '1'
            return str(p[0])
        
        # if 3p line, can have all kinds of fault
        elif len(p) == 3:
            self.type = np.random.choice(['1','2','3'])
            if self.type == '1':
                return np.random.choice(['1','2','3'])
            elif self.type == '2' or self.type == '2g':
                return np.random.choice(['1','2','3'], 2, replace=False)
            else:
                return ['1','2','3']

        
    def rand_resistance(self):
        # corresponding to low, med, high res fault
        fault_r_range = [[0.002,0.01],[0.01, 0.1],[0.1,1],[1,15]]
        fault_r = fault_r_range[np.random.choice([0,1,2,3])]
        R = np.random.uniform(fault_r[0],fault_r[1])
        R = round(R, 4)
        return R

    def rand_time(self, ts):
        return np.floor(np.random.uniform(15, 30)) * ts

    # generate DSS command string from randomized attributes
    def get_cmd_string(self):
        cmd = 'New Fault.F1 '
        # number of phases
        cmd += 'Phases=' + str(len(self.phases))
        # format the faulted lines to the input form
        if self.type == '1':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0]
        elif self.type == '2':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.0'
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '2g':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.' + self.phases[0]
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '3':
            cmd += ' Bus1=' + self.bus + '.1.2.3'
        # fault resistance
        cmd += ' R=' + str(self.R)
        # fault time
        cmd += ' ONtime=' + str(self.T)

        return cmd


# load pv and load profile
def parse_profile(pv_path, load_path):
    date_range = pd.date_range(datetime(2019, 1, 1), datetime(2019, 12, 31))
    
    df_pv = pd.read_csv(pv_path, index_col='date', parse_dates=True)
    df_pv = df_pv.loc[df_pv.index.isin(date_range)]
    df_pv = df_pv.set_index([df_pv.index, 'fuel'])
    df_pv = pd.DataFrame({'value': df_pv.stack()}).reset_index(1)
    df_pv.index = pd.to_datetime(
        df_pv.index.get_level_values(0).strftime('%Y-%m-%d ') + df_pv.index.get_level_values(1))
    df_pv.index.name = 'date_time'

    df_load = pd.read_csv(load_path, index_col='date', parse_dates=True)
    df_load = df_load.loc[df_load.index.isin(date_range)]
    df_load = pd.DataFrame({'load': df_load.stack()})
    df_load.index = pd.to_datetime(
        df_load.index.get_level_values(0).strftime('%Y-%m-%d ') + df_load.index.get_level_values(1))
    df_load.index.name = 'date_time'

    pv_profile = df_pv[df_pv['fuel']=='solar']['value']
    load_profile = df_load['load']

    pv_profile = normalize(pv_profile)
    load_profile = normalize(load_profile)

    return pv_profile, load_profile

