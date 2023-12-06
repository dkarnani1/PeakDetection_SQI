# #import the WFDB package
import wfdb
import numpy as np
from jqrs import jqrs
import matplotlib.pyplot as plt




def read_ecg_dat_file(file_path):
    # Read the ECG data file
    record = wfdb.rdrecord(file_path)
    
    # Extract the signal values
    signal = record.p_signal
    
    return signal

# Example usage
file_path = '101'
ecg_data = read_ecg_dat_file(file_path)


HRVparams = {
    'PeakDetect': {
        'REF_PERIOD': 0.250,
        'THRES': 0.6,
        'fid_vec': [],
        'SIGN_FORCE': [],
        'debug': True,
        'windows': 15,
        'ecgType': 'MECG'
    },
    'Fs': 1000,  # Replace with the actual sampling frequency
}

# Call the jqrs function
qrs_pos, sign, en_thres = jqrs(ecg_data[1:3600, 0:1], HRVparams)

# Print the results
print("QRS Positions:", qrs_pos)
print("Sign:", sign)
print("Energy Threshold:", en_thres)