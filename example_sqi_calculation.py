import wfdb
import numpy as np
from wqrsm import wqrsm
from wqrsm_fast import wqrsm_fast
from run_qrsdet_by_seg import run_qrsdet_by_seg
from run_sqi import QRSComparison
import bsqi

# Function to initialize HRV parameters
def initialize_HRVparams():
    return {
        'sqi': {
            'windowlength': 5,
            'TimeThreshold': 0.1,
            'margin': 0.05
        },
        'Fs': 1000,
        'PeakDetect': {
            'REF_PERIOD': 0.250,
            'THRES': 0.6,
            'fid_vec': [],
            'SIGN_FORCE': [],
            'debug': True,
            'windows': 15,
            'ecgType': 'MECG'
        },
    }

# Function to read ECG data from a .dat file
def read_ecg_dat_file(file_path):
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal[:, 0]
    return signal, record.fs

# Load ECG data
file_path = "101"
ecg_data, fs = read_ecg_dat_file(file_path)

# Run QRS detection using wqrsm
qrs_wqrsm, _ = wqrsm(ecg_data, fs=fs)

# # Run QRS detection using wqrsm_fast
# qrs_wqrsm_fast, _ = wqrsm_fast(ecg_data, fs=fs)

# Initialize HRV parameters
HRVparams = initialize_HRVparams()


# Run QRS detection using jqrs algorithm
qrs_jqrs = run_qrsdet_by_seg(ecg_data, HRVparams)

# # Compute SQI using QRSComparison class
# sqi_calc = QRSComparison(qrs_wqrsm, qrs_jqrs, fs=fs)
# F1, Se, PPV, Nb = sqi_calc.run_sqi()

# # Print SQI results
# print("F1 score:", F1)
# print("Sensitivity (Se):", Se)
# print("Positive Predictive Value (PPV):", PPV)
# print("Counts:", Nb)

# Compute SQI using bsqi function
F1_bsqi, StartIdxSQIwindows = bsqi.bsqi(qrs_wqrsm, qrs_jqrs, HRVparams)

# Print bsqi results
print("F1 score (bsqi):", F1_bsqi)
print("Start Index SQI windows:", StartIdxSQIwindows)
