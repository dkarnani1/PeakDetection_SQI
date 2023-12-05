# #import the WFDB package
import wfdb
import numpy as np
from wqrsm import wqrsm
from wqrsm_fast import wqrsm_fast
# load a record using the 'rdrecord' function
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

# Now, ecg_data is a NumPy array containing the ECG signal
# print(ecg_data)

#wqrsm

# qrs, jpoints = wqrsm(ecg_data[1:3600, 0:1])

# #[:, 1:2]
# print("QRS locations:", qrs)
# print("J-point locations:", jpoints)


#wqrsm_fast

qrs, jpoints = wqrsm_fast(ecg_data[1:3600, 0:1])
print("QRS:", qrs)
print("J-points:", jpoints)







