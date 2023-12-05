import numpy as np
from run_sqi import QRSComparison

def bsqi(ann1, ann2, HRVparams=None):
    if HRVparams is None:
        HRVparams = initialize_HRVparams()

    windowlength = HRVparams['sqi']['windowlength']
    threshold = HRVparams['sqi']['TimeThreshold']
    margin = HRVparams['sqi']['margin']
    fs = HRVparams['Fs']

    ann1 = np.array(ann1) / fs
    ann2 = np.array(ann2) / fs

    # Fictitious start indices for simplicity (replace with your logic if needed)
    StartIdxSQIwindows = np.arange(0, len(ann1), windowlength)

    F1 = np.full(len(StartIdxSQIwindows), np.nan)

    for seg in range(len(StartIdxSQIwindows)):
        if not np.isnan(StartIdxSQIwindows[seg]):
            idx_ann1_in_win = np.where(
                (ann1 >= StartIdxSQIwindows[seg]) & (ann1 < StartIdxSQIwindows[seg] + windowlength)
            )[0]

            a1 = ann1[idx_ann1_in_win] - StartIdxSQIwindows[seg]
            a2 = ann2 - StartIdxSQIwindows[seg]

            F1_score, _, _, _ = run_sqi(a1, a2, threshold, margin, windowlength, fs)
            F1[seg] = F1_score

    return F1, StartIdxSQIwindows


# Placeholder for run_sqi function
def run_sqi(a1, a2, threshold, margin, windowlength, fs):
    qrs_comp = QRSComparison(a1, a2, thres=threshold, margin=margin, windowlen=windowlength, fs=fs)
    return qrs_comp.run_sqi()


def bsqi_example():
    # Generate example data
    fs = 1000  # Sample rate in Hz (e.g., 1000 Hz for ECG data)
    duration = 60  # Duration in seconds (e.g., 60 seconds of ECG data)
    
    # Simulate QRS peaks occurring roughly every 1 second
    # ann1 is the reference, and ann2 is slightly offset
    ann1 = np.arange(1, duration, 1) * fs  # Every second, in sample indices
    ann2 = (np.arange(1, duration, 1) + 0.1) * fs  # Offset by 0.1 second

    # Initialize_HRVparams
    HRVparams = {
        'sqi': {
            'windowlength': 10,  # Analysis window length in seconds
            'TimeThreshold': 0.15,  # Threshold in seconds
            'margin': 1  # Margin in seconds
        },
        'Fs': fs
    }

    # Call bsqi function
    F1, StartIdxSQIwindows = bsqi(ann1, ann2, HRVparams)

    # Print the results
    print("F1:", F1)
    print("StartIdxSQIwindows:", StartIdxSQIwindows)

if __name__ == '__main__':
    # Run the example
    bsqi_example()
