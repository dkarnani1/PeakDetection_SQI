
from jqrs import jqrs
import numpy as np

def run_qrsdet_by_seg(ecg, HRVparams):
    """
    Run the QRS detector for each non-overlapping window.

    Parameters:
    - ecg: ECG signal
    - HRVparams: Parameters for QRS detection (as a dictionary or object)

    Returns:
    - QRS: QRS locations in samples
    """

    # Extract parameters
    
    fs = HRVparams['Fs']
    window = HRVparams['PeakDetect']['windows']
    thres = HRVparams['PeakDetect']['THRES']
    ecgType = HRVparams['PeakDetect']['ecgType']

    # General parameters
    segsize_samp = int(window * fs)  # convert window to number of samples
    nb_segments = int(np.floor(len(ecg) / segsize_samp))
    QRS = []

    start = 0
    stop = segsize_samp
    sign_force = 0  # if we want to force the sign of the peak we are looking for

    try:
        for _ in range(nb_segments):
            # For each segment, perform QRS detection
            qrs_temp = []

            # Take +/-1 second around the selected subsegment except for the borders.
            if start == 0:
                # First subsegment
                dT_plus = fs
                dT_minus = 0
            elif stop == len(ecg):
                # Last subsegment
                dT_plus = 0
                dT_minus = fs
            else:
                # Any other subsegment
                dT_plus = fs
                dT_minus = fs

            # Lowering the threshold in case not enough beats are detected.
            # Also changed the refractory period to be different between mother and foetus.
            # Sign of peaks is determined by the sign on the first window and then is forced for the following windows.
            if ecgType == 'FECG':
                thres_trans = thres
                while len(qrs_temp) < 20 and thres_trans > 0.1:
                    qrs_temp, sign_force = jqrs(ecg[start - dT_minus:stop + dT_plus], HRVparams)
                    thres_trans -= 0.1
            else:
                qrs_temp, sign_force = jqrs(ecg[start - dT_minus:stop + dT_plus], HRVparams)

            new_qrs = np.array(qrs_temp) + (start - dT_minus)
            new_qrs = new_qrs[(new_qrs >= start) & (new_qrs <= stop)]

            if len(QRS) > 0 and len(new_qrs) > 0 and new_qrs[0] - QRS[-1] < 0.25 * fs:
                # This is needed to avoid double detection at the transition point between two windows
                new_qrs = new_qrs[1:]

            QRS.extend(new_qrs)

            start += segsize_samp
            stop += segsize_samp

    except Exception as e:
        print(e)
        QRS = [1000, 2000]

    return QRS




