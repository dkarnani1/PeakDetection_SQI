import numpy as np
from scipy.signal import filtfilt
from scipy.ndimage import median_filter
from scipy.signal import resample
from scipy.interpolate import interp1d

def jqrs(ecg, HRVparams):
    REF_PERIOD = HRVparams['PeakDetect']['REF_PERIOD']
    THRES = HRVparams['PeakDetect']['THRES']
    fs = HRVparams['Fs']
    fid_vec = HRVparams['PeakDetect']['fid_vec']
    SIGN_FORCE = HRVparams['PeakDetect']['SIGN_FORCE']
    debug = HRVparams['PeakDetect']['debug']
    ecg = np.squeeze(ecg)
    NB_SAMP = len(ecg)
    INT_NB_COEFF = round(7 * fs / 256)  # Assuming you want to keep the same definition as in the MATLAB code
    MED_SMOOTH_NB_COEFF = round(fs / 100)  # Assuming you want to keep the same definition as in the MATLAB code
    MAX_FORCE = None
    SEARCH_BACK = 1
    tm = np.arange(1/fs, (NB_SAMP/fs) + 1/fs, 1/fs)

    # Bandpass filtering for ECG signal
    b1 = np.array([-7.757327341237223e-05, -2.357742589814283e-04, -6.689305101192819e-04, -0.001770119249103,
                   -0.004364327211358, -0.010013251577232, -0.021344241245400, -0.042182820580118, -0.077080889653194,
                   -0.129740392318591, -0.200064921294891, -0.280328573340852, -0.352139052257134, -0.386867664739069,
                   -0.351974030208595, -0.223363323458050, 0, 0.286427448595213, 0.574058766243311,
                   0.788100265785590, 0.867325070584078, 0.788100265785590, 0.574058766243311, 0.286427448595213, 0,
                   -0.223363323458050, -0.351974030208595, -0.386867664739069, -0.352139052257134,
                   -0.280328573340852, -0.200064921294891, -0.129740392318591, -0.077080889653194, -0.042182820580118,
                   -0.021344241245400, -0.010013251577232, -0.004364327211358, -0.001770119249103,
                   -6.689305101192819e-04, -2.357742589814283e-04, -7.757327341237223e-05])

    #b1 = resample(b1, int(250 / fs * len(b1)))
    resampled_b1 = interp1d(np.linspace(0, 1, len(b1)), b1, kind='linear')(np.linspace(0, 1, int(250 / fs * len(b1))))
    b1 = resampled_b1
    bpfecg = filtfilt(b1, 1, ecg)


    MIN_AMP = 0.1  # Define MIN_AMP here

    if np.sum(np.abs(bpfecg) > MIN_AMP) / NB_SAMP > 0.20:
        # P&T operations
        dffecg = np.diff(bpfecg)
        sqrecg = dffecg * dffecg
        intecg = np.convolve(sqrecg, np.ones(INT_NB_COEFF) / INT_NB_COEFF, mode='valid')
        mdfint = median_filter(intecg, size=MED_SMOOTH_NB_COEFF)
        delay = int(np.ceil(INT_NB_COEFF / 2))
        mdfint = np.roll(mdfint, -delay)

        if fid_vec:
            mdfintFidel = mdfint
        else:
            mdfintFidel = np.zeros_like(mdfint)
            mdfintFidel[fid_vec > 2] = 0

        xs = np.sort(mdfintFidel[fs:fs*90] if NB_SAMP/fs > 90 else mdfintFidel[fs:])

        if not MAX_FORCE:
            ind_xs = int(np.ceil(98/100 * len(xs)))
            en_thres = xs[ind_xs]
        else:
            en_thres = MAX_FORCE

        poss_reg = mdfint > (THRES * en_thres)

        if not poss_reg.any():
            poss_reg[10] = 1

        if SEARCH_BACK:
            indAboveThreshold = np.where(poss_reg)[0]
            RRv = np.diff(tm[indAboveThreshold])
            medRRv = np.median(RRv[RRv > 0.01])
            indMissedBeat = np.where(RRv > 1.5 * medRRv)[0]
            indStart = indAboveThreshold[indMissedBeat]
            indEnd = indAboveThreshold[indMissedBeat + 1]

            for i in range(len(indStart)):
                poss_reg[indStart[i]:indEnd[i]] = mdfint[indStart[i]:indEnd[i]] > (0.5 * THRES * en_thres)

        left = np.where(np.diff(np.concatenate(([0], poss_reg))) == 1)[0]
        right = np.where(np.diff(np.concatenate((poss_reg, [0]))) == -1)[0]

        if SIGN_FORCE:
            sign = SIGN_FORCE
        else:
            nb_s = len(left < 30 * fs)
            loc = np.zeros(nb_s)
            for j in range(nb_s):
                loc[j] = np.argmax(np.abs(bpfecg[left[j]:right[j]]))
                loc[j] = loc[j] - 1 + left[j]
            sign = np.mean(ecg[loc])

        compt = 1
        NB_PEAKS = len(left)
        maxval = np.zeros(NB_PEAKS)
        maxloc = np.zeros(NB_PEAKS)
        for i in range(NB_PEAKS):
            if sign > 0:
                maxval[compt - 1], maxloc[compt - 1] = np.max(ecg[left[i]:right[i]])
            else:
                maxval[compt - 1], maxloc[compt - 1] = np.min(ecg[left[i]:right[i]])
            maxloc[compt - 1] = maxloc[compt - 1] - 1 + left[i]

            if compt > 1:
                if maxloc[compt - 1] - maxloc[compt - 2] < fs * REF_PERIOD and abs(maxval[compt - 1]) < abs(
                        maxval[compt - 2]):
                    compt -= 1
                elif maxloc[compt - 1] - maxloc[compt - 2] < fs * REF_PERIOD and abs(maxval[compt - 1]) >= abs(
                        maxval[compt - 2]):
                    compt -= 1
                else:
                    compt += 1
            else:
                compt += 1

        qrs_pos = maxloc
        R_t = tm[maxloc]
        R_amp = maxval
        hrv = 60 / np.diff(R_t)

        # Debug plots
        if debug:
            import matplotlib.pyplot as plt
        
            FONTSIZE = 20
            fig, ax = plt.subplots(4, 1, figsize=(10, 15))
        
            # Plot raw ECG and filtered ECG
            ax[0].plot(tm, ecg)
            ax[0].plot(tm, bpfecg, 'r')
            ax[0].set_title('Raw ECG (blue) and Zero-phase FIR Filtered ECG (red)')
            ax[0].set_ylabel('ECG')
            ax[0].set_xlim([0, tm[-1]])
        
            # Plot integrated ECG with scan boundaries over scaled ECG
            mdfint = np.concatenate(([0], mdfint))  # Corrected line
            ax[1].plot(tm[0:len(mdfint)], mdfint)
            ax[1].plot(tm, max(mdfint) * bpfecg / (2 * max(bpfecg)), 'r')
            ax[1].plot(tm[left], mdfint[left], 'og')
            ax[1].plot(tm[right], mdfint[right], 'om')
            ax[1].set_title('Integrated ECG with Scan Boundaries over Scaled ECG')
            ax[1].set_ylabel('Integrated ECG')
            ax[1].set_xlim([0, tm[-1]])
        
            # Plot ECG with R-peaks (black) and S-points (green) over ECG
            ax[2].plot(tm, bpfecg, 'r')
            ax[2].plot(R_t, R_amp, '+k')
            ax[2].set_title('ECG with R-peaks (black) and S-points (green) over ECG')
            ax[2].set_ylabel('ECG+R+S')
            ax[2].set_xlim([0, tm[-1]])
        
            # Plot Heart Rate (RR intervals)
            ax[3].plot(R_t[0:len(hrv)], hrv, 'r+')
            ax[3].set_title('Heart Rate (RR Intervals)')
            ax[3].set_ylabel('RR (s)')
            ax[3].set_xlim([0, tm[-1]])

        plt.show()

    else:
        qrs_pos = []
        R_t = []
        R_amp = []
        hrv = []
        sign = []
        en_thres = []

   
      


    return qrs_pos, sign, en_thres
