import numpy as np

def wqrsm(data, fs=125, PWfreq=60, TmDEF=100, jflag=0):
    global wqrsm_Yn, wqrsm_Yn1, wqrsm_Yn2
    global wqrsm_lt_tt, wqrsm_lbuf, wqrsm_ebuf
    global wqrsm_aet, wqrsm_et
    global wqrsm_LPn, wqrsm_LP2n, wqrsm_lfsc
    global wqrsm_data
    global wqrsm_BUFLN
    global wqrsm_LTwindow

    if jflag is None:
        jflag = 0
    if TmDEF is None:
        TmDEF = 100
    if PWfreq is None:
        PWfreq = 60
    if fs is None:
        fs = 125

    wqrsm_BUFLN = 16384
    EYE_CLS = 0.25
    MaxQRSw = 0.13
    NDP = 2.5
    PWFreqDEF = PWfreq
    WFDB_DEFGAIN = 200.0

    timer_d = 0
    gain = WFDB_DEFGAIN
    wqrsm_lfsc = int(1.25 * gain * gain / fs)
    PWFreq = PWFreqDEF

    datatest = data[:int(len(data) / fs) * fs]
    if len(datatest) > fs:
        datatest = np.reshape(datatest, (fs, -1))
    test_ap = np.median(np.max(datatest, axis=0) - np.min(datatest, axis=0))
    if test_ap < 10:
        data = data * gain

    wqrsm_data = data

    wqrsm_lbuf = np.zeros(wqrsm_BUFLN)
    wqrsm_ebuf = np.zeros(wqrsm_BUFLN)
    wqrsm_ebuf[:] = np.floor(np.sqrt(wqrsm_lfsc))

    wqrsm_lt_tt = 0
    wqrsm_aet = 0
    wqrsm_Yn = 0
    wqrsm_Yn1 = 0
    wqrsm_Yn2 = 0

    qrs = []
    jpoints = []

    Tm = int(TmDEF / 5.0)
    wqrsm_LPn = int(fs / PWFreq)
    if wqrsm_LPn > 8:
        wqrsm_LPn = 8
    wqrsm_LP2n = 2 * wqrsm_LPn
    EyeClosing = int(fs * EYE_CLS)
    ExpectPeriod = int(fs * NDP)
    wqrsm_LTwindow = int(fs * MaxQRSw)

    t1 = fs * 8
    if t1 > int(wqrsm_BUFLN * 0.9):
        t1 = int(wqrsm_BUFLN / 2)

    T0 = 0
    for t in range(1, t1 + 1):
        T0 = T0 + ltsamp(t)

    T0 = T0 / t1
    Ta = 3 * T0

    t = 1
    learning = 1
    while t < len(data):

        if learning == 1:
            if t > t1:
                learning = 0
                T1 = T0
                t = 1
            else:
                T1 = 2 * T0

        if ltsamp(t) > T1:
            timer_d = 0
            maxd = ltsamp(t)
            mind = maxd
            for tt in range(t + 1, t + int(EyeClosing / 2) + 1):
                if ltsamp(tt) > maxd:
                    maxd = ltsamp(tt)
            for tt in range(t - 1, t - int(EyeClosing / 2) - 1, -1):
                if ltsamp(tt) < mind:
                    mind = ltsamp(tt)
            if maxd > mind + 10:
                onset = int(maxd / 100) + 2
                tpq = t - 5
                for tt in range(t, t - int(EyeClosing / 2) - 1, -1):
                    if (
                        ltsamp(tt) - ltsamp(tt - 1) < onset
                        and ltsamp(tt - 1) - ltsamp(tt - 2) < onset
                        and ltsamp(tt - 2) - ltsamp(tt - 3) < onset
                        and ltsamp(tt - 3) - ltsamp(tt - 4) < onset
                    ):
                        tpq = tt - wqrsm_LP2n
                        break

                if learning != 1:
                    if tpq > len(data):
                        break
                    qrs.append(tpq)

                    if jflag:
                        tj = t + 5
                        for tt in range(t, t + int(EyeClosing / 2) + 1):
                            if ltsamp(tt) > maxd - int(maxd / 10):
                                tj = tt
                                break
                        if tj > len(data):
                            break
                        jpoints.append(tj)

                Ta = Ta + (maxd - Ta) / 10
                T1 = Ta / 3
                t = t + EyeClosing
        elif learning != 1:
            timer_d = timer_d + 1
            if timer_d > ExpectPeriod and Ta > Tm:
                Ta = Ta - 1
                T1 = Ta / 3
        t = t + 1

    return qrs, jpoints

def ltsamp(t):
    global wqrsm_Yn, wqrsm_Yn1, wqrsm_Yn2
    global wqrsm_lt_tt, wqrsm_lbuf, wqrsm_ebuf
    global wqrsm_aet, wqrsm_et
    global wqrsm_LPn, wqrsm_LP2n, wqrsm_lfsc
    global wqrsm_data
    global wqrsm_BUFLN
    global wqrsm_LTwindow

    while t > wqrsm_lt_tt:
        wqrsm_Yn2 = wqrsm_Yn1
        wqrsm_Yn1 = wqrsm_Yn

        v0 = wqrsm_data[wqrsm_lt_tt] if 0 <= wqrsm_lt_tt < len(wqrsm_data) else wqrsm_data[0]

        index_lp_n = wqrsm_lt_tt - wqrsm_LPn
        v1 = (
            wqrsm_data[index_lp_n] if 0 <= index_lp_n < len(wqrsm_data) else wqrsm_data[0]
        )

        index_lp_2n = wqrsm_lt_tt - wqrsm_LP2n
        v2 = (
            wqrsm_data[index_lp_2n] if 0 <= index_lp_2n < len(wqrsm_data) else wqrsm_data[0]
        )

        if v0 != -32768 and v1 != -32768 and v2 != -32768:
            wqrsm_Yn = 2 * wqrsm_Yn1 - wqrsm_Yn2 + v0 - 2 * v1 + v2
        dy = int((wqrsm_Yn - wqrsm_Yn1) / wqrsm_LP2n)
        wqrsm_lt_tt = wqrsm_lt_tt + 1
        wqrsm_et = int(np.sqrt(wqrsm_lfsc + dy * dy))
        id = wqrsm_lt_tt % wqrsm_BUFLN
        if id == 0:
            id = wqrsm_BUFLN - 1
        wqrsm_ebuf[id] = wqrsm_et
        id2 = (wqrsm_lt_tt - wqrsm_LTwindow) % wqrsm_BUFLN
        if id2 == 0:
            id2 = wqrsm_BUFLN - 1
        wqrsm_aet = wqrsm_aet + (wqrsm_et - wqrsm_ebuf[id2])
        wqrsm_lbuf[id] = wqrsm_aet

    id3 = t % wqrsm_BUFLN
    if id3 == 0:
        id3 = wqrsm_BUFLN - 1
    lt_data = wqrsm_lbuf[id3]
    return lt_data
