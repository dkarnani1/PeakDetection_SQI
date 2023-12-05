import numpy as np

def wqrsm_fast(data, fs=125, PWfreq=60, TmDEF=100, jflag=0):
    BUFLN = 16384
    EYE_CLS = 0.25
    MaxQRSw = 0.13
    NDP = 2.5
    WFDB_DEFGAIN = 200.0

    timer_d = 0
    gain = WFDB_DEFGAIN
    lfsc = int(1.25 * gain * gain / fs)
    PWFreq = PWfreq

    if PWFreq > 8:
        PWFreq = 8

    LPn = int(fs / PWFreq)
    LP2n = 2 * LPn
    EyeClosing = int(fs * EYE_CLS)
    ExpectPeriod = int(fs * NDP)
    LTwindow = int(fs * MaxQRSw)

    data = np.array(data)
    datatest = data[:int(len(data) / fs) * fs].reshape(fs, -1)
    test_ap = np.median(np.max(datatest, axis=0) - np.min(datatest, axis=0))

    if test_ap < 10:
        data *= gain

    lbuf = np.zeros(BUFLN)
    ebuf = np.zeros(BUFLN)
    ebuf[:BUFLN] = np.sqrt(lfsc)
    lt_tt = 0
    aet = 0
    Yn = 0
    Yn1 = 0
    Yn2 = 0

    qrs = []
    jpoints = []

    Tm = int(TmDEF / 5.0)

    T0 = 0
    t1 = fs * 8

    if t1 > int(BUFLN * 0.9):
        t1 = int(BUFLN / 2)

    for t in range(1, t1 + 1):
        lt_data, Yn, Yn1, Yn2, aet, lbuf, ebuf = ltsamp(t, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                        lbuf, ebuf, aet, LTwindow)
        T0 += lt_data

    T0 /= t1
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

        lt_data, Yn, Yn1, Yn2, aet, lbuf, ebuf = ltsamp(t, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                        lbuf, ebuf, aet, LTwindow)

        if lt_data > T1:
            timer_d = 0
            maxd = lt_data
            mind = maxd

            for tt in range(t + 1, t + int(EyeClosing / 2) + 1):
                lt_data, _, _, _, _, _, _ = ltsamp(tt, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                    lbuf, ebuf, aet, LTwindow)
                if lt_data > maxd:
                    maxd = lt_data

            for tt in range(t - 1, t - int(EyeClosing / 2) - 1, -1):
                lt_data, _, _, _, _, _, _ = ltsamp(tt, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                    lbuf, ebuf, aet, LTwindow)
                if lt_data < mind:
                    mind = lt_data

            if maxd > mind + 10:
                onset = int(maxd / 100) + 2
                tpq = t - 5

                for tt in range(t, t - int(EyeClosing / 2) - 1, -1):
                    lt_data, _, _, _, _, _, _ = ltsamp(tt, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                        lbuf, ebuf, aet, LTwindow)
                    lt_data_1, _, _, _, _, _, _ = ltsamp(tt - 1, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                          lbuf, ebuf, aet, LTwindow)
                    lt_data_2, _, _, _, _, _, _ = ltsamp(tt - 2, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                          lbuf, ebuf, aet, LTwindow)
                    lt_data_3, _, _, _, _, _, _ = ltsamp(tt - 3, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                          lbuf, ebuf, aet, LTwindow)
                    lt_data_4, _, _, _, _, _, _ = ltsamp(tt - 4, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2,
                                                          lbuf, ebuf, aet, LTwindow)

                    if (
                        lt_data - lt_data_1 < onset
                        and lt_data_1 - lt_data_2 < onset
                        and lt_data_2 - lt_data_3 < onset
                        and lt_data_3 - lt_data_4 < onset
                    ):
                        tpq = tt - LP2n
                        break

                if learning != 1 and tpq < len(data):
                    qrs.append(tpq)

                    if jflag:
                        tj = t + 5

                        for tt in range(t, t + int(EyeClosing / 2) + 1):
                            lt_data, _, _, _, _, _, _ = ltsamp(tt, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1,
                                                                Yn2, lbuf, ebuf, aet, LTwindow)
                            if lt_data > maxd - int(maxd / 10):
                                tj = tt
                                break

                        if tj < len(data):
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


def ltsamp(t, fs, data, lt_tt, lfsc, LP2n, LPn, BUFLN, Yn, Yn1, Yn2, lbuf, ebuf, aet, LTwindow):
    while t > lt_tt:
        Yn2 = Yn1
        Yn1 = Yn
        v0 = data[0]
        v1 = data[0]
        v2 = data[0]

        if 0 < lt_tt <= len(data):
            v0 = data[lt_tt]

        if 0 < lt_tt - LPn <= len(data):
            v1 = data[lt_tt - LPn]

        if 0 < lt_tt - LP2n <= len(data):
            v2 = data[lt_tt - LP2n]

        if v0 != -32768 and v1 != -32768 and v2 != -32768:
            Yn = 2 * Yn1 - Yn2 + v0 - 2 * v1 + v2

        dy = int((Yn - Yn1) / LP2n)
        lt_tt += 1
        et = int(np.sqrt(lfsc + dy * dy))
        id = lt_tt % BUFLN if lt_tt % BUFLN != 0 else BUFLN
        ebuf[id - 1] = et

        id2 = (lt_tt - LTwindow) % BUFLN if (lt_tt - LTwindow) % BUFLN != 0 else BUFLN
        aet = aet + (et - ebuf[id2 - 1])
        lbuf[id - 1] = aet

    id3 = t % BUFLN if t % BUFLN != 0 else BUFLN
    lt_data = lbuf[id3 - 1]

    return lt_data, Yn, Yn1, Yn2, aet, lbuf, ebuf


# Example usage:
# data = ...  # Provide your ECG data as a list or NumPy array
# qrs, jpoints = wqrsm_fast(data)
# print("QRS points:", qrs)
# print("J-points:", jpoints)
