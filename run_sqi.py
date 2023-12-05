import numpy as np
import matplotlib.pyplot as plt

class QRSComparison:
    def __init__(self, refqrs, testqrs, thres=0.05, margin=2, windowlen=60, fs=1000):
        self.refqrs = refqrs
        self.testqrs = testqrs
        self.thres = thres
        self.margin = margin
        self.windowlen = windowlen
        self.fs = fs

    def run_sqi(self):
        start = int(self.margin * self.fs)
        stop = int((self.windowlen - self.margin) * self.fs)
        refqrs = np.array(self.refqrs) * self.fs
        testqrs = np.array(self.testqrs) * self.fs
        refqrs = refqrs.astype(int)
        testqrs = testqrs.astype(int)

        try:
            refqrs = refqrs[(refqrs > start) & (refqrs < stop)]
            testqrs = testqrs[(testqrs > start) & (testqrs < stop)]

            if refqrs.size > 0:
                NB_REF = len(refqrs)
                NB_TEST = len(testqrs)

                indbord = np.where((refqrs < self.thres * self.fs) | (refqrs > (self.windowlen - self.thres) * self.fs))[0]
                if indbord.size > 0:
                    IndMatchBord, DistQRSbord = self._dsearchn(testqrs, refqrs[indbord])
                    Indeces_below_threshold = DistQRSbord < self.thres * self.fs
                    IndMatchBord = IndMatchBord[Indeces_below_threshold]
                    NB_QRS_BORD = len(indbord)
                    NB_MATCHING = len(IndMatchBord)
                    if len(IndMatchBord) == 0:
                        refqrs = np.delete(refqrs, indbord)
                    elif NB_MATCHING < NB_QRS_BORD:
                        refqrs = np.delete(refqrs, indbord[~Indeces_below_threshold])

                indbord = np.where((testqrs < self.thres * self.fs) | (testqrs > (self.windowlen - self.thres) * self.fs))[0]
                if indbord.size > 0:
                    IndMatchBord, DistQRSbord = self._dsearchn(refqrs, testqrs[indbord])
                    Indeces_below_threshold = DistQRSbord < self.thres * self.fs
                    IndMatchBord = IndMatchBord[Indeces_below_threshold]
                    NB_QRS_BORD = len(indbord)
                    NB_MATCHING = len(IndMatchBord)
                    if len(IndMatchBord) == 0:
                        testqrs = np.delete(testqrs, indbord)
                    elif NB_MATCHING < NB_QRS_BORD:
                        testqrs = np.delete(testqrs, indbord[~Indeces_below_threshold])

                IndMatch, Dist = self._dsearchn(refqrs, testqrs)
                IndMatchInWindow = IndMatch[Dist < self.thres * self.fs]
                NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
                TP = NB_MATCH_UNIQUE
                FN = NB_REF - TP
                FP = NB_TEST - TP
                Se = TP / (TP + FN)
                PPV = TP / (FP + TP)
                F1 = 2 * Se * PPV / (Se + PPV)

                Nb = {'TP': TP, 'FN': FN, 'FP': FP}

                return F1, Se, PPV, Nb
            else:
                return None, None, None, None
        except Exception as e:
            print(e)
            return None, None, None, None

    def _dsearchn(self, X, Y):
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        X_2 = np.sum(X ** 2, axis=1)
        Y_2 = np.sum(Y ** 2, axis=1)
        XY = np.dot(X, Y.T)
        dist = np.sqrt(X_2[:, np.newaxis] - 2 * XY + Y_2[np.newaxis, :])
        IndMatch = np.argmin(dist, axis=0)
        Dist = dist[IndMatch, np.arange(dist.shape[1])]
        return IndMatch, Dist

