import cython
cimport cython
import numpy as np
cimport numpy as np

from numpy import fft
import math
import time

import wave
import pyaudio

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
 
cdef int nextpow2(int n) :
    cdef int m
    if(n < 0) :
        return 0
    else :
        m = int(math.ceil(math.log2(n)))
        return m

#### 周波数成分を求める ####
def culSpectrum(np.ndarray[DTYPE_t, ndim=1] wave, DTYPE_t sampleRate) :
    cdef int fftSize = 2 ** (nextpow2(len(wave)))

    #### 振幅を信号長で正規化 ####
    cdef np.ndarray[DTYPE_t, ndim=1] src
    src = wave / len(wave)

    #### FFT変換 ####
    #### cdef np.ndarray[DTYPE_t, ndim=1] spectrum
    spectrum = np.abs(fft.fft(src, fftSize))
    spectrum = spectrum / np.sum(spectrum)

    #### 対数振幅スペクトル導出 ####
    cdef np.ndarray[DTYPE_t, ndim=1] specLog
    specLog = 20 * np.log10(np.abs(spectrum))

    #### 周波数のビン ####
    cdef np.ndarray[DTYPE_t, ndim=1] freqs
    freqs = np.array(range(0, fftSize)) * sampleRate / fftSize

    return specLog, freqs

#### 音量(dB)を求める ####
cdef DTYPE_t detectLoudness(np.ndarray[DTYPE_t, ndim=1] waveform) :
    cdef DTYPE_t square_mean
    square_mean = np.sqrt(np.mean(waveform * waveform))

    cdef DTYPE_t loudness
    loudness = np.log10(square_mean) * 20

    return loudness

cdef int OnPeriods = 20

def makeFeature(np.ndarray[DTYPE_t, ndim=1] waveform, DTYPE_t sampleRate) :
    #### 入力された波形から特徴ベクトル作成

    #### 50dB以上の音量を持つ部分(ON period)から 2sec, 0.05secごとに切り分ける(計40個)
    #### 切り分けた波形ごとに周波数成分を求める
    #### 全てまとめてfeature vector として出力

    cdef np.ndarray[DTYPE_t, ndim=2] featureVector
    featureVector = np.zeros([OnPeriods, 1858])

    cdef DTYPE_t BinSize
    BinSize = 0.05 * sampleRate
    cdef int length
    length = int(len(waveform) / BinSize)

    #### 最初のOn periodを求める

    cdef int flag = 0

    cdef int step
    cdef int firstON
    cdef DTYPE_t loudness
    
    for step in range(0, length) :
        #### 音量を求める
        loudness = detectLoudness(waveform[int(BinSize * step) : int(BinSize * (step + 1))])
        if((loudness >= 50) and (flag == 1)) :
            firstON = step
            break
        if(loudness <= 50) :
            flag = 1

    #### 最初のOn periodから0.05secごとに周波数成分を求める
    cdef np.ndarray[DTYPE_t, ndim=1] specLog
    for step in range(firstON, min(firstON + OnPeriods, length)) :
        specLog, freqs = culSpectrum(waveform[int(BinSize * step) : int(BinSize * (step + 1))], sampleRate)
        #### 20kHz以下の成分のみを利用する
        specLog = specLog[np.where(freqs < 20000)]

        featureVector[step - firstON, :] = specLog

    ####if(step == length - 1) :
        ####print(filename)

    return featureVector

def makePrediction(np.ndarray[DTYPE_t, ndim=2] feature, np.ndarray[DTYPE_t, ndim=3] sigma, np.ndarray[DTYPE_t, ndim=2] mu, np.ndarray[DTYPE_t, ndim=1] detSigma, np.ndarray[DTYPE_t, ndim=3] invSigma) :
    #### 他クラス分類問題をMAP推定によって解く
    #### feature: 特徴ベクトル
    #### sigma: クラスごとの共分散行列
    #### mu: クラスごとの平均ベクトル

    cdef int numClass, d
    numClass = sigma.shape[0]
    d = sigma.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=1] logDist
    logDist = np.zeros(numClass)

    cdef np.ndarray[DTYPE_t, ndim=1] x
    cdef np.ndarray[DTYPE_t, ndim=2] sigmaC, invSigmaC, x_muC
    cdef np.ndarray[DTYPE_t, ndim=1] muC
    cdef DTYPE_t detSigmaC

    for x in feature :
        for c in range(0, numClass) :
            sigmaC = sigma[c]
            muC = mu[c]
            detSigmaC = detSigma[c]
            invSigmaC = invSigma[c]

            # x - muのベクトル
            x_muC = np.array([x - muC]).T

            logDist[c] += np.log((2 * np.pi) ** (-d / 2.0) * detSigmaC ** (-1.0 / 2)) - (x_muC).T.dot(invSigmaC).dot(x_muC) / 2

    # 予測クラス
    cdef int predClass
    predClass = np.argmax(logDist)

    return predClass

cdef np.ndarray[DTYPE_t, ndim=1] makeMu(np.ndarray[DTYPE_t, ndim=2] data) :
    cdef np.ndarray[DTYPE_t, ndim=1] mu
    mu = np.average(data, 0)
    return mu

cdef np.ndarray[DTYPE_t, ndim=2] makeSigma(np.ndarray[DTYPE_t, ndim=2] data) :
    cdef np.ndarray[DTYPE_t, ndim=2] sigma
    sigma = np.cov(data, rowvar=0)
    return sigma

def detectFreqs(np.ndarray[DTYPE_t, ndim=4] features, int numFreqs) :
    print("preparation")
    time1 = time.time()

    cdef int numCoins, numData, numTimes, lengthData
    numCoins = features.shape[0]
    numData = features.shape[1]
    numTimes = features.shape[2]
    lengthData = features.shape[3]

    ####
    # コインの種類xデータx時間x周波数成分 のデータを
    # コインの種類x(データx時間)x周波数成分 に変える
    ####

    cdef np.ndarray[DTYPE_t, ndim=2] X, tempX, tempX2
    cdef np.ndarray[DTYPE_t, ndim=1] Y
    X = np.zeros([numCoins * numData * numTimes, lengthData])
    tempX = np.zeros([numCoins * numData * numTimes, numFreqs])
    Y = np.zeros(numCoins * numData * numTimes)

    cdef int c, cc, ccc
    for c in range(0, numCoins) :
        for cc in range(0, numData) :
            for ccc in range(0, numTimes) :
                X[c * (numData * numTimes) + cc * (numTimes) + ccc, :] = features[c, cc, ccc, :]
                Y[c * (numData * numTimes) + cc * (numTimes) + ccc] = c
                

    cdef np.ndarray[DTYPE_t, ndim=2] mu
    cdef np.ndarray[DTYPE_t, ndim=3] sigma
    cdef np.ndarray[np.int_t, ndim=1] freqs
    cdef np.ndarray[DTYPE_t, ndim=1] detSigma
    cdef np.ndarray[DTYPE_t, ndim=3] invSigma

    mu = np.zeros([numCoins, numFreqs])
    sigma = np.zeros([numCoins, numFreqs, numFreqs])
    freqs = np.array(range(0, numFreqs)).astype('int64')

    detSigma = np.zeros(numCoins)
    invSigma = np.zeros([numCoins, numFreqs, numFreqs])

    ####
    # 予測に聞く周波数成分をnumFreqs個選択する
    # アルゴリズムは Inferring objects from a multitude of oscillations参照
    ####
    cdef np.ndarray[np.int_t, ndim=1] freqs_old
    cdef int argminR, omega, f
    cdef DTYPE_t minR, tempR
    cdef np.ndarray[DTYPE_t, ndim=2] tempDis
    cdef np.ndarray[DTYPE_t, ndim=2] tempF
    cdef np.ndarray[DTYPE_t, ndim=1] x

    cdef np.ndarray[DTYPE_t, ndim=2] A

    print("main loop start")
    while(True) :
        print("********************************")
        print("major loop")
        print("--------------------------------")

        freqs_old = freqs.copy()

        for f in range(0, numFreqs) :
            freqs[f] = -1
            argminR = -1
            minR = np.inf
            maxQda = -np.inf
            for omega in range(0, lengthData) :
                time2 = time.time()
                # print("minor minor loop : ", time2 - time1)
                time1 = time.time()
                # Rが最小となるomegaを探す
                # freqsの中に、重複するものがあってはいけない
                if (len(np.where(freqs == omega)[0]) != 0) :
                    continue
                freqs_temp = freqs.copy()
                freqs_temp[f] = omega
                A = np.zeros([numFreqs, lengthData])
                for temp_index, temp in enumerate(freqs_temp) :
                    A[temp_index, temp] = 1
                tempDis = np.zeros([numCoins, numCoins * numData * numTimes])

                tempX = np.dot(A, X.T).T
                
                for c in range(0, numCoins) :
                    tempF = tempX[np.where(Y == c)]
                    mu[c, :] = makeMu(tempF)
                    sigma[c, :, :] = makeSigma(tempF)
                    detSigma[c] = np.linalg.det(sigma[c, :, :])
                    invSigma[c, :, :] = np.linalg.inv(sigma[c, :, :])

                for c in range(0, numCoins) :
                    tempX2 = tempX - mu[c]
                    tempDis[c] = np.exp(-0.5 * np.sum(np.dot(tempX2, invSigma[c]) * tempX2, axis=1)) * ((2 * np.pi) ** (-numFreqs / 2.0) * detSigma[c] ** (-0.5))

                predictedC = np.argmax(tempDis, axis=0)
                tempR = np.sum(1 - tempDis[predictedC, np.array(range(0, predictedC.shape[0]))] / np.sum(tempDis, axis=0))
                
                # qda = QDA(sigma=sigma, mu=mu)

                tempR /= numCoins * numData * numTimes
                if(minR > tempR) :
                    minR = tempR
                    argminR = omega
                    # print(maxQda < qda)
                    # maxQda = qda

            freqs[f] = argminR
            print("minor loop : ", f)
            print("Selected freqs", (freqs).astype("int"))
            print("R=", minR)
            print("--------------------------------")
        if((freqs_old == freqs).min()) :
            break

    print("done")

    return freqs, sigma, mu, detSigma, invSigma

def culRsup(sigma, mu, detSigma, invSigma) :
    cdef int c, d
    d = sigma.shape[1]

    cons = np.array([((2 * np.pi) ** (-d / 2.0) * detSigma[c] ** (-0.5) * np.exp(-np.matrix(mu[c]).dot(invSigma[c]).dot(np.matrix(mu[c]).T) / 2))[0, 0] for c in range(0, sigma.shape[0])])

    def function(np.ndarray[DTYPE_t, ndim=1] x, int c) :
        x_mat = np.array([x])
        arr = np.array([np.exp(-(x_mat.dot(invSigma[coin]).dot(x_mat.T) - 2 * np.array([mu[coin]]).dot(invSigma[coin]).dot(np.array([mu[coin]]).T)) / 2)[0, 0] for coin in range(0, sigma.shape[0])]) * cons

        return arr[c] / np.sum(arr)

    return function


cdef DTYPE_t QDA(np.ndarray[DTYPE_t, ndim=3] sigma, np.ndarray[DTYPE_t, ndim=2] mu) :
    cdef np.ndarray[DTYPE_t, ndim=2] allSigma, sumSigma
    allSigma = makeSigma(mu)

    sumSigma = np.sum(sigma, 0) / 6

    return np.linalg.det(allSigma) / np.linalg.det(sumSigma)

class Bayes :
    def __init__(self, np.ndarray[DTYPE_t, ndim=2] mu, np.ndarray[DTYPE_t, ndim=3] sigma) :
        cdef int c
        self.numClass = mu.shape[0]
        self.numDimension = mu.shape[1]

        self.detSigma = np.zeros(self.numClass)
        self.invSigma = np.zeros_like(sigma)
        self.mu = mu
        self.sigma = sigma

        for c in range(0, self.numClass) :
            self.detSigma[c] = np.linalg.det(sigma[c])
            self.invSigma[c, :, :] = np.linalg.inv(sigma[c])

        self.cons = np.array([((2 * np.pi) ** (-self.numDimension / 2.0) * self.detSigma[c] ** (-0.5) * np.exp(-np.matrix(self.mu[c]).dot(self.invSigma[c]).dot(np.matrix(self.mu[c]).T) / 2))[0, 0] for c in range(0, self.numClass)])


    def culLikelihood(self, np.ndarray[DTYPE_t, ndim=1] x, int c) :
        cdef np.ndarray[DTYPE_t, ndim=2] x_mat
        cdef DTYPE_t arr
        x_mat = np.array([x])
        arr = np.exp(-(x_mat.dot(self.invSigma[c]).dot(x_mat.T) - 2 * np.array([self.mu[c]]).dot(self.invSigma[c]).dot(x_mat.T)) / 2)[0, 0] * self.cons[c]

        return arr

    def makePrediction(self, x) :
        cdef int predictedClass
        cdef int c

        return np.argmax([self.culLikelihood(x, c) for c in range(0, self.numClass)])

    def score(self, x) :
        cdef np.ndarray[DTYPE_t, ndim=1] l
        cdef int c
        l = np.array([self.culLikelihood(x, c) for c in range(0, self.numClass)])
        c = np.argmax(l)
        return l[c] / np.sum(l)
        
        
def getNewFile(filename, sigma, mu, detSigma, invSigma, freqs) :
    wavefile = wave.open(filename, "r")
    cdef DTYPE_t framerate = wavefile.getframerate()
    data = wavefile.readframes(wavefile.getnframes())
    wavefile.close()
    x = np.frombuffer(data, dtype="int16")
    x = x.astype("float64")
    cdef DTYPE_t BinSize = 0.05 * framerate

    cdef np.ndarray[DTYPE_t, ndim=2] f = makeFeature(x, framerate)[:, freqs]

    predicted = makePrediction(f, sigma=sigma, mu=mu, detSigma=detSigma, invSigma=invSigma)

    return predicted
