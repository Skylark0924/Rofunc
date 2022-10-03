import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import librosa as lib
import numpy as np
import math

def filtering(data_raw, n):
    '''
    Filter the original EMG signals (from the Delsys system, 1000 Hz) to the desired frequency
    '''
    data = []
    for i in range(0, len(data_raw) - n + 1, n):
        data_new = data_raw[i]
        data.append(data_new)
    data = np.array(data)
    return data

def absolutevalue(data):
    '''
    Take the absolute value of the EMG signals
    '''
    for i in range(len(data)):
        data[i] = abs(data[i])
    return data

def processing(data, sample):
    '''
    Clean the raw EMG signals and calculate the Maximum Voluntary Contraction (MVC) of the EMG signals
    '''
    signals, info = nk.emg_process(data, sampling_rate=sample)
    signals_array = signals.values
    EMG_clean = signals_array[:, 1]
    MVC = signals_array[:, 2]
    return EMG_clean, MVC, signals

def activationlevel(data, MVC):
    '''
    Calculate the activation level of the EMG signals
    '''
    A = [0] * len(data)
    for i in range(len(data)):
        A[i] = data[i] / MVC
        # A[i] = abs(A[i])
        # if A[i] > 100:
        #     A[i] = 100
    A = np.array(A)
    return A

if __name__ == '__main__':
    a = 20
    b = 0.05
    n = 4
    FRAME_SIZE = 64
    HOP_LENGTH = 32
    # Generate 10 seconds of EMG signal (recorded at 250 samples / second)
    emg_1 = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3)
    emg_2 = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=0)

    emg_filter_1 = filtering(emg_1, n)
    emg_filter_2 = filtering(emg_2, n)

    EMG_clean_1, MVC_1, signals_1 = processing(emg_filter_1, int(1000 / n))
    EMG_clean_2, MVC_2, signals_2 = processing(emg_filter_2, int(1000 / n))

    EMG_abs_1 = absolutevalue(EMG_clean_1)
    EMG_abs_2 = absolutevalue(EMG_clean_2)

    '''Calculate the root mean square (RMS) of the EMG signals'''
    EMG_rms_1 = lib.feature.rms(EMG_abs_1, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    EMG_rms_2 = lib.feature.rms(EMG_abs_2, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    A_1 = activationlevel(MVC_1, 0.35)
    A_2 = activationlevel(MVC_2, 0.35)

    c_h = [0] * len(A_1)
    for i in range(len(A_1)):
        c_h[i] = a * (1 - math.exp(- b * (A_1[i] + A_2[i]))) / (1 + math.exp(- b * (A_1[i] + A_2[i])))
        # c_h[i] = a * (1 - math.exp(- b * (MVC_1[i] + MVC_2[i]))) / (1 + math.exp(- b * (MVC_1[i] + MVC_2[i])))
    c_h = np.array(c_h)
    plt.plot(c_h)

    # Visualise the processing
    nk.emg_plot(signals_1, sampling_rate=int(1000 / n))
    plt.show()