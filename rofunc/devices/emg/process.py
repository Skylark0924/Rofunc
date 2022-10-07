import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np


def filtering(data_raw, n):
    """
    Filter the original EMG signals (from the Delsys system, 1000 Hz) to the desired frequency
    """
    data = []
    for i in range(0, len(data_raw) - n + 1, n):
        data_new = data_raw[i]
        data.append(data_new)
    data = np.array(data)
    return data


def absolutevalue(data):
    """
    Take the absolute value of the EMG signals
    """
    for i in range(len(data)):
        data[i] = abs(data[i])
    return data


def processing(data, sample):
    """
    Clean the raw EMG signals and calculate the Maximum Voluntary Contraction (MVC) of the EMG signals
    """
    signals, info = nk.emg_process(data, sampling_rate=sample)
    signals_array = signals.values
    EMG_clean = signals_array[:, 1]
    MVC = signals_array[:, 2]
    return EMG_clean, MVC, signals


def activationlevel(data, MVC):
    """
    Calculate the activation level of the EMG signals
    """
    A = [0] * len(data)
    for i in range(len(data)):
        A[i] = data[i] / MVC
        # A[i] = abs(A[i])
        # if A[i] > 100:
        #     A[i] = 100
    A = np.array(A)
    return A

def process(data, n):
    """
    Filter the original EMG signals (from the Delsys system, 1000 Hz) to the desired frequency
    """
    data_filter = []
    for i in range(0, len(data) - n + 1, n):
        data_new = data[i]
        data_filter.append(data_new)
    data_filter = np.array(data_filter)

if __name__ == '__main__':
    # FRAME_SIZE = 64
    # HOP_LENGTH = 32
    SAMPING_RATE = 2000
    n = 4
    # Generate 10 seconds of EMG signal (recorded at 250 samples / second)
    # emg_1 = nk.emg_simulate(duration=10, sampling_rate=SAMPING_RATE, burst_number=3)
    # emg_2 = nk.emg_simulate(duration=10, sampling_rate=SAMPING_RATE, burst_number=2)
    emg = np.load('./data/emg_data.npy')

    emg_filter_1 = filtering(emg[:, 0], n)
    emg_filter_2 = filtering(emg[:, 1], n)

    EMG_clean_1, MVC_1, signals_1 = processing(emg_filter_1, int(SAMPING_RATE / n))
    EMG_clean_2, MVC_2, signals_2 = processing(emg_filter_2, int(SAMPING_RATE / n))

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Raw and Clean EMG Signals", fontweight="bold", fontdict={'family': 'Times New Roman'},
                 fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, emg_filter_1.shape[0] / int(SAMPING_RATE / n), emg_filter_1.shape[0])
    # Plot cleaned and raw EMG.
    legend_font = {"family": "Times New Roman"}
    ax0.set_title("Sensor_1", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, emg_filter_1, color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(
        x_axis, EMG_clean_1, color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5
    )
    ax0.legend(loc="upper right", frameon=True, prop=legend_font)

    ax1.set_title("Sensor_2", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax1.plot(x_axis, emg_filter_2, color="#B0BEC5", label="Raw", zorder=1)
    ax1.plot(
        x_axis, EMG_clean_2, color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5
    )
    ax1.legend(loc="upper right", frameon=True, prop=legend_font)
    plt.show()

    EMG_abs_1 = absolutevalue(EMG_clean_1)
    EMG_abs_2 = absolutevalue(EMG_clean_2)

    # '''Calculate the root mean square (RMS) of the EMG signals'''
    # EMG_rms_1 = lib.feature.rms(EMG_abs_1, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    # EMG_rms_2 = lib.feature.rms(EMG_abs_2, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    A_1 = activationlevel(MVC_1, 0.35)
    A_2 = activationlevel(MVC_2, 0.35)
    # Plot absolute value and MVC of EMG.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Absolute Value and MVC of EMG signals", fontweight="bold",
                 fontdict={'family': 'Times New Roman'}, fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, EMG_abs_1.shape[0] / int(SAMPING_RATE / n), EMG_abs_1.shape[0])
    # Plot cleaned and raw EMG.
    ax0.set_title("Sensor_1", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, EMG_abs_1, color="#B0BEC5", label="ABS", zorder=1)
    ax2 = ax0.twinx()
    ax2.plot(
        x_axis, MVC_1, color="#FA6839", label="MVC", linewidth=1.5
    )
    ax0.legend(loc="upper left", frameon=True, prop=legend_font)
    ax2.legend(loc="upper right", frameon=True, prop=legend_font)

    ax1.set_title("Sensor_2", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax1.plot(x_axis, EMG_abs_2, color="#B0BEC5", label="ABS", zorder=1)
    ax3 = ax1.twinx()
    ax3.plot(
        x_axis, MVC_2, color="#FA6839", label="MVC", linewidth=1.5
    )
    ax1.legend(loc="upper left", frameon=True, prop=legend_font)
    ax3.legend(loc="upper right", frameon=True, prop=legend_font)
    plt.show()

    # c_h = [0] * len(A_1)
    # for i in range(len(A_1)):
    #     c_h[i] = a * (1 - math.exp(- b * (A_1[i] + A_2[i]))) / (1 + math.exp(- b * (A_1[i] + A_2[i])))
    #     # c_h[i] = a * (1 - math.exp(- b * (MVC_1[i] + MVC_2[i]))) / (1 + math.exp(- b * (MVC_1[i] + MVC_2[i])))
    # c_h = np.array(c_h)

    # Visualise the processing
    # nk.emg_plot(signals_1, sampling_rate=int(1000 / n))
    # plt.show()
