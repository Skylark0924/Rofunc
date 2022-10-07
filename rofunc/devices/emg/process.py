import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np


def process(data, sampling_rate, n):
    """

    Args:
        data: raw emg data
        sampling_rate: recorded at samples / second
        n: filtering rate of data to samples / n / second


    Returns:
    data_filter: Filter the original EMG signals (from the Delsys system, 2000 Hz) to the desired frequency
    data_clean: Clean the raw EMG signals
    data_mvc: Calculate the Maximum Voluntary Contraction (MVC) of the EMG signals
    data_abs: Take the absolute value of the EMG signals
    """
    data_filter = []
    for i in range(0, len(data) - n + 1, n):
        data_new = data[i]
        data_filter.append(data_new)
    data_filter = np.array(data_filter)

    signals, info = nk.emg_process(data_filter, sampling_rate=sampling_rate)
    signals_array = signals.values
    data_clean = signals_array[:, 1]
    data_mvc = signals_array[:, 2]

    data_abs = [0] * len(data_clean)
    for i in range(len(data_clean)):
        data_abs[i] = abs(data_clean[i])
    data_abs = np.array(data_abs)
    return data_filter, data_clean, data_mvc, data_abs


if __name__ == '__main__':
    # FRAME_SIZE = 64
    # HOP_LENGTH = 32

    # Generate 10 seconds of EMG signal (recorded at 250 samples / second)
    # emg_1 = nk.emg_simulate(duration=10, sampling_rate=SAMPING_RATE, burst_number=3)
    # emg_2 = nk.emg_simulate(duration=10, sampling_rate=SAMPING_RATE, burst_number=2)

    # '''Calculate the root mean square (RMS) of the EMG signals'''
    # EMG_rms_1 = lib.feature.rms(EMG_abs_1, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    # EMG_rms_2 = lib.feature.rms(EMG_abs_2, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    SAMPING_RATE = 2000
    n = 4
    emg = np.load('./data/emg_data.npy')
    data_filter_1, data_clean_1, data_mvc_1, data_abs_1 = process(emg[:, 0], SAMPING_RATE, n)
    data_filter_2, data_clean_2, data_mvc_2, data_abs_2 = process(emg[:, 1], SAMPING_RATE, n)

    # Plot cleaned and raw EMG.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Raw and Clean EMG Signals", fontweight="bold", fontdict={'family': 'Times New Roman'},
                 fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, data_filter_1.shape[0] / int(SAMPING_RATE / n), data_filter_1.shape[0])
    legend_font = {"family": "Times New Roman"}
    ax0.set_title("Sensor_1", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, data_filter_1, color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(
        x_axis, data_clean_1, color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5
    )
    ax0.legend(loc="upper right", frameon=True, prop=legend_font)
    ax1.set_title("Sensor_2", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax1.plot(x_axis, data_filter_2, color="#B0BEC5", label="Raw", zorder=1)
    ax1.plot(
        x_axis, data_clean_2, color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5
    )
    ax1.legend(loc="upper right", frameon=True, prop=legend_font)
    plt.show()

    # Plot absolute value and MVC of EMG.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Absolute Value and MVC of EMG signals", fontweight="bold",
                 fontdict={'family': 'Times New Roman'}, fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, data_abs_1.shape[0] / int(SAMPING_RATE / n), data_abs_1.shape[0])
    ax0.set_title("Sensor_1", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, data_abs_1, color="#B0BEC5", label="ABS", zorder=1)
    ax2 = ax0.twinx()
    ax2.plot(
        x_axis, data_mvc_1, color="#FA6839", label="MVC", linewidth=1.5
    )
    ax0.legend(loc="upper left", frameon=True, prop=legend_font)
    ax2.legend(loc="upper right", frameon=True, prop=legend_font)
    ax1.set_title("Sensor_2", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax1.plot(x_axis, data_abs_2, color="#B0BEC5", label="ABS", zorder=1)
    ax3 = ax1.twinx()
    ax3.plot(
        x_axis, data_mvc_2, color="#FA6839", label="MVC", linewidth=1.5
    )
    ax1.legend(loc="upper left", frameon=True, prop=legend_font)
    ax3.legend(loc="upper right", frameon=True, prop=legend_font)
    plt.show()

    # Visualise the processing
    # nk.emg_plot(signals_1, sampling_rate=int(1000 / n))
    # plt.show()
