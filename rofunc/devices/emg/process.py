import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np


def process_one_channel(data, sampling_rate, k):
    """

    Args:
        data: raw emg data
        sampling_rate: recorded at samples / second
        k: filtering rate of data to samples / k / second


    Returns:
    data_filter: Filter the original EMG signals (from the Delsys system, 2000 Hz) to the desired frequency
    data_clean: Clean the raw EMG signals
    data_mvc: Calculate the Maximum Voluntary Contraction (MVC) of the EMG signals
    data_abs: Take the absolute value of the EMG signals
    """
    data_filter = []
    for i in range(0, len(data) - k + 1, k):
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


def process_all_channels(data, n, sampling_rate, k):
    """

    Args:
        data: raw emg data
        n: number of emg channels
        sampling_rate: recorded at samples / second
        k: filtering rate of data to samples / k / second


    Returns:
    DATA_FILTER: Filter the original EMG signals (from the Delsys system, 2000 Hz) to the desired frequency
    DATA_CLEAN: Clean the raw EMG signals
    DATA_MVC: Calculate the Maximum Voluntary Contraction (MVC) of the EMG signals
    DATA_ABS: Take the absolute value of the EMG signals
    """
    DATA_FILTER = []
    DATA_CLEAN = []
    DATA_MVC = []
    DATA_ABS = []
    for j in range(n):
        data_filter = []
        for i in range(0, len(data[:, j]) - k + 1, k):
            data_new = data[i, j]
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

        DATA_FILTER.append(data_filter)
        DATA_CLEAN.append(data_clean)
        DATA_MVC.append(data_mvc)
        DATA_ABS.append(data_abs)
    DATA_FILTER = np.transpose(np.array(DATA_FILTER), (1, 0))
    DATA_CLEAN = np.transpose(np.array(DATA_CLEAN), (1, 0))
    DATA_MVC = np.transpose(np.array(DATA_MVC), (1, 0))
    DATA_ABS = np.transpose(np.array(DATA_ABS), (1, 0))
    return DATA_FILTER, DATA_CLEAN, DATA_MVC, DATA_ABS


def plot_raw_and_clean(data_filter, data_clean, k):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax0.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Raw and Clean EMG Signals", fontweight="bold", fontdict={'family': 'Times New Roman'},
                 fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, data_filter.shape[0] / int(2000 / k), data_filter.shape[0])
    legend_font = {"family": "Times New Roman"}
    ax0.set_title("Sensor", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, data_filter, color="#B0BEC5", label="Raw", zorder=1)
    ax0.plot(
        x_axis, data_clean, color="#FFC107", label="Cleaned", zorder=1, linewidth=1.5
    )
    ax0.legend(loc="upper right", frameon=True, prop=legend_font)


def plot_abs_and_mvc(data_abs, data_mvc, k):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax0.set_xlabel("Time (seconds)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
    fig.suptitle("Absolute Value and MVC of EMG signals", fontweight="bold",
                 fontdict={'family': 'Times New Roman'}, fontsize=16)
    plt.subplots_adjust(hspace=0.2)
    x_axis = np.linspace(0, data_abs.shape[0] / int(2000 / k), data_abs.shape[0])
    legend_font = {"family": "Times New Roman"}
    ax0.set_title("Sensor", fontdict={'family': 'Times New Roman'}, fontsize=12)
    ax0.plot(x_axis, data_abs, color="#B0BEC5", label="ABS", zorder=1)
    ax1 = ax0.twinx()
    ax1.plot(
        x_axis, data_mvc, color="#FA6839", label="MVC", linewidth=1.5
    )
    ax0.legend(loc="upper left", frameon=True, prop=legend_font)
    ax1.legend(loc="upper right", frameon=True, prop=legend_font)


if __name__ == '__main__':
    emg = np.load('./data/emg_data.npy')
    SAMPING_RATE = 2000
    k = 4
    n = 4
    data_filter, data_clean, data_mvc, data_abs = process_all_channels(emg, n, SAMPING_RATE, k)

    for i in range(n):
        plot_raw_and_clean(data_filter[:, i], data_clean[:, i], k)
        plot_abs_and_mvc(data_abs[:, i], data_mvc[:, i], k)
    plt.show()

    # # process single channel
    # data_filter_1, data_clean_1, data_mvc_1, data_abs_1 = process(emg[:, 0], SAMPING_RATE, n)
    # data_filter_2, data_clean_2, data_mvc_2, data_abs_2 = process(emg[:, 1], SAMPING_RATE, n)
