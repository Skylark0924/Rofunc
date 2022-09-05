import socket  # Allows us to communicate via TCP/IP with the Delsys
import sys  # Abstractions for various low-level system functions...
import struct  # Allows us to parse binary data received from Delsys
import threading  # Enables multithreading
import time  # Allows for precise time-stamping of Delsys data
import csv  # Allows easy writing of Delsys data to CSV files
import datetime  # Allows us to set correct date & time in filenames
import os  # Abstractions for various low-level file system/OS operations
import json  # Allows for easy parsing of JSON files to dicts
import collections  # Provides instance of orderd dictionary


def command_escaper(command):
    """All commands sent to the Delsys controller must be escaped properly"""
    command = command + b'\r\n\r\n'
    # print(command)
    return command


def bytes_to_raw_array(data, format_str):
    """Convert raw byte data from socket connection to floating point numbers"""
    return struct.unpack(format_str, data)


def filename_formatter(sensor_class):
    """Generates filenames from an instance of a sensor class.
        NOT used in the current implementation"""
    f = {}
    for i in sensor_class.getDataStreams():
        f[i] = '{} {} {}.csv'.format(str(datetime.date.today()),
                                     sensor_class.getName(),
                                     i)
    return f


def open_JSON_settings():
    """This function opens, reads, and returns a settings dict from the Delsys JSON preference file"""
    file_list = os.listdir('./')
    file_list = [x for x in file_list if x.startswith('delsys') or x.startswith('Delsys')]
    if len(file_list) > 1 or not any(file_list):
        print('Unable To Auto-Import System Settings', end='\n')
        file_list = input('Please Input Filename for Delsys Parameter JSON -->\n')
        if not file_list:
            print('Filename Not Valid', file=sys.stderr)
            sys.exit(1)
    if isinstance(file_list, list):
        file_list = file_list[0]
    try:
        with open(file_list) as f:
            json_params = json.load(f, object_pairs_hook=collections.OrderedDict)
    except FileNotFoundError:
        print('Preference File Not Found', file=sys.stderr)
        sys.exit(1)
    return json_params


def json_parser(json_setting_file):
    # First we extract the types of data streams each sensor has

    streams = json_setting_file['data_streams']

    # Next, we extract the dicts which contain our sensor info

    sensors = json_setting_file['sensors']

    # Then we extract socket addresses

    socket_addresses = json_setting_file['addresses']

    # Extract the names of our sensors

    sensor_names = json_setting_file['sensors'].keys()
    print(sensor_names)

    # Create a dictionary that will contain attributes for our DataStreamer classes

    sensor_attributes = {}

    # Pre-populate the dictionary with the proper data structures for appending later

    for i in streams:
        sensor_attributes[i] = {
            'address': socket_addresses[i],
            'byte_size': 0,
            'sensor_IDs': sensor_names,  # For now, we will assume that there is a consistent pairing of
            # sensor with data streams
            'channel_labels': [],
            'unpack_template': ''
        }

    # Now unpack each sensor's data into our data structure

    for i in streams:
        for j in sensor_names:
            sensor_attributes[i]['byte_size'] += sensors[j]['byte_count'][i]
            sensor_attributes[i]['channel_labels'].extend(sensors[j]['channel_labels'][i])
            sensor_attributes[i]['unpack_template'] += sensors[j]['mask'][i]

    return socket_addresses, sensor_attributes


def create_filnames(sensor_types):
    """Creates timestamped filenames for EMG and accelerometer data streams"""
    fnames = {}
    date_and_time = str(datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d %X"))
    for i in sensor_types:
        fnames[i] = ("{} {}.csv".format(date_and_time, i))


def open_fds(filename_list):
    """Converts filenames to actual file descriptors used for writing"""
    fd_out = {}
    for key, value in filename_list:
        fd_out[key] = open(value, 'w', newline='')
    return fd_out


def close_fds(fd_list):
    """Closes file descriptors at the end of an experiment"""
    for i in fd_list:
        i.close()


class Controller(threading.Thread):
    start_str = b'START'
    stop_str = b'STOP'

    def __init__(self, address=None, run_duration=None, is_daemon=False):
        # Note: Run Duration Must Be Provided in Seconds
        super().__init__(daemon=is_daemon)
        self.address = address
        self.run_duration = run_duration
        self.sock = None

    def run(self):
        self.start_socket()
        self.sock.send(command_escaper(Controller.start_str))
        init_time = time.time()
        read_out = ""
        while (time.time() - init_time) < self.run_duration:
            try:
                read_out = self.sock.recv(
                    100).decode()  # Note: this code and the timeout are a very messy way to do this. We will eventually use selectors
            except socket.timeout:
                pass
            if read_out:
                print(read_out)
                read_out = ""
            time.sleep(.01)
        self.sock.send(command_escaper(Controller.stop_str))
        self.sock.close()
        '''We sleep for 5 seconds before exiting the thread 
        to allow sufficient time for all data streamer sockets to ingest data'''
        time.sleep(5)

    def start_socket(self):
        self.sock = socket.socket()
        self.sock.connect(self.address)
        self.sock.settimeout(.001)
        # self.sock.setblocking()


class DataStreamer(threading.Thread):
    def __init__(self, file_descriptor=None, address=None, byte_size=None, sensor_IDs=None, channel_labels=None,
                 unpack_template=None, is_daemon=True):
        super().__init__(daemon=is_daemon)
        self.fd = file_descriptor
        self.sock = None  # Socket will be instantiated once thread starts running.
        self.address = address
        self.byte_size = byte_size
        self.sensor_IDs = sensor_IDs
        self.num_sensors = len(sensor_IDs)
        self.csv_multiplier = self.num_sensors * len(channel_labels)
        self.channel_labels = channel_labels
        self.unpack_template = unpack_template
        self.csv_writer_instance = None

    def run(self):
        self.start_socket()
        self.prep_csv_writer()
        while True:
            data_in = self.sock.recv(self.byte_size)
            if len(data_in) != self.byte_size:
                break  # If we received a truncated packet, then break.
            time_in = time.time()
            raw_floats = struct.unpack(self.unpack_template, data_in)
            self.write_data(raw_floats, time_in)

    def prep_csv_writer(self):
        self.csv_writer_instance = csv.writer(self.fd)
        self.csv_writer_instance.writerow(['sensor_name', 'packet_time', 'sensor_dtype', 'sensor_value'])

    def write_data(self, data, time_stamp):
        self.csv_writer_instance.writerows(zip([self.sensor_IDs] * self.csv_multiplier,
                                               [time_stamp] * self.csv_multiplier,
                                               [self.channel_labels] * self.num_sensors,
                                               data))

    def start_socket(self):
        self.sock = socket.socket()
        self.sock.connect(self.address)


class Sensor:
    """This class implements an abstraction for sensors.
        It is not used in the current implementation"""

    def __init__(self, name, data_streams, byte_sizes, masks):
        self._name = name
        self._data_streams = data_streams
        self._byte_sizes = byte_sizes
        self._masks = masks

    def getMask(self):
        return self._masks

    def getBytes(self):
        return self._byte_sizes

    def getName(self):
        return self._name

    def getDataStreams(self):
        return self._data_streams

    def generateFilnames(self):  # This method might be deprecated in favor
        return filename_formatter(self)  # of a different methodology for creating filenames


def main():
    # Main is currently written purely for testing sockets
    # I will implement the actual main() tomorrow...
    test_class = Controller(('10.13.52.82', 50040), 15, True)
    test_class.start()
    test_class.join()
    sys.exit(0)


if __name__ == '__main__':
    main()
