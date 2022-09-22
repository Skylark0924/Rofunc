import sys
import time
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData
from multiprocessing import Process, Manager
import socket


# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.
def receive_new_frame(data_dict):
    # order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
    #               "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedMdelsChangedo"]
    dump_args = False
    if dump_args:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict:
                out_string += data_dict[key] + " "
            out_string += "/"
        print(out_string)


# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame(new_id, position, rotation):
    pass


def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


# keep receiving msg from optitrack stream
def receive_from_optitrack(rbd_dict, connection):
    recv_buffer_size = 1024 * 1024
    offset = 4
    major = connection.get_major()
    minor = connection.get_minor()

    while True:
        in_socket = connection.data_socket
        data, addr = in_socket.recvfrom(recv_buffer_size)
        packet_size = int.from_bytes(data[2:4], byteorder="little")
        offset_tmp, mocap_data = connection.unpack_mocap_data(
            data[offset:], packet_size, major, minor
        )
        data_rbd = mocap_data.rigid_body_data
        data_rbd = data_rbd.get_as_string()
        data_rbd = data_rbd.encode("utf-8")
        rbd_dict["data_rbd"] = data_rbd


def send_to_ubuntu(server, rbd_dict, connection):
    while True:
        try:
            data_rbd = rbd_dict["data_rbd"]
            connection.sendall(data_rbd)
            rec = connection.recv(64)
        except KeyError:
            print(
                "Rigid body dict has no data_rbd key, maybe optitrack stream is not alive"
            )
            time.sleep(1.0)
        except ConnectionError:
            print("The connection has lost, waiting for reconnection ...")
            connection, (host, port) = server.accept()


if __name__ == "__main__":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("192.168.13.118", 6688))
    server.listen(5)
    connect, (host, port) = server.accept()
    print("the client %s:%s has connected." % (host, port))

    optionsDict = {
        "clientAddress": "127.0.0.1",
        "serverAddress": "127.0.0.1",
        "use_multicast": True,
    }

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame
    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.Creat_data_command_socket()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")
    time.sleep(1)

    with Manager() as manager:
        dict_share = manager.dict()
        optitrack_stream_process = Process(
            target=receive_from_optitrack,
            args=(
                dict_share,
                streaming_client,
            ),
        )
        send2client_process = Process(
            target=send_to_ubuntu,
            args=(
                server,
                dict_share,
                connect,
            ),
        )
        optitrack_stream_process.start()
        send2client_process.start()
        optitrack_stream_process.join()
        send2client_process.join()
