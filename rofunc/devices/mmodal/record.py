import sys
from time import time 
sys.path.append("..") 
from xsens import XsensInterface
from optitrack import opti_run
from pynput.keyboard import Key, Listener


event = threading.Event()


class multimodal(object):

    def __init__(self) -> None:
        self.xsens = XsensInterface(ip, port, ref_frame=ref_frame)
        self.optitrack = OptiTrackClient()

    

    def save_file_thread(self, root_dir: str, exp_name: str) -> None:
        """
        save xsens motion data to the file
        Args:
            root_dir: root dictionary
            exp_name: dictionary saving the npy file, named according to time
        Returns:
            None
        """
        xsens_data = []
        while True:
            data = self.get_datagram()
            print(type(data))
            if type(data) == list:
                print(data)
                xsens_data.append(data)
            if event.isSet():
                np.save(root_dir + '/' + exp_name + '/' + 'xsens_data.npy', np.array(xsens_data))
                break


    def time_sync(self):
        """
        synchronize multi-modal data
        """
        self.data = 
        

def on_press(key):
    # 当按下esc，结束监听
    if key == Key.esc:
        event.set()
        print(f"你按下了esc，监听结束")
        return False