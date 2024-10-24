#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com


import os


class BeautyLogger:
    """
    Lightweight logger for Rofunc package.
    """

    def __init__(self, log_path: str, log_name: str = 'rofunc.log', verbose: bool = True):
        """
        Lightweight logger for Rofunc package.

        Example::

            >>> from rofunc.utils.logger import BeautyLogger
            >>> logger = BeautyLogger(log_path=".", log_name="rofunc.log", verbose=True)

        :param log_path: the path for saving the log file
        :param log_name: the name of the log file
        :param verbose: whether to print the log to the console
        """
        self.log_path = log_path
        self.log_name = log_name
        self.verbose = verbose

    def _write_log(self, content, type):
        with open(os.path.join(self.log_path, self.log_name), "a") as f:
            f.write("[Rofunc:{}] {}\n".format(type.upper(), content))

    def warning(self, content, local_verbose=True):
        """
        Print the warning message.

        Example::

            >>> logger.warning("This is a warning message.")

        :param content: the content of the warning message
        :param local_verbose: whether to print the warning message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="warning")
        self._write_log(content, type="warning")

    def module(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.module("This is a module message.")

        :param content: the content of the module message
        :param local_verbose: whether to print the module message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.info("This is a info message.")

        :param content: the content of the info message
        :param local_verbose: whether to print the info message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")


def beauty_print(content, type=None):
    """
    Print the content with different colors.

    Example::

        >>> import rofunc as rf
        >>> rf.logger.beauty_print("This is a warning message.", type="warning")

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error"
    :return:
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m[Rofunc:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
        print("\033[1;33m[Rofunc:MODULE] {}\033[0m".format(content))  # For a new module (light yellow)
    elif type == "info":
        print("\033[1;35m[Rofunc:INFO] {}\033[0m".format(content))  # For info (light purple)
    elif type == "error":
        print("\033[1;31m[Rofunc:ERROR] {}\033[0m".format(content))  # For error (red)
    else:
        raise ValueError("Invalid level")
