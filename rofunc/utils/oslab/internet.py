# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import platform
import socket
from contextlib import contextmanager


def get_so_reuseport():
    """
    Get the port with ``SO_REUSEPORT`` flag set.

    :return: port number or None
    """
    try:
        return socket.SO_REUSEPORT
    except AttributeError:
        if platform.system() == "Linux":
            major, minor, *_ = platform.release().split(".")
            if (int(major), int(minor)) > (3, 9):
                # The interpreter must have been compiled on Linux <3.9.
                return 15
    return None


@contextmanager
def reserve_sock_addr():
    """
    Reserve an available TCP port to listen on.

    The reservation is done by binding a TCP socket to port 0 with
    ``SO_REUSEPORT`` flag set (requires Linux >=3.9). The socket is
    then kept open until the generator is closed.

    To reduce probability of 'hijacking' port, socket should stay open
    and should be closed _just before_ starting of ``tf.train.Server``

    Example::

        >>> import os
        >>> from tensorboard import program
        >>> from rofunc.utils.oslab.internet import reserve_sock_addr
        >>> tb = program.TensorBoard()
        >>> # Find a free port
        >>> with reserve_sock_addr() as (h, p):
        ...     argv = ['tensorboard', f"--logdir={os.getcwd()}", f"--port={p}"]
        ...     tb_extra_args = os.getenv('TB_EXTRA_ARGS', "")
        ...     if tb_extra_args:
        ...         argv += tb_extra_args.split(' ')
        ...     tb.configure(argv)
        >>> # Launch TensorBoard
        >>> url = tb.launch()
    """
    so_reuseport = get_so_reuseport()
    if so_reuseport is None:
        raise RuntimeError("SO_REUSEPORT is not supported by the operating system") from None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, so_reuseport, 1)
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.getfqdn(), port)
