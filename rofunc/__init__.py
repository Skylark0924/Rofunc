import os
import warnings

import shutup

shutup.please()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

# try:
#     import pbdlib
# except ImportError:
#     print("pbdlib is not installed. Install it automatically...")
#     pip.main(
#         ['install', 'https://github.com/Skylark0924/Rofunc/releases/download/v0.0.2.3/pbdlib-0.1-py3-none-any.whl'])

from .devices import zed, xsens, optitrack, mmodal, emg
from . import simulator as sim
from .learning import RofuncML, RofuncIL, RofuncRL
from .planning_control import lqt, lqr
from .utils import visualab, robolab, logger, oslab, maniplab
from .utils.robolab import ergonomics
from .utils.datalab import primitive, data_generator
from . import config

from .learning.RofuncML import tpgmm, gmr, tpgmr
