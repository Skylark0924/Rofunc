import os
import warnings
import shutup

shutup.please()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

from .devices import zed, xsens, optitrack, mmodal, emg
from . import simulator as sim
from .learning import ml, dl, rl
from .planning_control import lqt, lqr
from .utils import visualab, robolab, data_generator, primitive, logger, file
from . import config

from .learning.ml import tpgmm, gmr, tpgmr
from .learning.dl import bc, strans
from .learning.rl import PPOAgent
