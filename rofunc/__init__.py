import os
import warnings
import shutup

shutup.please()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

from .devices import zed, xsens, optitrack, mmodal, emg
from .simulator import franka, curi, walker, curi_mini, object
from .lfd import ml, dl, rl
from .planning import lqt, lqr
from .utils import visualab, robolab, data_generator, primitive, logger, file
from . import config

from .lfd.ml import tpgmm, gmr, tpgmr
from .lfd.dl import bc, strans
from .lfd.rl.online import dqn
