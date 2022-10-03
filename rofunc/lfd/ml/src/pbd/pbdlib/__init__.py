from .functions import *

from .gmm import GMM
from .gmr import GMR
from .hmm import HMM
from .hsmm import HSMM
from .model import Model
from .mvn import *
from .plot import *
from .poglqr import PoGLQR, LQR, GMMLQR, BiPoGLQR
from .mtmm import MTMM, VBayesianGMM, VMBayesianGMM, VBayesianHMM
from .dmp import DMP
# from .vhmm import BayesianMarkovianGaussianMixture

try:
	import gui
except ImportError as e:
	print("Could not import gui: {0}".format(e.msg))
	print("run : sudo apt-get install tkinter")
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

from . import utils
from . import plot

