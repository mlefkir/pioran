from jax.config import config
config.update("jax_enable_x64", True)

from .core import GaussianProcess

from .acvf_base import CovarianceFunction
from .psd_base import PowerSpectralDensity
from .psdtoacv import PSDToACV

from .inference import Inference
from .plots import plot_prediction, plot_residuals

from .simulate import Simulations

__author__ = "Mehdy Lefkir"
__version__ = "0.1.0"
