import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from figaro.cosmology import CosmologicalParameters

# Cosmology (Planck 2018)
omega = CosmologicalParameters(0.674, 0.315, 0.685, -1., 0.)

# w = Theta/4 interpolant (from https://arxiv.org/pdf/1405.7016.pdf - Appendix)
# 1 detector
data_P_w_1det = np.genfromtxt(Path(Path(__file__).resolve().parent, 'interpolants/Pw_single.dat'), names = ['w', 'p_w']).T
p_w_1det      = UnivariateSpline(data_P_w_1det['w'], data_P_w_1det['p_w'][::-1], s = 0).derivative()

# 3 detectors
data_P_w_3det = np.genfromtxt(Path(Path(__file__).resolve().parent, 'interpolants/Pw_three.dat'), names = ['w', 'p_w']).T
p_w_3det      = UnivariateSpline(data_P_w_3det['w'], data_P_w_3det['p_w'][::-1], s = 0).derivative()

# Parameters (from Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf)
snr_th    = 8.
sigma_Mc  = 0.08*snr_th
sigma_eta = 0.022*snr_th
sigma_w   = 0.21*snr_th
d_fid     = 300
