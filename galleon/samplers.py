import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.stats import truncnorm
from figaro.utils import rejection_sampler

from galleon.settings import omega, p_w_1det, p_w_3det, snr_th, sigma_Mc, sigma_eta, sigma_w

def w_from_interpolant(n_draws, n_det = 'one'):
    """
    Sampler for w = Theta/4 (extrinsic parameter).
    
    Arguments:
        :int n_draws: number of draws
        :str n_det:   number of detectors ('one' or 'three')
    
    Returns:
        :np.ndarray: w samples
    """
    if n_det == 'one':
        pdf = p_w_1det
        bounds = [0.,1.]
    elif det == 'three':
        pdf = p_w_3det
        bounds = [0.,1.4]
    else:
        raise ValueError("Invalid n_det. Please provide 'one' or 'three'.")
    return rejection_sampler(int(n_draws), pdf, bounds)

def snr_sampler(true_snr):
    """
    Samples observed SNRs given a set of true SNRs.
    
    Arguments:
        :np.ndarray true_snr: true SNRs obtained as w*SNR_opt
    
    Returns:
        :np.ndarray: observed SNRs
    """
    return true_snr + np.random.normal(loc = 0., scale = 1., size = len(true_snr))

def Mc_sampler(Mc, snr_obs, n_draws = 1e3):
    """
    Produces detector-frame chirp mass posterior samples given true values.
    
    Arguments:
        :np.ndarray Mc:      true detector-frame chirp mass values
        :np.ndarray snr_obs: observed SNRs
        :int n_draws:        number of posterior samples to draw for each event
    
    Returns:
        :np.ndarray: set of single-event posterior samples
    """
    max_L_Mc = np.random.normal(loc = np.log(Mc), scale = sigma_Mc/snr_obs)
    return np.array([np.exp(np.random.normal(loc = Mc_i, scale = sigma_Mc/snr_i, size = int(n_draws))) for Mc_i, snr_i in zip(max_L_Mc, snr_obs)])

def eta_sampler(eta, snr_obs, n_draws = 1e3):
    """
    Produces symmetric mass ratio posterior samples given true values.
    
    Arguments:
        :np.ndarray eta:     true symmetric mass ratio values
        :np.ndarray snr_obs: observed SNRs
        :int n_draws:        number of posterior samples to draw for each event
    
    Returns:
        :np.ndarray: set of single-event posterior samples
    """
    # Bounds
    lowclip = -eta/(sigma_eta/snr_obs)
    highclip = (0.25-eta)/(sigma_eta/snr_obs)
    # Maximum likelihood value
    max_L_eta = truncnorm(a = lowclip, b = highclip, loc = eta, scale = sigma_eta/snr_obs).rvs()
    # Sampling
    samples = np.zeros(shape = (len(eta), int(n_draws)))
    for i, (eta_i, snr_i) in enumerate(zip(max_L_eta, snr_obs)):
        lowclip = -eta_i/(sigma_eta/snr_i)
        highclip = (0.25-eta_i)/(sigma_eta/snr_i)
        samples[i] = truncnorm(a = lowclip, b = highclip, loc = eta_i, scale = sigma_eta/snr_i).rvs(int(n_draws))
    return samples

def w_sampler(w, snr_obs, n_draws = 1e3):
    """
    Produces w posterior samples given true values.
    
    Arguments:
        :np.ndarray w:       true w values
        :np.ndarray snr_obs: observed SNRs
        :int n_draws:        number of posterior samples to draw for each event
    
    Returns:
        :np.ndarray: set of single-event posterior samples
    """
    # Bounds
    lowclip = -w/(sigma_w/snr_obs)
    highclip = (1.-w)/(sigma_w/snr_obs)
    # Maximum likelihood value
    max_L_w = truncnorm(a = lowclip, b = highclip, loc = w, scale = sigma_w/snr_obs).rvs()
    # Sampling
    samples = np.zeros(shape = (len(w), int(n_draws)))
    for i, (w_i, snr_i) in enumerate(zip(max_L_w, snr_obs)):
        lowclip = -w_i/(sigma_w/snr_i)
        highclip = (1.-w_i)/(sigma_w/snr_i)
        samples[i] = truncnorm(a = lowclip, b = highclip, loc = w_i, scale = sigma_w/snr_i).rvs(int(n_draws))
    return samples
