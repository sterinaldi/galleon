import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.stats import norm, truncnorm
from figaro.utils import rejection_sampler

from galleon.settings import omega, p_w_1det, p_w_3det, snr_th, sigma_Mc, sigma_q, sigma_w

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
    elif n_det == 'three':
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
    return np.exp(norm(loc = np.log(Mc), scale = sigma_Mc/snr_obs).rvs(size = (int(n_draws), len(Mc)))).T

def q_sampler(q, snr_obs, n_draws = 1e3):
    """
    Produces mass ratio posterior samples given true values.
    
    Arguments:
        :np.ndarray eta:     true symmetric mass ratio values
        :np.ndarray snr_obs: observed SNRs
        :int n_draws:        number of posterior samples to draw for each event
    
    Returns:
        :np.ndarray: set of single-event posterior samples
    """
    q_samples = truncnorm(a = -q/(q*sigma_q/snr_obs), b = (1.-q)/(q*sigma_q/snr_obs), loc = q, scale = q*sigma_q/snr_obs).rvs((int(n_draws),len(q))).T
    return q_samples

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
    return truncnorm(a = lowclip, b = highclip, loc = w, scale = sigma_w/snr_obs).rvs((int(n_draws),len(w))).T
