import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.stats import truncnorm

from pycbc.waveform import get_fd_waveform
from pycbc.psd.analytical import from_string
from pycbc.filter import sigma

from figaro.utils import rejection_sampler
from figaro.load import _find_redshift

from galleon.settings import omega, p_w_1det, p_w_3det, rho_th, sigma_Mc, sigma_eta, sigma_w

def w_sampler(n_draws, n_det = 'one'):
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

# Optimal SNR
def snr_optimal(m1, m2, z = None, DL = None, approximant = 'IMRPhenomXPHM', psd = 'aLIGOZeroDetHighPower', deltaf = 1/40., flow = 10.):
    '''
    Compute the SNR from m1,m2,z
    Follows Davide Gerosa's code (https://github.com/dgerosa/gwdet).
    
    Arguments:
        :np.ndarray m1:   source-frame primary masses
        :np.ndarray m2:   source-frame secondary masses
        :np.ndarray z:    source redshifts
        :str approximant: waveform approximant
        :str psd:         psd to be used
        :double deltaf:   frequency bin
        :double flow:     lower frequency
    
    Returns:
        :np.ndarray: optimal SNRs
    '''
    snr = np.zeros(len(m1))
    if DL is None and z is None:
        raise ValueError("Please provide DL and/or z.")
    if DL is None:
        DL = omega.LuminosityDistance(z)
    if z is None:
        z = np.array([_find_redshift(omega, d) for d in DL])
    
    for i, (m1i, m2i, zi, Di) in tqdm(enumerate(zip(m1, m2, z, DL)), total = len(m1), desc = 'Optimal SNR'):
        # FIXME: add spin parameters!
        hp, hc = get_fd_waveform(approximant = approximant,
                                 mass1       = m1i*(1.+zi),
                                 mass2       = m2i*(1.+zi),
                                 delta_f     = deltaf,
                                 f_lower     = flow,
                                 distance    = Di,
                                 )
                                 
        evaluatedpsd = from_string(psd,
                                   len(hp),
                                   deltaf,
                                   flow,
                                   )
                                   
        # Keep only hp polarisation (face-on binary)
        snr[i] = sigma(hp, psd=evaluatedpsd, low_frequency_cutoff=flow)
    return snr

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
    # Bounds
    lowclip = -eta/(sigma_eta*snr_obs)
    highclip = (0.25-eta)/(sigma_eta*snr_obs)
    # Maximum likelihood value
    max_L_eta = truncnorm(a = lowclip, b = highclip, loc = eta, scale = sigma_eta*snr_obs).rvs()
    # Sampling
    samples = np.zeros(shape = (len(eta), int(n_draws)))
    for i, (eta_i, snr_i) in enumerate(zip(eta, snr_obs)):
        lowclip = -eta_i/(sigma_eta*snr_i)
        highclip = (0.25-eta_i)/(sigma_eta*snr_i)
        samples[i] = truncnorm(a = lowclip, b = highclip, loc = eta_i, scale = sigma_eta*snr_i).rvs(int(n_draws))
    return samples

def theta_sampler(theta, snr_obs, n_draws = 1e3):
    # Bounds
    lowclip = -theta/(sigma_theta*snr_obs)
    highclip = (1.-theta)/(sigma_theta*snr_obs)
    # Maximum likelihood value
    max_L_theta = truncnorm(a = lowclip, b = highclip, loc = theta, scale = sigma_theta*snr_obs).rvs()
    # Sampling
    samples = np.zeros(shape = (len(theta), int(n_draws)))
    for i, (theta_i, snr_i) in enumerate(zip(theta, snr_obs)):
        lowclip = -theta_i/(sigma_theta*snr_i)
        highclip = (1.-theta_i)/(sigma_theta*snr_i)
        samples[i] = truncnorm(a = lowclip, b = highclip, loc = theta_i, scale = sigma_theta*snr_i).rvs(int(n_draws))
    return samples
