import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path

from figaro.load import _find_redshift

from pycbc.waveform import get_fd_waveform
from pycbc.psd.analytical import from_string
from pycbc.filter import sigma

from galleon.settings import d_fid, p_w_1det, p_w_3det

def component_masses(Mc, eta):
    """
    Detector-frame component masses from detector-frame chirp mass and symmetric mass ratio.
    
    Arguments:
        :np.ndarray Mc:  detector-frame chirp mass
        :np.ndarray eta: symmetric mass ratio
    
    Returns:
        :np.ndarray: detector-frame primary mass
        :np.ndarray: detector-frame secondary mass
    """
    M  = Mc/(eta**(3./5.))
    m1 = M*(1 + np.sqrt(1. - 4.*eta))/2.
    m2 = M*(1 - np.sqrt(1. - 4.*eta))/2.
    return m1, m2

def chirp_mass_eta(m1, m2):
    """
    Symmetric mass ratio and detector-frame chirp mass from detector-frame component masses
    
    Arguments:
        :np.ndarray m1: detector-frame primary mass
        :np.ndarray m2: detector-frame secondary mass
    
    Returns:
        :np.ndarray: detector-frame chirp mass
        :np.ndarray: symmetric mass ratio
    """
    Mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    eta = m1*m2/(m1+m2)**2
    return Mc, eta

def snr_optimal(m1, m2, z = None, DL = None, approximant = 'IMRPhenomXPHM', psd = 'aLIGOZeroDetHighPower', deltaf = 1/16., flow = 20.):
    '''
    Compute the SNR from m1, m2, z.
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
    
    for i, (m1i, m2i, zi, Di) in enumerate(zip(m1, m2, z, DL)):
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

def obs_distance(m1_obs, m2_obs, z, w_obs, snr_obs):
    """
    Compute the observed luminosity distance given detector-frame masses, redshift, w and SNR.
    
    Arguments:
        :np.ndarray m1_obs:  detector-frame primary mass
        :np.ndarray m2_obs:  detector-frame secondary mass
        :double z:           true redshift
        :np.ndarray w_obs:   observed w
        :np.ndarray snr_obs: observed SNR
        
    Returns:
        :np.ndarray: observed luminosity distance
    """
    snr_opt = snr_optimal(m1_obs/(1+z), m2_obs/(1+z), np.ones(len(m1_obs))*z, np.ones(len(m1_obs))*d_fid)
    return d_fid*w_obs*snr_opt/snr_obs

def PE_prior(w, DL, volume = 1., n_det  = 'one'):
    """
    Prior used in PE runs.
    
    Arguments:
        :np.ndarray w:  observed w parameter
        :np.ndarray DL: observed luminosity distance
        :double volume: normalising volume
        :str n_det:     number of detectors ('one' or 'three')
    
    Returns:
        :np.ndarray: prior
    """
    if n_det == 'one':
        pdf = p_w_1det
    elif det == 'three':
        pdf = p_w_3det
    else:
        raise ValueError("Invalid n_det. Please provide 'one' or 'three'.")
    return pdf(w)*DL**2/volume

def jacobian(m1, m2, DL, z, w):
    """
    Jacobian of the coordinate change from (snr, Mc, eta, w) to (m1, m2, DL, w).
    
    Arguments:
        :np.ndarray m1: detector-frame primary mass
        :np.ndarray m2: detector-frame secondary mass
        :np.ndarray DL: luminosity distance
        :np.ndarray z:  redshift corresponding to DL
        :np.ndarray w:  w parameter
    
    Returns:
        :np.ndarray: jacobian
    """
    snr_opt = snr_optimal(m1/(1+z), m2/(1+z), np.ones(len(m1))*z, np.ones(len(m1))*d_fid)
    Mc, eta = chirp_mass_eta(m1, m2)
    return w*snr_opt*(d_fid/DL**2)*((m1-m2)/(m1+m2)**2)*(eta**(3./5.))

def save_event(m1, m2, Mc, q, z, DL, snr, name = 'injections', out_folder = '.'):
    """
    Save samples to h5 file, formatted using the same convention as LVK.
    
    Arguments:
        :np.ndarray m1:          source-frame primary masses
        :np.ndarray m2:          source-frame secondary masses
        :np.ndarray Mc:          source-frame chirp masses
        :np.ndarray q:           mass ratios
        :np.ndarray z:           redshifts
        :np.ndarray DL:          luminosity distances
        :np.ndarray snr:         observed SNRs
        :str name:               file name
        :str or Path out_folder: folder
    """
    file = Path(out_folder, name + '.h5')
    with h5py.File(file, 'w') as f:
        ps = f.create_group('MDC/posterior_samples')
        # Dictionary
        dict_v = {'mass_1_source': m1,
                  'mass_2_source': m2,
                  'mass_1': m1*(1+z),
                  'mass_2': m2*(1+z),
                  'chirp_mass_source': Mc,
                  'total_mass_source': m1+m2,
                  'mass_ratio': q,
                  'redshift': z,
                  'luminosity_distance': DL,
                  'snr': snr,
                  }
        for key, value in dict_v.items():
            ps.create_dataset(key, data = value)

def save_posteriors(m1, m2, Mc, q, z, DL, snr, name = 'injections', out_folder = '.'):
    """
    Save posterior samples to h5 files, formatted using the same convention as LVK.
    
    Arguments:
        :np.ndarray m1:          source-frame primary mass sets
        :np.ndarray m2:          source-frame secondary mass sets
        :np.ndarray Mc:          source-frame chirp mass sets
        :np.ndarray q:           mass ratio sets
        :np.ndarray z:           redshift sets
        :np.ndarray DL:          luminosity distance sets
        :np.ndarray snr:         observed SNRs
        :str name:               catalog name
        :str or Path out_folder: folder
    """
    for i, (m1i, m2i, Mci, qi, zi, DLi, snri) in tqdm(enumerate(zip(m1, m2, Mc, q, z, DL, snr)), total = len(m1), desc = 'Saving'):
        file_name = '{0}_{1}'.format(name, i+1)
        save_event(m1i, m2i, Mci, qi, zi, DLi, snri*np.ones(len(m1i)), file_name, out_folder)
