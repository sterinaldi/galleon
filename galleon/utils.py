import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path

from pycbc.waveform import get_fd_waveform
from pycbc.psd.analytical import from_string
from pycbc.filter import sigma

from galleon.settings import d_fid, z_fid, p_w_1det, p_w_3det
from galleon.samplers import snr_sampler

def q_from_eta(eta):
    return ((1-2*eta) - np.sqrt((2*eta-1)**2 - 4*eta**2))/(2*eta)

def component_masses(Mc, q):
    """
    Detector-frame component masses from detector-frame chirp mass and symmetric mass ratio.
    
    Arguments:
        :np.ndarray Mc:  detector-frame chirp mass
        :np.ndarray q:   mass ratio
    
    Returns:
        :np.ndarray: detector-frame primary mass
        :np.ndarray: detector-frame secondary mass
    """
    eta = q/((1+q)**2)
    M   = Mc/(eta**(3./5.))
    m1  = M/(1+q)
    m2  = m1*q
    return m1, m2

def chirp_mass_q(m1, m2):
    """
    Mass ratio and detector-frame chirp mass from detector-frame component masses
    
    Arguments:
        :np.ndarray m1: detector-frame primary mass
        :np.ndarray m2: detector-frame secondary mass
    
    Returns:
        :np.ndarray: detector-frame chirp mass
        :np.ndarray: mass ratio
    """
    Mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    q  = m2/m1
    return Mc, q

def chirp_mass_eta(m1, m2):
    """
    Symmetric mass ratio and detector-frame chirp mass from detector-frame component masses
    
    Arguments:
        :np.ndarray m1: detector-frame primary mass
        :np.ndarray m2: detector-frame secondary mass
    
    Returns:
        :np.ndarray: detector-frame chirp mass
        :np.ndarray: mass ratio
    """
    Mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    eta  = m1*m2/((m1+m2)**2)
    return Mc, eta

def snr_optimal(m1z, m2z, z = None, DL = None, approximant = 'IMRPhenomXPHM', psd = 'aLIGOZeroDetHighPower', deltaf = 1/16., flow = 20., bounds_m = None, sensitivity = False):
    '''
    Compute the SNR from m1, m2, z.
    Follows Davide Gerosa's code (https://github.com/dgerosa/gwdet).
    
    Arguments:
        :np.ndarray m1z:   detector-frame primary masses
        :np.ndarray m2z:   detector-frame secondary masses
        :np.ndarray z:    source redshifts
        :str approximant: waveform approximant
        :str psd:         psd to be used
        :double deltaf:   frequency bin
        :double flow:     lower frequency
        :iter bounds_m:   mass bounds
    
    Returns:
        :np.ndarray: optimal SNRs
    '''
    snr = np.zeros(len(m1z))
    if DL is None and z is None:
        raise ValueError("Please provide DL and/or z.")
    if DL is None:
        DL = omega.LuminosityDistance(z)
    if z is None:
        z = omega.Redshift(DL)
    
    if sensitivity:
        loop = tqdm(enumerate(zip(m1z, m2z, z, DL)), desc = 'Generating sensitivity estimate', total = len(m1z))
    else:
        loop = enumerate(zip(m1z, m2z, z, DL))
    
    for i, (m1zi, m2zi, zi, Di) in loop:
        # FIXME: add spin parameters!
        # Check WF validity
        if approximant == 'IMRPhenomXPHM':
            if m1zi/m2zi > 20.:
                continue
        hp, hc = get_fd_waveform(approximant = approximant,
                                 mass1       = m1zi,
                                 mass2       = m2zi,
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

def obs_distance_snr(m1_obs, m2_obs, w_obs, snr_obs, bounds_m = None):
    """
    Compute the observed luminosity distance given detector-frame masses, redshift, w and SNR.
    
    Arguments:
        :np.ndarray m1_obs:  detector-frame primary mass
        :np.ndarray m2_obs:  detector-frame secondary mass
        :np.ndarray w_obs:   observed w
        :np.ndarray snr_obs: observed SNR
        :iter bounds_m:   mass bounds
        
    Returns:
        :np.ndarray: observed luminosity distance
        :np.ndarray
    """
    snr_opt = snr_optimal(m1_obs, m2_obs, z = np.ones(len(m1_obs))*z_fid, DL = np.ones(len(m1_obs))*d_fid)
    dl_obs = d_fid*w_obs*snr_opt/snr_obs
    return dl_obs, snr_opt
    
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
    elif n_det == 'three':
        pdf = p_w_3det
    else:
        raise ValueError("Invalid n_det. Please provide 'one' or 'three'.")
    return pdf(w)*DL**2/volume

def jacobian(m1, m2, DL, z, w, snr):
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
    snr_opt = snr*d_fid/DL
    Mc, eta = chirp_mass_eta(m1, m2)
    return w*snr_opt*(d_fid/DL**2)*Mc/(m1**2)#((m1-m2)/(m1+m2)**2)*(eta**(3./5.))

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

def save_injections(m1, m2, Mc, q, z, DL, snr, p_m1, p_m2, p_z, p_dl, n_total, name = 'injections', out_folder = '.'):
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
        ps = f.create_group('injections')
        ps.attrs['total_generated'] = n_total
        ps.attrs['analysis_time_s'] = (60.*60.*24.*365)
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
                  'mass1_source_sampling_pdf': p_m1,
                  'mass1_source_mass2_source_sampling_pdf': p_m1*p_m2,
                  'redshift_sampling_pdf': p_z,
                  'luminosity_distance_sampling_pdf': p_dl,
                  'snr': snr,
                  }
        for key, value in dict_v.items():
            ps.create_dataset(key, data = value)
