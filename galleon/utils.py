import numpy as np
from figaro.load import _find_redshift

from pycbc.waveform import get_fd_waveform
from pycbc.psd.analytical import from_string
from pycbc.filter import sigma

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
