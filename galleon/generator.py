import numpy as np
from figaro.utils import rejection_sampler
from figaro.load import _find_redshift

from galleon.samplers import *
from galleon.utils import *
from galleon.settings import omega

mass_parameters_list = ['m1m2', 'm1q']

class Generator:
    def __init__(self, bounds_m  = [5,100],
                       bounds_z  = None,
                       bounds_d  = None,
                       bounds_q  = None,
                       n_det     = 'one',
                       mass_pars = 'm1q',
                       dist_par  = 'z',
                       snr_cut   = 8.,
                       ):
        """
        Class to generate posterior distributions.
        
        Arguments:
            :double mmin:    minimum BH mass
            :double mmax:    maximum BH mass
            :double zmin:    minimum redshift
            :double zmax:    maximum redshift
            :str n_det:      number of detectors ('one' or 'three')
            :str mass_pars:  set of parameters used to generate the masses ('m1m2', 'm1q')
            :str dist_par:   set of parameters used to generate the distance ('z', 'DL')
            :double snr_cut: SNR for detectability. 8 is default for LVK, 0 (or negative values) removes the filter.
        
        Returns:
            :Generator: instance of Generator class
        """
        self.bounds_m = bounds_m
        if bounds_q is not None:
            self.bounds_q = bounds_q
        else:
            self.bounds_q = [0.2,1.]
        if bounds_z is not None:
            self.bounds_z = bounds_z
            self.bounds_d = [omega.LuminosityDistance_double(self.bounds_z[0]), omega.LuminosityDistance_double(self.bounds_z[1])]
        else:
            self.bounds_z = [0.001, 1.]
        if bounds_d is not None:
            self.bounds_d = bounds_d
            self.bounds_z = [_find_redshift(omega, bounds_d[0]), _find_redshift(omega, bounds_d[1])]
        else:
            self.bounds_d = [omega.LuminosityDistance_double(self.bounds_z[0]), omega.LuminosityDistance_double(self.bounds_z[1])]
        
        self.volume = (self.bounds_d[1]**3 - self.bounds_d[0]**3)/3.
        self.dist_par = dist_par
        if mass_pars in mass_parameters_list:
            self.mass_pars = mass_pars
        else:
            raise ValueError("Mass parameters not supported. Please provide one of the following: "+" ".join(mass_parameters_list))
        self.n_det   = n_det
        self.snr_cut = snr_cut
    
    def mass_distribution(self, m):
        """
        Uniform mass distribution. Replace in child classes.
        
        Arguments:
            :np.ndarray m: mass values
        
        Returns:
            :np.ndarray: probabilities
        """
        return np.ones(np.shape(m))/(self.bounds_m[1] - self.bounds_m[0])
    
    def mass_ratio_distribution(self, q):
        """
        Uniform mass ratio distribution. Replace in child classes.
        
        Arguments:
            :np.ndarray q: mass ratio values
        
        Returns:
            :np.ndarray: probabilities
        """
        return np.ones(np.shape(q))/(self.bounds_q[1] - self.bounds_q[0])
        
    def redshift_distribution(self, z):
        """
        Uniform redshift distribution. Replace in child classes.
        
        Arguments:
            :np.ndarray z: redshift values
        
        Returns:
            :np.ndarray: probabilities
        """
        return np.ones(np.shape(z))/(self.bounds_z[1] - self.bounds_z[0])

    def luminosity_distance_distribution(self, d):
        """
        Uniform luminosity distance distribution. Replace in child classes.
        
        Arguments:
            :np.ndarray d: luminosity distance values
        
        Returns:
            :np.ndarray: probabilities
        """
        return np.ones(np.shape(d))/(self.bounds_d[1] - self.bounds_d[0])
    
    def sample_binary_parameters(self, n_samps, mass_pars, dist_par):
        """
        Produce a set of astrophysical binary parameters (m1, m2, z, w).
        
        Arguments:
            :int n_samps:   number of binaries
            :str mass_pars: set of parameters used to generate the masses ('m1m2', 'm1q')
        
        Return:
            :np.ndarray: primary masses
            :np.ndarray: secondary masses
            :np.ndarray: redshift
            :np.ndarray: w parameter
        """
        # Masses
        m1_temp = rejection_sampler(int(n_samps), self.mass_distribution, self.bounds_m)
        if mass_pars == 'm1m2':
            m2_temp = rejection_sampler(int(n_samps), self.mass_distribution, self.bounds_m)
            m1 = np.array([m1i if m1i > m2i else m2i for m1i, m2i in zip(m1_temp, m2_temp)])
            m2 = np.array([m2i if m1i > m2i else m1i for m1i, m2i in zip(m1_temp, m2_temp)])
        if mass_pars == 'm1q':
            q  = rejection_sampler(int(n_samps), self.mass_ratio_distribution, self.bounds_q)
            m1 = m1_temp
            m2 = q*m1
        # Redshift
        if dist_par == 'z':
            z = rejection_sampler(int(n_samps), self.redshift_distribution, self.bounds_z)
        if dist_par == 'DL':
            d = rejection_sampler(int(n_samps), self.luminosity_distance_distribution, self.bounds_d)
            z = np.array([_find_redshift(omega, di) for di in tqdm(d, desc = 'Converting to z')])
        # w
        w = w_from_interpolant(int(n_samps), self.n_det)
        return m1, m2, z, w

    def generate_binaries(self, n_obs):
        """
        Generate a catalog of existing binaries (both observable and unobservable) containing n_events observable binaries.
        A binary is observable if snr_obs > snr_cut
        
        Arguments:
            :int n_obs: number of observable binaries
        
        Returns:
            :np.ndarray: source-frame primary masses
            :np.ndarray: source-frame secondary masses
            :np.ndarray: detector-frame chirp masses
            :np.ndarray: symmetric mass ratios
            :np.ndarray: luminosity distances
            :np.ndarray: redshifts
            :np.ndarray: observed SNRs
        """
        observed = 0
        # Arrays
        m1      = np.array([])
        m2      = np.array([])
        Mc      = np.array([])
        eta     = np.array([])
        DL      = np.array([])
        z       = np.array([])
        w       = np.array([])
        snr_obs = np.array([])
        while observed < n_obs:
            m1_temp, m2_temp, z_temp, w_temp = self.sample_binary_parameters(n_obs, self.mass_pars, self.dist_par)
            m1z_temp          = m1_temp*(1+z_temp)
            m2z_temp          = m2_temp*(1+z_temp)
            Mc_temp, eta_temp = chirp_mass_eta(m1z_temp, m2z_temp)
            DL_temp           = omega.LuminosityDistance(z_temp)
            # Generate SNRs
            snr_opt_temp  = snr_optimal(m1_temp, m2_temp, z = z_temp, DL = DL_temp)
            snr_true_temp = w_temp*snr_opt_temp
            snr_obs_temp  = snr_sampler(snr_true_temp)
            observed += len(np.where(snr_obs_temp > self.snr_cut)[0])
            # Extend
            m1      = np.append(m1, m1_temp)
            m2      = np.append(m2, m2_temp)
            Mc      = np.append(Mc, Mc_temp)
            eta     = np.append(eta, eta_temp)
            DL      = np.append(DL, DL_temp)
            z       = np.append(z, z_temp)
            w       = np.append(w, w_temp)
            snr_obs = np.append(snr_obs, snr_obs_temp)
        # Trim arrays
        last    = np.where(snr_obs > self.snr_cut)[0][n_obs]
        m1      = m1[:last]
        m2      = m2[:last]
        Mc      = Mc[:last]
        eta     = eta[:last]
        DL      = DL[:last]
        z       = z[:last]
        w       = w[:last]
        snr_obs = snr_obs[:last]
        return m1, m2, Mc, eta, DL, z, w, snr_obs

    def generate_posteriors(self, n_events, n_samps = 1e4, out_folder = '.'):
        """
        Generate a set of single-event posterior distributions.
        
        Arguments:
            :int n_events:           number of events to generate
            :int n_samps:            number of samples per event
            :str or Path out_folder: folder where to save the posteriors
        """
        # Catalog identifier
        id       = np.random.randint(int(1e6))
        cat_name = 'MDC_'+str(id)
        # Prepare folders
        self.out_folder = Path(out_folder)
        self.events_folder = Path(self.out_folder, cat_name)
        if not self.events_folder.exists():
            self.events_folder.mkdir()
        m1, m2, Mc, eta, DL, z, w, snr_obs = self.generate_binaries(n_events)
        # Save true values
        if self.snr_cut > 0.:
            # Save full catalog
            save_event(m1, m2, Mc/(1+z), m2/m1, z, DL, snr_obs, name = cat_name + '_full', out_folder = self.out_folder)
        idx_obs = np.where(snr_obs > self.snr_cut)
        # Keeping only observed binaries
        m1      = m1[idx_obs]
        m2      = m2[idx_obs]
        Mc      = Mc[idx_obs]
        eta     = eta[idx_obs]
        DL      = DL[idx_obs]
        z       = z[idx_obs]
        w       = w[idx_obs]
        snr_obs = snr_obs[idx_obs]
        # Save observed catalog
        save_event(m1, m2, Mc/(1+z), m2/m1, z, DL, snr_obs, name = cat_name + '_obs', out_folder = self.out_folder)
        # Generate posteriors for each (Mc, eta, z, w)
        Mc_events  = Mc_sampler(Mc, snr_obs, int(n_samps))
        eta_events = eta_sampler(eta, snr_obs, int(n_samps))
        w_events   = w_sampler(w, snr_obs, int(n_samps))
        # Transform to (m1z, m2z, DL, w)
        m1z_events, m2z_events = component_masses(Mc_events, eta_events)
        DL_events = np.array([obs_distance(m1z_i, m2z_i, z_i, w_i, snr_i) for m1z_i, m2z_i, z_i, w_i, snr_i in tqdm(zip(m1z_events, m2z_events, z, w_events, snr_obs), desc = 'Sampling DL', total = n_events)])
        z_events = np.array([np.array([_find_redshift(omega, d) for d in DL_i]) for DL_i in tqdm(DL_events, desc = 'Converting to z', total = n_events)])
        # Final lists
        m1_final = []
        m2_final = []
        Mc_final = []
        DL_final = []
        z_final  = []
        q_final  = []
        # Reweight to account for prior
        for m1z_i, m2z_i, Mc_i, DL_i, z_i, w_i in tqdm(zip(m1z_events, m2z_events, Mc_events, DL_events, z_events, w_events), desc = 'Reweighting posteriors', total = n_events):
            p  = PE_prior(w_i, DL_i, n_det = self.n_det, volume = self.volume)*jacobian(m1z_i, m2z_i, DL_i, z_i, w_i)
            p /= p.sum()
            vals = np.random.uniform(size = len(p))*np.max(p)
            idx = np.where(p > vals)
            # Resampling
            m1_final.append(m1z_i[idx]/(1+z_i[idx]))
            m2_final.append(m2z_i[idx]/(1+z_i[idx]))
            Mc_final.append(Mc_i[idx]/(1+z_i[idx]))
            DL_final.append(DL_i[idx])
            z_final.append(z_i[idx])
            q_final.append(m2z_i[idx]/m1z_i[idx])
        # Save posteriors
        save_posteriors(m1_final, m2_final, Mc_final, q_final, z_final, DL_final, snr_obs, name = cat_name, out_folder = self.events_folder)
