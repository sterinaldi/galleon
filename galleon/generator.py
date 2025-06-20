import numpy as np
from tqdm import tqdm

from figaro.utils import rejection_sampler

from galleon.samplers import *
from galleon.utils import *
from galleon.settings import omega

mass_parameters_list = ['m1m2', 'm1q']

class Generator:
    def __init__(self, bounds_m  = [2,100],
                       bounds_z  = None,
                       bounds_d  = None,
                       bounds_q  = None,
                       n_det     = 'three',
                       mass_pars = 'm1q',
                       dist_par  = 'z',
                       snr_cut   = 10.,
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
            :double snr_cut: SNR for detectability. 10 is default for LVK, 0 (or negative values) removes the filter.
        
        Returns:
            :Generator: instance of Generator class
        """
        self.bounds_m = bounds_m
        if bounds_q is not None:
            self.bounds_q = bounds_q
        else:
            self.bounds_q = [0.1,1.]
        if bounds_z is not None:
            self.bounds_z = bounds_z
            self.bounds_d = [omega.LuminosityDistance(self.bounds_z[0]), omega.LuminosityDistance(self.bounds_z[1])]
        else:
            self.bounds_z = [0.001, 2.3]
        if bounds_d is not None:
            self.bounds_d = bounds_d
            self.bounds_z = [omega.Redshift(bounds_d[0]), omega.Redshift(bounds_d[1])]
        else:
            self.bounds_d = [omega.LuminosityDistance(self.bounds_z[0]), omega.LuminosityDistance(self.bounds_z[1])]
        
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
            p_m2 = self.mass_distribution(m2)
        if mass_pars == 'm1q':
            q  = rejection_sampler(int(n_samps), self.mass_ratio_distribution, self.bounds_q)
            m1 = m1_temp
            m2 = q*m1
            p_m2 = self.mass_ratio_distribution(q)/m1
        p_m1 = self.mass_distribution(m1)
        # Redshift
        if dist_par == 'z':
            z = rejection_sampler(int(n_samps), self.redshift_distribution, self.bounds_z)
            d = omega.LuminosityDistance(z)
            p_z = self.redshift_distribution(z)
            p_dl = p_z / omega.dDLdz(z)
        if dist_par == 'DL':
            d = rejection_sampler(int(n_samps), self.luminosity_distance_distribution, self.bounds_d)
            z = omega.Redshift(d)
            p_dl = self.luminosity_distance_distribution(d)
            p_z = p_dl*omega.dDLdz(z)
        # w
        w = w_from_interpolant(int(n_samps), self.n_det)
        return m1, m2, d, z, w, p_m1, p_m2, p_dl, p_z

    def generate_binaries(self, n_obs, sensitivity = False):
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
        q       = np.array([])
        DL      = np.array([])
        z       = np.array([])
        w       = np.array([])
        snr_obs = np.array([])
        p_m1    = np.array([])
        p_m2    = np.array([])
        p_dl    = np.array([])
        p_z     = np.array([])
        if not sensitivity:
            count = tqdm(total = n_obs, desc = 'Generating binaries')
        while observed < n_obs:
            m1_temp, m2_temp, DL_temp, z_temp, w_temp, p_m1_temp, p_m2_temp, p_dl_temp, p_z_temp = self.sample_binary_parameters(n_obs, self.mass_pars, self.dist_par)
            m1z_temp        = m1_temp*(1+z_temp)
            m2z_temp        = m2_temp*(1+z_temp)
            Mc_temp, q_temp = chirp_mass_q(m1z_temp, m2z_temp)
            # Generate SNRs
            snr_opt_temp  = snr_optimal(m1z_temp, m2z_temp, z = z_temp, DL = DL_temp, sensitivity = sensitivity)
            snr_true_temp = w_temp*snr_opt_temp
            snr_obs_temp  = np.abs(snr_sampler(snr_true_temp))
            acc = len(np.where(snr_obs_temp > self.snr_cut)[0])
            # Progress bar
            if not sensitivity:
                count.update(np.min((acc, n_obs-observed)))
                observed += acc
            # Extend
            m1      = np.append(m1, m1_temp)
            m2      = np.append(m2, m2_temp)
            Mc      = np.append(Mc, Mc_temp)
            q       = np.append(q, q_temp)
            DL      = np.append(DL, DL_temp)
            z       = np.append(z, z_temp)
            w       = np.append(w, w_temp)
            snr_obs = np.append(snr_obs, snr_obs_temp)
            p_m1    = np.append(p_m1, p_m1_temp)
            p_m2    = np.append(p_m2, p_m2_temp)
            p_dl    = np.append(p_dl, p_dl_temp)
            p_z     = np.append(p_z, p_z_temp)
            if sensitivity:
                break
        # Trim arrays
        if sensitivity:
            n_obs = acc
        last    = np.where(snr_obs > self.snr_cut)[0][n_obs-1]
        m1      = m1[:last+1]
        m2      = m2[:last+1]
        Mc      = Mc[:last+1]
        q       = q[:last+1]
        DL      = DL[:last+1]
        z       = z[:last+1]
        w       = w[:last+1]
        snr_obs = snr_obs[:last+1]
        p_m1    = p_m1[:last+1]
        p_m2    = p_m2[:last+1]
        p_dl    = p_dl[:last+1]
        p_z     = p_z[:last+1]
        return m1, m2, Mc, q, DL, z, w, snr_obs, p_m1, p_m2, p_dl, p_z

    def generate_posteriors(self, n_events, n_samps = 1e4, out_folder = '.', id = None):
        """
        Generate a set of single-event posterior distributions.
        
        Arguments:
            :int n_events:           number of events to generate
            :int n_samps:            number of samples per event
            :str or Path out_folder: folder where to save the posteriors
        """
        # Catalog identifier
        if id == None:
            id = np.random.randint(int(1e6))
        cat_name = 'MDC_'+str(id)
        # Prepare folders
        self.out_folder = Path(out_folder)
        self.events_folder = Path(self.out_folder, cat_name)
        if not self.events_folder.exists():
            self.events_folder.mkdir()
        m1, m2, Mc, q, DL, z, w, snr_obs, _, _, _, _ = self.generate_binaries(n_events)
        # Save true values
        if self.snr_cut > 0.:
            # Save full catalog
            save_event(m1, m2, Mc/(1+z), q, z, DL, snr_obs, name = cat_name + '_full', out_folder = self.out_folder)
        idx_obs = np.where(snr_obs > self.snr_cut)
        # Keeping only observed binaries
        m1      = m1[idx_obs]
        m2      = m2[idx_obs]
        Mc      = Mc[idx_obs]
        q       = q[idx_obs]
        DL      = DL[idx_obs]
        z       = z[idx_obs]
        w       = w[idx_obs]
        snr_obs = snr_obs[idx_obs]
        # Save observed catalog
        save_event(m1, m2, Mc/(1+z), q, z, DL, snr_obs, name = cat_name + '_obs', out_folder = self.out_folder)
        # Generate posteriors for each (Mc, q, z, w)
        Mc_events = Mc_sampler(Mc, snr_obs, int(n_samps))
        q_events  = q_sampler(q, snr_obs, int(n_samps))
        w_events  = w_sampler(w, snr_obs, int(n_samps))
        # Transform to (m1z, m2z, DL, w)
        m1z_events, m2z_events = component_masses(Mc_events, q_events)
        DL_and_snr  = np.atleast_2d([obs_distance_snr(m1z_i, m2z_i, w_i, snr_i) for m1z_i, m2z_i, w_i, snr_i in tqdm(zip(m1z_events, m2z_events, w_events, snr_obs), desc = 'Sampling DL and SNR', total = n_events)])
        DL_events   = DL_and_snr[:,0,:]
        snr_events  = DL_and_snr[:,1,:]
        DL_events[DL_events < 0.1] = 0.1
        snr_events  = np.atleast_2d(snr_events)
        DL_events   = np.atleast_2d(DL_events)
        Mc_events   = np.atleast_2d(Mc_events)
        q_events    = np.atleast_2d(q_events)
        w_events    = np.atleast_2d(w_events)
        m1z_events  = np.atleast_2d(m1z_events)
        m2z_events  = np.atleast_2d(m2z_events)
        z_events    = np.atleast_2d([omega.Redshift(DL_i) for DL_i in tqdm(DL_events, desc = 'Converting to z', total = n_events)])
        # Final lists
        m1_final = []
        m2_final = []
        Mc_final = []
        DL_final = []
        z_final  = []
        q_final  = []
        # Reweight to account for prior
        for m1z_i, m2z_i, Mc_i, DL_i, z_i, w_i, snr_i in tqdm(zip(m1z_events, m2z_events, Mc_events, DL_events, z_events, w_events, snr_events), desc = 'Reweighting posteriors', total = n_events):
            idx_snr = np.where(snr_i > 0.)[0]
#            p  = PE_prior(w_i[idx_snr], DL_i[idx_snr], n_det = self.n_det, volume = self.volume)*jacobian(m1z_i[idx_snr], m2z_i[idx_snr], DL_i[idx_snr], z_i[idx_snr], w_i[idx_snr], snr_i[idx_snr])
            p = np.ones(len(m1z_i))
            if len(p) == 0:
                # fix for irrealistic events
                p = np.ones(len(m1z_i))
                idx_snr = np.ones(len(m1z_i), dtype = int)
            p = p/p.sum()
            vals = np.random.uniform(size = len(p))*np.max(p)
            idx = np.where(p > vals)
            # Resampling
            m1_final.append((m1z_i[idx_snr])[idx]/(1+(z_i[idx_snr])[idx]))
            m2_final.append((m2z_i[idx_snr])[idx]/(1+(z_i[idx_snr])[idx]))
            Mc_final.append((Mc_i[idx_snr])[idx]/(1+(z_i[idx_snr])[idx]))
            DL_final.append((DL_i[idx_snr])[idx])
            z_final.append((z_i[idx_snr])[idx])
            q_final.append((m2z_i[idx_snr])[idx]/(m1z_i[idx_snr])[idx])
        # Save posteriors
        save_posteriors(m1_final, m2_final, Mc_final, q_final, z_final, DL_final, snr_obs, name = cat_name, out_folder = self.events_folder)

    def generate_sensitivity_estimate(self, n_injections, out_folder = '.', id = None):
        """
        Generate a set of mock injections.
        
        Arguments:
            :int n_injections:       total number of injections to generate
            :str or Path out_folder: folder where to save the posteriors
        """
        # Injection identifier
        if id == None:
            id = np.random.randint(int(1e6))
        inj_name = 'injections_'+str(id)
        # Prepare folder
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok = True)
        m1, m2, Mc, q, DL, z, w, snr_obs, p_m1, p_m2, p_dl, p_z = self.generate_binaries(n_injections, sensitivity = True)
        # Save true values
        idx_obs = np.where(snr_obs > self.snr_cut)
        # Keeping only observed binaries
        m1      = m1[idx_obs]
        m2      = m2[idx_obs]
        Mc      = Mc[idx_obs]
        q       = q[idx_obs]
        DL      = DL[idx_obs]
        z       = z[idx_obs]
        w       = w[idx_obs]
        snr_obs = snr_obs[idx_obs]
        p_m1    = p_m1[idx_obs]
        p_m2    = p_m2[idx_obs]
        p_dl    = p_dl[idx_obs]
        p_z     = p_z[idx_obs]
        save_injections(m1, m2, Mc, q, z, DL, snr_obs, p_m1, p_m2, p_z, p_dl, n_injections, name = inj_name, out_folder = out_folder)
    
    def generate_mock_catalogue(self, n_events, n_samps, n_injections, out_folder = '.', id = None):
        if id == None:
            id = np.random.randint(int(1e6))
        self.generate_posteriors(n_events   = int(n_events),
                                 n_samps    = int(n_samps),
                                 out_folder = out_folder,
                                 id         = id,
                                 )
        self.generate_sensitivity_estimate(n_injections = int(n_injections),
                                           out_folder   = out_folder,
                                           id           = id,
                                           )
