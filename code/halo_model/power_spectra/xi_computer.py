import pyccl as ccl
import copy
import treecorr

from halo_model.power_spectra.matter_power_baryons import MatterPowerBaryons
from halo_model.halos.base.profile.profile import CompositeProfile

from halo_model.config.config import Config


#import ingredient models

from halo_model.halos.base.clump_mass_func.models.giocoli2010_mod import ClumpMassGiocoli2010_mod
from halo_model.halos.base.clump_mass_func.models.giocoli2010 import ClumpMassGiocoli2010
from halo_model.halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc

from halo_model.halos.base.concentration.models.neto2007 import ConcentrationNeto2007
from halo_model.halos.base.concentration.models.pieri2009_clumps import ConcentrationPieri2009_clumps
from halo_model.halos.base.concentration.models.scaled_clumps import ConcentrationScaledClumps
from halo_model.halos.base.concentration.models.scaled_distribution import ConcentrationScaledDistribution

from halo_model.halos.base.mass_func.models.tinker2008_mod import MassFuncTinker2008_mod

from halo_model.halos.base.profile.models.nfw import ProfileNFW
from halo_model.halos.base.profile.models.stellar_truncated_powerlaw import ProfileStellarTruncatedPowerLaw
from halo_model.halos.base.profile.models.gas_cored_powerlaw import ProfileGasCoredPowerLaw

from halo_model.halos.base.bias.models.tinker2010 import BiasTinker2010
from halo_model.halos.base.r_vir.models.SO import R_virSO

from halo_model.halos.base.shmr.models.moster2013 import SHMRMoster2013
from halo_model.halos.base.shmr.models.moster2013_centrals import SHMRMoster2013Centrals
from halo_model.halos.base.shmr.models.Niemiec2022 import SMHRNiemiec2022

from halo_model.power_spectra.Pm_computer import Pm_computer



###################################################################################
# computing Pk_2d
###################################################################################



def compute_Pk_2d(cfg: Config)->ccl.Pk2D:
    
    # define model objects
    c_smooth = ConcentrationNeto2007(cfg)
    c_clump_base = ConcentrationPieri2009_clumps(cfg)
    c_clump = ConcentrationScaledClumps(cfg, c_smooth)
    c_distribution = ConcentrationScaledDistribution(cfg, c_smooth)
    
    mass_func = MassFuncTinker2008_mod(cfg)
    shmr = SHMRMoster2013(cfg, c_smooth)
    shmr_central = SHMRMoster2013Centrals(cfg, c_smooth)
    smooth_profile_dm = ProfileNFW(c_smooth, R_virSO(cfg))
    clump_profile_dm = ProfileNFW(c_clump, R_virSO(cfg))
    bias = BiasTinker2010(cfg)
    clump_distribution = ProfileNFW(c_distribution, R_virSO(cfg))
    
    # instantiate storage arrays
    z_grid = np.linspace(cfg.z_max, cfg.z_min, cfg.N_z) # from high redshift to low redshift so scale factor is monotonically increasing
    a_grid = 1/(1+z_grid)
    k_grid = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.N_k)
    Nz = len(z_grid)
    Nk = len(k_grid)
    P_tot_grid = np.zeros((Nz, Nk))   # shape (z, k)

    #computing model power spectra
    for i,z in enumerate(z_grid):
        cfg_local = copy.deepcopy(cfg)
        cfg_local.z = z
        gas_profile = ProfileGasCoredPowerLaw(cfg_local, shmr, c_smooth)
        stellar_profile = ProfileStellarTruncatedPowerLaw(cfg_local, c_smooth)

        print(f"computing power spectrum at redshift {z:.1f}")
        
        # compute CDM power spectrum
        clump_mass_func = ClumpMassGiocoli2010_mod(cfg_local)
        
        Pm = MatterPowerBaryons(cfg_local, 
                        mass_func=mass_func, 
                        shmr=shmr,
                        shmr_central=shmr_central,
                        smooth_profile_dm=smooth_profile_dm,
                        stellar_profile=stellar_profile, 
                        gas_profile=gas_profile,
                        bias=bias,
                        clump_mass_func=clump_mass_func, 
                        clump_profile_dm=clump_profile_dm, 
                        clump_distribution=clump_distribution)
            
        Pm_dict = Pm_computer(Pm, k_grid)
        P_tot_grid[i, :] = Pm_dict['P_tot']
        
    Pk_tot = ccl.Pk2D(
        a_arr=a_grid,
        lk_arr=np.log(k_grid),
        pk_arr=P_tot_grid,
        is_logp=False  
    )       
    return Pk_tot 


###################################################################################
# Pk_2d to Cls
###################################################################################

# converting to Cls requires a source distribution:

# Load in source distribution from flagship

from astropy.io import fits
import numpy as np

# fits_file = fits.util.get_testdata_filepath("23586.fits")
hdul = fits.open("/home/milan/Desktop/thesis/flagship_sim/25224.fits")
hdr = hdul[1].header
data = hdul[1].data

z = data['true_redshift_gal']
flux = data['euclid_vis']
mag = -2.5 * np.log10(flux) - 48.6


# create a mask in redshift and apply the luminosity cut:
z_min = 0.1
z_max = 2
mask = np.where((z > z_min) & (z < z_max) & (mag < 25))

z = z[mask]
ra = data['ra_gal'][mask]
dec = data['dec_gal'][mask]

kappa = data['kappa'][mask]
gamma1 = data['gamma1'][mask]
gamma2 = data['gamma2'][mask]
eps1 = data['eps1_gal'][mask]
eps2 = data['eps2_gal'][mask]

u_flux_abs = data['cfht_u_abs'][mask]
r_flux_abs = data['subaru_r_abs'][mask]

# now lets make cuts so we only select blue galaxies, for this we follow along 
# Euclid preparation:
# Calibrated intrinsic galaxy alignments in the Euclid Flagship simulation
# Euclid Collaboration: K. Hoffmann ....
# page 4 bottom right

h = 0.67
u_mag_abs = -2.5 * np.log10(u_flux_abs) - 48.6 + 5 * np.log10(h)
r_mag_abs = -2.5 * np.log10(r_flux_abs) - 48.6 + 5 * np.log10(h)

mask = (u_mag_abs - r_mag_abs < 1.32)

z = z[mask]
ra = ra[mask]
dec = dec[mask]

kappa = kappa[mask]
gamma1 = gamma1[mask]
gamma2 = gamma2[mask]
eps1 = eps1[mask]
eps2 = eps2[mask]


def Pk_2d_to_Cl(cfg:Config, Pk_2d:ccl.Pk2D):
    
    # setup ccl cosmology object
    H0 = cfg.cosmo.H0
    Om0 = cfg.cosmo.Om0
    Ob0 = cfg.cosmo.Ob0
    sigma8 = cfg.cosmo.sigma8
    ns = cfg.cosmo.ns
    cosmo = ccl.Cosmology(Omega_c = Om0, Omega_b=Ob0, h=0.01*H0, sigma8=sigma8, n_s=ns)
    
    # compute the Cls 
    # source distribution follows the sources from the flagship simulation
    z_edges = np.linspace(cfg.z_min, cfg.z_max, cfg.N_z)
    nzs = np.ndarray((z_edges.shape))

    for i in range(len(z_edges) - 1):
        nzs[i] = np.sum(np.where((z<z_edges[i+1]) & (z>z_edges[i])))
        nzs /= len(z)

    # define shear tracers
    tr = ccl.WeakLensingTracer(cosmo=cosmo, dndz=(z_edges, nzs))

    # compute C_ell
    ell = np.geomspace(cfg.l_min, cfg.l_max, cfg.N_l + 1)
    Cl = ccl.angular_cl(cosmo, tracer1=tr, tracer2=tr, ell=ell, p_of_k_a=Pk_2d)
    
    return Cl


def Cl_to_xi(cfg:Config, Cl):
    # setup ccl cosmology object
    H0 = cfg.cosmo.H0
    Om0 = cfg.cosmo.Om0
    Ob0 = cfg.cosmo.Ob0
    sigma8 = cfg.cosmo.sigma8
    ns = cfg.cosmo.ns
    cosmo = ccl.Cosmology(Omega_c = Om0, Omega_b=Ob0, h=0.01*H0, sigma8=sigma8, n_s=ns)
    
    #set theta range and ell space
    theta_deg = np.geomspace(cfg.theta_min/3600, cfg.theta_max/3600, cfg.N_theta)
    ell = np.geomspace(cfg.l_min, cfg.l_max, cfg.N_l + 1)
    
    xi_plus = ccl.correlation(cosmo, ell=ell, C_ell=Cl, theta=theta_deg, type='GG+', method='bessel')
    return xi_plus


def compute_Cl(cfg:Config):
    Pk_2d = compute_Pk_2d(cfg)
    Cl = Pk_2d_to_Cl(cfg, Pk_2d)
    return Cl


def compute_xi(cfg:Config):
    Cl = compute_Cl(cfg)
    xi_plus = Cl_to_xi(cfg, Cl)
    return xi_plus


###################################################################################
# determine error from flagship
###################################################################################


#setup config for treecorr

def covariance_xi(cfg:Config):
    """
    returns the covariance for xi for a euclid-like half-sky survey given a config.
    For the error just take sqrt.
    """
    #create theta bins
    nbins = cfg.N_theta

    min_sep = cfg.theta_min/3600
    max_sep = cfg.theta_max/3600

    config = {
        "nbins": nbins,
        "min_sep":  min_sep,
        "max_sep": max_sep,
        "sep_units": "degrees",
        "bin_type": "Log",

        # "min_rpar": PI_max,
        # "max_rpar": PI_max,

        "bin_slop": None, # None sets bin_slop = 0.1
        #"angle_slop": None,

        # "split_method": "mean", # How to split the cells in the tree when building the tree structure. 
        "metric": "Euclidean", #see metrics: https://rmjarvis.github.io/TreeCorr/_build/html/metric.html#metrics 
        "var_method": 'jackknife',
        "cross_patch_weight": 'simple',
    }
        
    npatch = 30

    IA_cov_delz = get_cov_delz(config, npatch=npatch)
    IA_cov_delz = IA_cov_delz[:cfg.N_theta, :cfg.N_theta]
    IA_cov_delz_halfsky = (1/140) * IA_cov_delz

    return IA_cov_delz_halfsky
    


# get covariance function

def get_cov_delz(config, npatch, method='jackknife'):
    mask1 = (z<2)   #(z < 0.9 - redshift_separation/2)
    mask2 = (z>0.1) #(z > 0.9 + redshift_separation/2)

    cat1 = treecorr.Catalog(
        ra=ra[mask1], dec=dec[mask1], ra_units='deg', dec_units='deg', 
        k=kappa[mask1], g1=eps1[mask1], g2=eps2[mask1], npatch=npatch
    )

    cat2 = treecorr.Catalog(
        ra=ra[mask2], dec=dec[mask2], ra_units='deg', dec_units='deg', 
        k=kappa[mask2], g1=eps1[mask2], g2=eps2[mask2], npatch=npatch
    )

    gg = treecorr.GGCorrelation(config)
    gg.process(cat1, cat2)
    cov = gg.estimate_cov(method=method, cross_patch_weight='simple')

    return cov
